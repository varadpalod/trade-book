"""
explain.py
----------
Transparency helpers for the live dashboard.

Answers three questions on every tick:
    1. WHAT did the model decide?          -> describe_decision()
    2. WHY (which inputs moved it)?        -> FeatureInfluence.explain()
    3. Is that better than a dumb rule?    -> realistic_cost_bps() + baseline_*()

Cost convention used here (differs from the env's own `shortfall`):
    A SELL order always hits the BID. Positive bps = we sold below arrival mid
    (a cost). The env's internal shortfall credits market sells at the ASK, which
    books the half-spread as profit; that is why the old "Spread Capture" card
    always looked like outperformance.
"""
from __future__ import annotations

import numpy as np

# (key, human label, group)
OBS_SPEC = [
    ("time",       "Time elapsed",                 "clock"),
    ("remaining",  "Inventory left to sell",       "clock"),
    ("price_chg",  "Price move vs arrival",        "market"),
    ("urgency",    "Urgency (inventory / time)",   "clock"),
    ("vol",        "Volatility",                   "market"),
    ("spread",     "Bid-ask spread",               "market"),
    ("imbalance",  "Order-flow imbalance",         "market"),
    ("buy_share",  "Buy-initiated volume share",   "market"),
    ("sell_share", "Sell-initiated volume share",  "market"),
    ("pending",    "Resting limit qty",            "clock"),
    ("since_fill", "Ticks since last fill",        "clock"),
    ("filled",     "Inventory already sold",       "clock"),
    ("ctx_spread", "Asset avg spread",             "asset"),
    ("ctx_vol",    "Asset volatility rank",        "asset"),
    ("ctx_liq",    "Asset liquidity rank",         "asset"),
]
GROUP_LABELS = {
    "clock": "Clock & inventory",
    "market": "Live market signals",
    "asset": "Which asset it is",
}


class FeatureInfluence:
    """
    Local sensitivity of the policy to each of its 15 inputs.

    For every input j we nudge it by +/- one typical standard deviation (measured
    from the model's own rollouts) and measure how far the action vector moves.
    Influence is normalised to a share (sums to 1.0) so the UI can say, e.g.,
    "64% of this decision came from the clock and inventory".
    """

    def __init__(self, model, make_env, warmup_episodes: int = 30) -> None:
        self._model = model
        obs_rows = []
        for _ in range(warmup_episodes):
            env = make_env()
            obs, _ = env.reset()
            done = False
            while not done:
                obs_rows.append(obs.copy())
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
        self._sd = np.asarray(obs_rows).std(axis=0) + 1e-9

    def explain(self, obs: np.ndarray) -> dict:
        n = len(OBS_SPEC)
        batch = np.repeat(obs[None, :], 2 * n, axis=0)
        for j in range(n):
            batch[2 * j, j] += self._sd[j]
            batch[2 * j + 1, j] -= self._sd[j]
        acts, _ = self._model.predict(batch, deterministic=True)
        raw = np.array([np.abs(acts[2 * j] - acts[2 * j + 1]).sum() / 2 for j in range(n)])
        share = raw / max(raw.sum(), 1e-12)

        features = [
            {
                "key": key,
                "label": label,
                "group": group,
                "value": float(obs[j]),
                "influence": float(share[j]),
            }
            for j, (key, label, group) in enumerate(OBS_SPEC)
        ]
        groups = {g: 0.0 for g in GROUP_LABELS}
        for f in features:
            groups[f["group"]] += f["influence"]
        return {
            "features": sorted(features, key=lambda f: -f["influence"]),
            "groups": [
                {"key": g, "label": GROUP_LABELS[g], "influence": v} for g, v in groups.items()
            ],
        }


def describe_decision(action, shares_attempted: int, free_inventory: int) -> dict:
    """Plain-English reading of the 3D action [qty_fraction, price_aggression, order_type_bias]."""
    qty_frac = float(np.clip(action[0], 0.0, 1.0))
    price_agg = float(np.clip(action[1], -1.0, 1.0))
    type_bias = float(np.clip(action[2], 0.0, 1.0))
    is_market = type_bias > 0.5

    if shares_attempted <= 0:
        headline = "HOLD - no order this tick"
        kind = "hold"
    elif is_market:
        headline = f"MARKET SELL {shares_attempted:,} - cross the spread now"
        kind = "market"
    else:
        where = "passively, near the low end of its range" if price_agg <= 0 else "patiently, near the high end of its range"
        headline = f"LIMIT SELL {shares_attempted:,} - rest {where}"
        kind = "limit"

    return {
        "kind": kind,
        "headline": headline,
        "qty_frac": qty_frac,
        "price_agg": price_agg,
        "type_bias": type_bias,
        "shares_attempted": int(shares_attempted),
        "free_inventory": int(free_inventory),
    }


def realistic_cost_bps(arrival: float, fills: list[tuple[float, float]]) -> float | None:
    """Implementation shortfall in bps vs arrival mid. fills = [(qty, price)]. +ve = cost."""
    qty = sum(q for q, _ in fills)
    if qty <= 0 or arrival <= 0:
        return None
    avg_px = sum(q * p for q, p in fills) / qty
    return float((arrival - avg_px) / arrival * 1e4)


def baseline_curves(snapshots, arrival: float, total: int, intervals: int) -> dict:
    """
    Running average-cost curves (bps, sell-at-bid) for two dumb rules on the SAME ticks:
        twap : total/intervals every tick
        dump : everything on tick 0
    """
    twap_fills: list[tuple[float, float]] = []
    twap = []
    for i in range(intervals):
        twap_fills.append((total / intervals, snapshots[min(i, len(snapshots) - 1)].bid_price))
        twap.append(realistic_cost_bps(arrival, twap_fills))
    dump_cost = realistic_cost_bps(arrival, [(total, snapshots[0].bid_price)])
    return {"twap": twap, "dump": [dump_cost] * intervals}
