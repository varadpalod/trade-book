"""
execution_env_v2.py
-------------------
Execution environment with REALISTIC COSTS. The original ExecutionEnvMulti is left
untouched; this is a separate env so old models/tests keep working.

What changed vs ExecutionEnvMulti (and why):
  1. Market SELLS fill at the BID (old env filled at the ask => dumping looked like profit).
  2. Market impact: temporary (this order) + permanent (lowers price for later ticks),
     both proportional to the fraction of the parent order, so dumping costs real money.
  3. Order size is tied to market volume: parent order = `participation` x horizon volume
     (volumes are rescaled so the parent order is always `total_shares` units).
  4. Limit orders rest up to `limit_ttl` ticks and fill at THEIR OWN price. Limit prices
     range from mid to ask (passive only; a sell resting at the bid would be a free
     impact-less market order).
  5. Reward = improvement over a TWAP replayed on the same ticks with the same cost model
     (mark-to-market, bps). Beating TWAP is the only way to score. No inventory penalty.
  6. Each step = `tick_stride` raw trades, horizon = `intervals` steps. Episodes always
     run the full horizon (early termination would forfeit TWAP-relative credit).
  7. Train/eval split by time within each asset (first `split_frac` train, rest eval).

Observation (15D, same layout as v1; market features rescaled to O(1)):
    0 time | 1 inventory left | 2 price move vs arrival (x1e3) | 3 urgency |
    4 rolling vol (x1e3) | 5 spread (x1e3) | 6 flow imbalance | 7 buy share | 8 sell share |
    9 resting qty | 10 ticks since fill | 11 inventory sold | 12-14 asset context

Action (3D): [qty_fraction, price_aggression, order_type_bias] as before.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.market.multi_asset_replay import MultiAssetReplay


@dataclass
class V2Config:
    total_shares: int = 1000
    intervals: int = 30
    tick_stride: int = 5
    participation: float = 0.10
    limit_ttl: int = 3
    impact_coef: float = 0.002     # temp impact: whole parent order at once = 20 bps
    perm_coef: float = 0.0005      # permanent impact: whole parent order = 5 bps
    split_frac: float = 0.7
    reward_scale: float = 10.0     # reward is (bps improvement vs TWAP) / reward_scale
    # stress-test knobs (defaults = unchanged behaviour)
    limit_vol_scale: float = 1.0   # <1 shrinks the volume available to fill our resting limits
    limit_edge_bps: float = 0.0    # price must trade THROUGH our limit by this many bps to fill


class _Shim:
    """Keeps server.py's `env._fill_sim.get_pending_qty()` working."""
    def __init__(self, env): self._e = env
    def get_pending_qty(self): return sum(o["qty"] for o in self._e._pending)


class ExecutionEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, multi_replay: MultiAssetReplay, config: V2Config | None = None,
                 random_seed: int = 42, split: str = "train") -> None:
        super().__init__()
        assert split in ("train", "eval")
        self._multi = multi_replay
        self.cfg = config or V2Config()
        self.split = split
        self._rng = np.random.default_rng(random_seed)

        self.total_shares = self.cfg.total_shares
        self.intervals = self.cfg.intervals
        self._fill_sim = _Shim(self)

        # per-asset mean window volume (train range) for volume scaling
        self._asset_vol = {}
        self._asset_sigma = {}
        for sym in self._multi.symbols:
            snaps = self._multi.get_replay(sym)._snapshots
            n = int(len(snaps) * self.cfg.split_frac)
            sample = snaps[:n:50]
            self._asset_vol[sym] = float(np.mean([s.bid_volume + s.ask_volume for s in sample])) + 1e-12
            # per-STEP return volatility (fraction of price), used by the Almgren-Chriss baseline
            self._asset_sigma[sym] = float(np.mean([s.rolling_vol for s in sample])) * np.sqrt(self.cfg.tick_stride)

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

        self.current_symbol = ""
        self.current_step = 0
        self.remaining_shares = self.total_shares
        self.filled_shares = 0
        self.arrival_price = 0.0
        self._episode_snapshots = []
        self._current_snapshot = None
        self._pending: list[dict] = []

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        sym = str(self._rng.choice(self._multi.symbols))
        replay = self._multi.get_replay(sym)
        n = len(replay)
        lo, hi = (0, int(n * c.split_frac)) if self.split == "train" else (int(n * c.split_frac), n)
        span = (c.intervals + 2) * c.tick_stride
        start = int(self._rng.integers(lo, max(lo + 1, hi - span)))
        self._episode_snapshots = replay.slice(start, span)[::c.tick_stride][: c.intervals + 2]
        self._ctx = self._multi.get_context(sym)

        # Baseline inputs (do not touch the rng, so training/eval episodes are unchanged):
        #  - VWAP volume profile from the steps JUST BEFORE this episode (no look-ahead)
        #  - per-step volatility for Almgren-Chriss
        hist = replay.slice(max(0, start - span), span)[::c.tick_stride][: c.intervals]
        hv = [h.bid_volume + h.ask_volume for h in hist]
        if len(hv) == c.intervals and sum(hv) > 0:
            self.hist_volume_profile = [v / sum(hv) for v in hv]
        else:
            self.hist_volume_profile = [1.0 / c.intervals] * c.intervals
        self.step_sigma = self._asset_sigma[sym]
        self.current_symbol = sym

        # volume rescale: parent order == participation x horizon volume == total_shares units
        step_vol_raw = self._asset_vol[sym] * c.tick_stride / 11.0
        raw_total = c.participation * c.intervals * step_vol_raw
        self._unit = self.total_shares / raw_total

        self.current_step = 0
        self.remaining_shares = self.total_shares
        self.filled_shares = 0
        self._sold_val = 0.0
        self._perm = 0.0
        self._pending = []
        self._since_fill = 0
        self._current_snapshot = self._episode_snapshots[0]
        self.arrival_price = self._current_snapshot.mid_price

        # TWAP shadow book (same costs, same ticks)
        self._tw_val = 0.0
        self._tw_qty = 0.0
        self._tw_perm = 0.0
        self._prev_gap = 0.0
        self.agent_bps = 0.0
        self.twap_bps = 0.0
        self.dump_bps = self._dump_bps()
        self.twap_bps_curve = self._twap_curve()
        return self._obs(), {}

    # ---------------------------------------------------------------- helpers
    def _dump_bps(self) -> float:
        s0 = self._episode_snapshots[0]
        px = s0.bid_price * (1 - self.cfg.impact_coef)
        return float((self.arrival_price - px) / self.arrival_price * 1e4)

    def _twap_curve(self) -> list[float]:
        c, a, N = self.cfg, self.arrival_price, self.total_shares
        q, val, perm, out = 0.0, 0.0, 0.0, []
        for k in range(c.intervals):
            frac = 1.0 / c.intervals
            px = self._episode_snapshots[k].bid_price * (1 - c.impact_coef * frac - perm)
            perm += c.perm_coef * frac
            q += N * frac
            val += N * frac * px
            out.append(float((a * q - val) / (a * q) * 1e4))
        return out

    def _mtm_bps(self, sold_qty, sold_val, remaining, mark_bid) -> float:
        a, N = self.arrival_price, self.total_shares
        return float(((a * sold_qty - sold_val) + remaining * (a - mark_bid)) / (a * N) * 1e4)

    # ------------------------------------------------------------------- step
    def step(self, action):
        c = self.cfg
        qty_frac = float(np.clip(action[0], 0.0, 1.0))
        price_agg = float(np.clip(action[1], -1.0, 1.0))
        use_market = float(np.clip(action[2], 0.0, 1.0)) > 0.5

        k = self.current_step
        snap = self._episode_snapshots[k]
        nxt = self._episode_snapshots[min(k + 1, len(self._episode_snapshots) - 1)]
        bid, ask, mid = snap.bid_price, snap.ask_price, snap.mid_price
        N = self.total_shares

        free = self.remaining_shares - self._fill_sim.get_pending_qty()
        attempt = int(qty_frac * free)
        info = {"symbol": self.current_symbol, "order_type": None, "fills": []}
        market_filled = limit_filled = cleanup = 0

        if attempt > 0:
            if use_market:
                frac = attempt / N
                px = bid * (1 - c.impact_coef * frac - self._perm)
                self._perm += c.perm_coef * frac
                self._book(attempt, px)
                market_filled = attempt
                info["fills"].append((attempt, px))
                info["order_type"] = "market"
            else:
                price = mid + (ask - mid) * (price_agg + 1.0) / 2.0
                ask_vol_u = snap.ask_volume * self._unit
                qf = min(0.9, ask_vol_u / max(ask_vol_u + attempt, 1.0))
                self._pending.append({"price": price, "qty": attempt, "posted": k, "qf": qf})
                info["order_type"] = "limit"
                info["limit_price"] = price

        # ---- resting limit orders vs next tick (conservative: must trade through our price)
        step_vol = (nxt.bid_volume + nxt.ask_volume) * self._unit * c.tick_stride / 11.0
        still = []
        for o in self._pending:
            if nxt.last_price * (1 - self._perm) >= o["price"] * (1 + c.limit_edge_bps / 1e4):
                avail = max(0.0, step_vol - o["qf"] * step_vol) * c.limit_vol_scale
                fq = int(min(o["qty"], avail))
                if fq > 0:
                    self._book(fq, o["price"])
                    o["qty"] -= fq
                    limit_filled += fq
                    info["fills"].append((fq, o["price"]))
            if o["qty"] > 0 and (k - o["posted"] + 1) < c.limit_ttl:
                still.append(o)
        self._pending = still

        if market_filled + limit_filled > 0:
            self._since_fill = 0
        else:
            self._since_fill += 1

        # ---- TWAP shadow
        frac = 1.0 / c.intervals
        tpx = bid * (1 - c.impact_coef * frac - self._tw_perm)
        self._tw_perm += c.perm_coef * frac
        self._tw_qty += N * frac
        self._tw_val += N * frac * tpx

        self.current_step += 1
        done = self.current_step >= c.intervals
        mark_bid = nxt.bid_price * (1 - self._perm)

        if done and self.remaining_shares > 0:           # forced liquidation WITH impact
            rem = self.remaining_shares
            self._pending = []
            px = nxt.bid_price * (1 - c.impact_coef * rem / N - self._perm)
            self._book(rem, px)
            cleanup = rem
            info["fills"].append((rem, px))
            mark_bid = nxt.bid_price

        agent_bps = self._mtm_bps(self.filled_shares, self._sold_val, self.remaining_shares, mark_bid)
        twap_rem = N - self._tw_qty
        twap_bps = self._mtm_bps(self._tw_qty, self._tw_val, twap_rem, nxt.bid_price * (1 - self._tw_perm))
        gap = twap_bps - agent_bps                        # >0 = agent cheaper than TWAP
        reward = (gap - self._prev_gap) / c.reward_scale
        self._prev_gap = gap
        self.agent_bps, self.twap_bps = agent_bps, twap_bps

        self._current_snapshot = self._episode_snapshots[min(self.current_step, len(self._episode_snapshots) - 1)]
        info.update({
            "filled_qty": limit_filled,
            "market_filled": market_filled,
            "cleanup": cleanup,
            "mid_price": mid,
            "spread": snap.spread,
            "remaining_shares": self.remaining_shares,
            "filled_shares": self.filled_shares,
            "shortfall": agent_bps / 1e4 * N,             # server's legacy field (units of shares)
            "fill_rate": self.filled_shares / N,
            "agent_bps": agent_bps, "twap_bps": twap_bps, "dump_bps": self.dump_bps,
        })
        return self._obs(), float(reward), done, False, info

    def _book(self, qty: int, px: float) -> None:
        self.remaining_shares -= qty
        self.filled_shares += qty
        self._sold_val += qty * px

    # -------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        s = self._current_snapshot
        if s is None:
            return np.zeros(15, dtype=np.float32)
        c, N, ctx = self.cfg, self.total_shares, self._ctx
        tot = s.bid_volume + s.ask_volume + 1e-12
        steps_left = max((c.intervals - self.current_step) / c.intervals, 1e-6)
        return np.array([
            self.current_step / c.intervals,
            self.remaining_shares / N,
            (s.mid_price - self.arrival_price) / self.arrival_price * 1e3,
            (self.remaining_shares / N) / steps_left,
            s.rolling_vol * 1e3,
            s.spread / s.mid_price * 1e3,
            s.imbalance,
            s.bid_volume / tot,
            s.ask_volume / tot,
            self._fill_sim.get_pending_qty() / N,
            min(self._since_fill / c.intervals, 1.0),
            self.filled_shares / N,
            ctx.avg_spread_norm * 1e3,
            ctx.volatility_rank,
            ctx.liquidity_rank,
        ], dtype=np.float32)
