"""
eval_v2.py
----------
Honest scorecard for a v2 model on the HELD-OUT split.

    python -m src.rl.eval_v2 --model sac_v2.zip --episodes 300

Every strategy is replayed on the IDENTICAL episodes (same ticks, same costs):
    SAC model | TWAP | VWAP | Almgren-Chriss (risk 1x and 10x) | Dump-everything

Cost is implementation shortfall in bps vs arrival mid (sell at bid, with market impact).
LOWER IS BETTER. "Edge vs TWAP" is positive when a strategy is cheaper than TWAP.

PASS needs the model to beat the BEST classic strategy by more than 2 standard errors
AND use market orders on fewer than 60% of ticks.
"""
import argparse
from dataclasses import replace
import logging
import warnings

import numpy as np
from stable_baselines3 import SAC

from src.api.explain import FeatureInfluence
from src.market.multi_asset_replay import MultiAssetReplay
from src.rl.baselines_v2 import ac_schedule, run_schedule, twap_schedule, vwap_schedule
from src.rl.execution_env_v2 import ExecutionEnvV2, V2Config

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sac_v2.zip")
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--split", default="eval")
    # stress test: make limit-order fills harder than in training
    ap.add_argument("--limit-vol-scale", type=float, default=1.0, help="e.g. 0.5 = half the fill volume")
    ap.add_argument("--limit-edge-bps", type=float, default=0.0, help="e.g. 1 = price must trade 1bp through our limit")
    ap.add_argument("--limit-ttl", type=int, default=3)
    a = ap.parse_args()

    model = SAC.load(a.model, device="cpu")
    mr = MultiAssetReplay(a.data_dir, [s.strip().upper() for s in a.symbols.split(",")])
    cfg = replace(V2Config(), limit_vol_scale=a.limit_vol_scale, limit_edge_bps=a.limit_edge_bps, limit_ttl=a.limit_ttl)
    if (a.limit_vol_scale, a.limit_edge_bps, a.limit_ttl) != (1.0, 0.0, 3):
        print(f"STRESS TEST: limit fill volume x{a.limit_vol_scale}, must trade {a.limit_edge_bps}bp through, ttl {a.limit_ttl}")
    env_m = ExecutionEnvV2(mr, config=cfg, split=a.split, random_seed=1)   # runs the model
    env_b = ExecutionEnvV2(mr, config=cfg, split=a.split, random_seed=2)   # runs the classic strategies

    names = ["SAC model", "TWAP", "VWAP", "Almgren-Chriss (risk 1x)", "Almgren-Chriss (risk 10x)", "Dump everything"]
    cost = {n: [] for n in names}
    mk = lm = hd = 0
    first_tick = []

    for i in range(a.episodes):
        seed = 700_000 + i
        # ---- the model
        o, _ = env_m.reset(seed=seed)
        d, k = False, 0
        while not d:
            act, _ = model.predict(o, deterministic=True)
            o, _, d, _, info = env_m.step(act)
            t = info["order_type"]
            mk += t == "market"
            lm += t == "limit"
            hd += t is None
            if k == 0:
                first_tick.append(env_m.total_shares - env_m.remaining_shares)
            k += 1
        cost["SAC model"].append(info["agent_bps"])
        cost["TWAP"].append(info["twap_bps"])          # env's own shadow TWAP on the same ticks
        cost["Dump everything"].append(info["dump_bps"])

        # ---- classic strategies on the identical episode
        env_b.reset(seed=seed)
        cost["VWAP"].append(run_schedule(env_b, vwap_schedule(env_b)))
        for label, risk in (("Almgren-Chriss (risk 1x)", 1.0), ("Almgren-Chriss (risk 10x)", 10.0)):
            env_b.reset(seed=seed)
            cost[label].append(run_schedule(env_b, ac_schedule(env_b, risk)))

    c = {n: np.array(v) for n, v in cost.items()}
    n_ep = a.episodes
    twap = c["TWAP"]

    print(f"\n== {a.model} on '{a.split}' split, {n_ep} episodes, identical ticks for every strategy ==")
    print(f"{'strategy':<28}{'cost (bps)':>11}{'edge vs TWAP':>16}{'cheaper than TWAP':>20}")
    for name in names:
        gap = twap - c[name]
        se = gap.std() / np.sqrt(n_ep)
        edge = "      (baseline)" if name == "TWAP" else f"{gap.mean():+7.2f} +/-{se:5.2f}"
        wins = "" if name == "TWAP" else f"{(gap > 0).mean() * 100:>16.0f}%"
        print(f"{name:<28}{c[name].mean():>+11.2f}{edge:>16}{wins:>20}")

    ticks = mk + lm + hd
    print(f"\nSAC order mix: market {mk / ticks * 100:.0f}% | limit {lm / ticks * 100:.0f}% | no order {hd / ticks * 100:.0f}%")
    print(f"SAC avg units sold on tick 1: {np.mean(first_tick):.0f} / {env_m.total_shares}")

    # what drives the model's decisions
    inf = FeatureInfluence(model, make_env=lambda: ExecutionEnvV2(mr, split=a.split, random_seed=int(np.random.randint(1_000_000))), warmup_episodes=20)
    g = {"clock": 0.0, "market": 0.0, "asset": 0.0}
    m = 0
    env2 = ExecutionEnvV2(mr, split=a.split, random_seed=77)
    for _ in range(40):
        o, _ = env2.reset()
        d = False
        while not d:
            for grp in inf.explain(o)["groups"]:
                g[grp["key"]] += grp["influence"]
            m += 1
            act, _ = model.predict(o, deterministic=True)
            o, _, d, _, _ = env2.step(act)
    print("SAC decision drivers: " + " | ".join(f"{k} {v / m * 100:.0f}%" for k, v in g.items()))

    # verdict: model vs the BEST classic strategy (lowest mean cost)
    classics = ["TWAP", "VWAP", "Almgren-Chriss (risk 1x)", "Almgren-Chriss (risk 10x)"]
    best = min(classics, key=lambda x: c[x].mean())
    gap = c[best] - c["SAC model"]
    se = gap.std() / np.sqrt(n_ep)
    ok = gap.mean() - 2 * se > 0 and mk / ticks < 0.6
    print(f"\nbest classic strategy here: {best} ({c[best].mean():+.2f} bps)")
    print(f"SAC vs {best}: {gap.mean():+.2f} +/- {se:.2f} bps (positive = SAC cheaper)")
    print(f"verdict: {'PASS' if ok else 'NOT YET'} (needs SAC cheaper than the best classic by > 2 std-err AND market orders < 60%)")


if __name__ == "__main__":
    main()
