"""
generate_microstructure_data.py
-------------------------------
Synthetic Binance-format trades WITH basic market-microstructure structure, for use
when real data is unavailable. Compared to generate_synthetic_data.py (pure GBM,
50/50 buy/sell, i.i.d. size) this adds the three stylised facts an execution agent
can actually react to:

  1. Volatility clustering   (log-vol follows an AR(1))
  2. Order-flow persistence  (aggressor sign repeats with prob. `persist`)
  3. Flow price impact       (aggressor sign pushes price: r = sigma*(z + kappa*sign))
     + volume rises with volatility

This is still SIMULATED. Real Binance data (download_binance_data.py) must be the
final test; this only proves the pipeline can learn market-reactive behaviour.

Usage:
    python -m src.market.generate_microstructure_data --symbol BTCUSDT --ticks 300000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRESETS = {
    "BTCUSDT": dict(price=50_000.0, sigma=0.0006, qty=0.01, seed=11),
    "ETHUSDT": dict(price=3_000.0, sigma=0.0009, qty=0.15, seed=22),
    "BNBUSDT": dict(price=400.0, sigma=0.0012, qty=1.0, seed=33),
}


def generate(symbol, ticks=300_000, persist=0.75, kappa=0.12, out_dir="data"):
    p = PRESETS[symbol]
    rng = np.random.default_rng(p["seed"])

    h = np.zeros(ticks)                      # log-vol state
    eps = rng.normal(size=ticks)
    for t in range(1, ticks):
        h[t] = 0.997 * h[t - 1] + 0.06 * eps[t]
    sigma_t = p["sigma"] * np.exp(h)

    sign = np.empty(ticks)
    sign[0] = 1.0
    flip = rng.random(ticks) > persist
    fresh = rng.choice([-1.0, 1.0], size=ticks)
    for t in range(1, ticks):
        sign[t] = fresh[t] if flip[t] else sign[t - 1]

    z = rng.normal(size=ticks)
    ret = sigma_t * (z + kappa * sign) - 0.5 * sigma_t ** 2
    price = p["price"] * np.exp(np.cumsum(ret))

    qty = np.exp(rng.normal(0, 1, ticks)) * p["qty"] * np.exp(0.5 * h)
    t_ms = 1_700_000_000_000 + np.cumsum(rng.exponential(100, ticks)).astype(np.int64)

    df = pd.DataFrame({
        "id": np.arange(ticks),
        "price": np.round(price, 4),
        "qty": np.maximum(np.round(qty, 6), 1e-6),
        "quote_qty": np.round(price * qty, 6),
        "time": t_ms,
        "is_buyer_maker": sign < 0,          # sell-initiated
        "is_best_match": True,
    })
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{symbol.lower()}_trades.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ALL")
    ap.add_argument("--ticks", type=int, default=300_000)
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()
    for s in (PRESETS if a.symbol == "ALL" else [a.symbol.upper()]):
        print("wrote", generate(s, a.ticks, out_dir=a.out_dir))
