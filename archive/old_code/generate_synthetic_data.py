"""
generate_synthetic_data.py
--------------------------
Generates a synthetic Binance-format trade CSV for testing
the LOB replay engine without downloading real data.

Simulates a realistic crypto tick stream using GBM + Poisson
trade arrival process.

Usage:
    python -m src.market.generate_synthetic_data
    python -m src.market.generate_synthetic_data --ticks 50000 --output data/synthetic_trades.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


def generate(
    ticks: int = 10_000,
    initial_price: float = 50_000.0,
    sigma_per_tick: float = 0.001,
    output: str = "data/synthetic_trades.csv",
    seed: int = 42,
) -> Path:
    """
    Generate a synthetic Binance-format trade CSV.

    Parameters
    ----------
    ticks           : number of trade events to generate
    initial_price   : starting price
    sigma_per_tick  : per-tick volatility
    output          : output file path
    seed            : random seed
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    price = initial_price
    timestamp_ms = 1_700_000_000_000  # synthetic start time (ms)
    tick_interval_ms = 100  # ~10 trades/second

    log.info(
        "Generating %d synthetic ticks | price=%.2f, sigma=%.4f",
        ticks, initial_price, sigma_per_tick,
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"])

        for i in range(ticks):
            # GBM price update
            z = rng.normal()
            price = price * np.exp(-0.5 * sigma_per_tick ** 2 + sigma_per_tick * z)
            price = max(price, 1.0)

            # Random trade size (log-normal distribution)
            qty = float(np.exp(rng.normal(0, 1)) * 0.01)
            qty = max(round(qty, 6), 0.000001)
            quote_qty = round(price * qty, 6)

            # Buyer maker: True = sell-initiated, False = buy-initiated
            is_buyer_maker = bool(rng.random() < 0.5)

            timestamp_ms += int(rng.exponential(tick_interval_ms))

            writer.writerow([
                i,
                round(price, 2),
                qty,
                quote_qty,
                timestamp_ms,
                is_buyer_maker,
                True,
            ])

    log.info("Synthetic data written to %s", output_path)
    print(f"Generated {ticks} ticks -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Binance trade CSV")
    parser.add_argument("--ticks", type=int, default=10_000)
    parser.add_argument("--price", type=float, default=50_000.0)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--output", type=str, default="data/synthetic_trades.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        ticks=args.ticks,
        initial_price=args.price,
        sigma_per_tick=args.sigma,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
