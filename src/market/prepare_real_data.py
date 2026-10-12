"""
prepare_real_data.py
--------------------
Turn raw Binance trade CSVs (headerless, millions of rows) into the files the project reads:
    data/<symbol>_trades.csv   (with header, capped at --max-rows, days concatenated in order)

Usage (PowerShell, from the project root):
    python -m src.market.prepare_real_data --symbol BTCUSDT --inputs raw\\BTCUSDT-trades-2024-03-01.csv raw\\BTCUSDT-trades-2024-03-02.csv
    python -m src.market.prepare_real_data --symbol ETHUSDT --inputs raw\\ETHUSDT-trades-2024-03-01.csv
    python -m src.market.prepare_real_data --symbol BNBUSDT --inputs raw\\BNBUSDT-trades-2024-03-01.csv

Refuses to overwrite existing data/ files unless --overwrite (move the generated ones first).
"""
import argparse
import csv
from pathlib import Path

HEADER = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]


def prepare(symbol, inputs, out_dir="data", max_rows=1_500_000, overwrite=False):
    out = Path(out_dir) / f"{symbol.lower()}_trades.csv"
    if out.exists() and not overwrite:
        raise SystemExit(f"{out} exists. Move it away (e.g. to data_generated/) or pass --overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(HEADER)
        for path in inputs:
            with open(path, newline="") as fi:
                for row in csv.reader(fi):
                    if not row or row[0].strip().lower() in ("id", "trade_id"):  # skip header if present
                        continue
                    if len(row) < 6:
                        continue
                    w.writerow(row[:7] if len(row) >= 7 else row[:6] + ["True"])
                    n += 1
                    if n >= max_rows:
                        break
            if n >= max_rows:
                break
    print(f"wrote {out}: {n:,} rows")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--max-rows", type=int, default=1_500_000)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    prepare(a.symbol.upper(), a.inputs, a.out_dir, a.max_rows, a.overwrite)
