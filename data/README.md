# Data Directory

This directory stores historical tick/LOB data for Phase 4 training.

---

## Option 1 — Binance Historical Trades (Free, Recommended for Crypto)

Download from: https://data.binance.vision/

```bash
# Example: BTCUSDT trades for 2024-01-15
wget https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-15.zip
unzip BTCUSDT-trades-2024-01-15.zip -d data/

# Rename for clarity
mv data/BTCUSDT-trades-2024-01-15.csv data/btcusdt_trades.csv
```

CSV columns: `id, price, qty, quote_qty, time, is_buyer_maker, is_best_match`

Then train:
```bash
python -m src.rl.train_sac_lob --data data/btcusdt_trades.csv --format binance
```

---

## Option 2 — LOBSTER (Equities, Academic Use)

Download from: https://lobsterdata.com/

LOBSTER provides L2 order book snapshots and message files for NASDAQ stocks.

Files needed: `<STOCK>_<DATE>_<START>_<END>_message_<LEVELS>.csv`

CSV columns: `time, type, order_id, size, price, direction`

Then train:
```bash
python -m src.rl.train_sac_lob --data data/AAPL_2023-01-01_34200000_57600000_message_10.csv --format lobster
```

---

## Option 3 — Generic OHLCV (Yahoo Finance, etc.)

Any OHLCV CSV with columns: `timestamp, open, high, low, close, volume`

```bash
python -m src.rl.train_sac_lob --data data/AAPL_ohlcv.csv --format ohlcv
```

---

## Generating a Synthetic Dataset (for testing without real data)

```bash
python -m src.market.generate_synthetic_data --ticks 10000 --output data/synthetic_trades.csv
```

---

## Option 4 — Indian Equities (NSE via Yahoo Finance)

For Indian markets, you can download free 1-minute OHLCV data using the provided script.

```bash
# Download 7 days of 1-minute data for Reliance Industries
python -m src.market.download_indian_data --ticker RELIANCE.NS

# Or for HDFC Bank
python -m src.market.download_indian_data --ticker HDFCBANK.NS
```

Then train the agent on the downloaded data:
```bash
python -m src.rl.train_sac_lob --data data/RELIANCE.NS_1m.csv --format ohlcv
```

---

## .gitignore

Large data files are excluded from git. Add to `.gitignore`:
```
data/*.csv
data/*.zip
```
