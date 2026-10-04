# Data Directory

This directory stores historical tick/LOB data for training.

**Decision:** We use Binance (Crypto) tick data as our primary training data source.
Order book mechanics (CLOB, queue priority, bid/ask spread) are mathematically
identical to Indian Equities (NSE). Once the model is trained, it can be deployed
on Indian Equities by simply swapping the CSV.

---

## Primary Source: Binance Historical Trades (Free, 100M+ ticks/day)

Download from: https://data.binance.vision/

**Using the built-in downloader (recommended):**
```bash
# Downloads 1 day of BTC/USDT trades (~3-5M ticks)
python -m src.market.download_binance_data --symbol BTCUSDT --date 2024-03-01

# Download multiple days for better training
python -m src.market.download_binance_data --symbol BTCUSDT --date 2024-03-01
python -m src.market.download_binance_data --symbol ETHUSDT --date 2024-03-01
```

CSV columns (auto-detected, no header required):
`id, price, qty, quote_qty, time, is_buyer_maker, is_best_match`

**Train Phase 8 (Limit Order agent) on Binance data:**
```bash
python -m src.rl.train_sac_lob --data data/btcusdt_trades.csv --format binance --timesteps 200000
```

**Train Phase 10 (Multi-Asset) on multiple symbols:**
```bash
python -m src.rl.train_multi_asset --data-dir data/ --symbols BTCUSDT,ETHUSDT,BNBUSDT
```

---

## Indian Equities (NSE via Yahoo Finance — OHLCV, 1-min)

For macro-execution research on Indian markets (no microstructure signals):
```bash
python -m src.market.download_indian_data --ticker RELIANCE.NS --days 7
python -m src.rl.train_sac_lob --data data/RELIANCE.NS_1m.csv --format ohlcv
```

Limitations: No bid/ask spread, no queue depth, no imbalance signals.
Use only for macro Smart-VWAP type agents.

---

## Synthetic Data (Instant, No Download)

For fast prototyping and unit tests:
```bash
python -m src.market.generate_synthetic_data --ticks 50000 --output data/synthetic_trades.csv
```

---

## .gitignore

Large data files are excluded from git:
```
data/*.csv
data/*.zip
```
