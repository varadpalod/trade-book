# data/

Tick data for BTCUSDT, ETHUSDT, BNBUSDT: `<symbol>_trades.csv` in Binance format
(`id, price, qty, quote_qty, time, is_buyer_maker, is_best_match`).

- **Generated (default):** `python -m src.market.generate_microstructure_data`
  Simulated, with volatility clustering, order-flow persistence and flow price impact. Good for testing the pipeline, not a final result.
- **Real:** `python -m src.market.download_binance_data --symbol BTCUSDT --date 2024-03-01`
  (repeat per symbol/day), then name/merge the CSVs as `btcusdt_trades.csv` etc. Real data is the true test.

Old v1 data lives in `archive/old_data/`.
