"""
download_indian_data.py
-----------------------
Downloads free 1-minute intraday OHLCV data for Indian Equities (NSE)
using Yahoo Finance (yfinance).

This data can be directly plugged into our LOB-backed SAC training environment:
    python -m src.rl.train_sac_lob --data data/RELIANCE.NS_1m.csv --format ohlcv

Usage:
    python -m src.market.download_indian_data --ticker RELIANCE.NS --days 7
"""
import argparse
from pathlib import Path
import yfinance as yf
from src.logger import get_logger

log = get_logger(__name__)


def download_data(ticker: str, days: int = 7, output_dir: str = "data") -> Path:
    """
    Downloads intraday data and saves it to a CSV compatible with lob_replay.py.
    yfinance limits 1-minute data to the last 7 days.
    """
    log.info("Downloading %d days of 1-minute data for %s...", days, ticker)
    
    # 1-minute data is only available for the last 7 days on Yahoo Finance
    period = f"{days}d"
    if days > 7:
        log.warning("yfinance restricts 1m data to 7 days. Capping period to 7d.")
        period = "7d"

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval="1m")

    if df.empty:
        raise ValueError(f"No data found for {ticker}. Check if the ticker symbol is correct (e.g., RELIANCE.NS for NSE).")

    # Format for lob_replay (OHLCV)
    # yfinance returns DatetimeIndex; convert to Unix timestamp in seconds
    df.reset_index(inplace=True)
    
    # yfinance index column is usually "Datetime" or "Date"
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["timestamp"] = df[dt_col].astype("int64") // 10**9

    # Standardise column names to lowercase for the OHLCV loader
    df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    # Select only required columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    # Filter out rows with 0 volume
    df = df[df["volume"] > 0]

    out_path = Path(output_dir) / f"{ticker}_1m.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    log.info("Successfully saved %d rows to %s", len(df), out_path)
    print(f"✅ Data saved: {len(df)} 1-minute bars downloaded to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Indian Equity Intraday Data")
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Yahoo Finance ticker symbol (e.g., HDFCBANK.NS)")
    parser.add_argument("--days", type=int, default=7, help="Number of days of 1-minute data (max 7)")
    args = parser.parse_args()

    download_data(ticker=args.ticker, days=args.days)


if __name__ == "__main__":
    main()
