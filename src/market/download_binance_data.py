"""
download_binance_data.py
------------------------
Utility to download historical tick data from Binance Vision.
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path
from src.logger import get_logger

log = get_logger(__name__)

def download_binance_trades(symbol: str, date: str, output_dir: str = "data") -> Path:
    """
    Download a zipped CSV of tick-by-tick trades from Binance Vision.
    Format: https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-15.zip
    """
    symbol = symbol.upper()
    filename = f"{symbol}-trades-{date}.zip"
    url = f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{filename}"
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / filename
    
    log.info("Downloading %s from Binance Vision...", filename)
    print(f"Downloading from: {url}")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        log.error("Download failed: %s", e)
        print(f"\n❌ Error downloading data for {date}. Make sure the date is valid (e.g., '2024-01-15').")
        raise

    print(f"✅ Downloaded zip to {zip_path}")
    
    # Unzip it
    csv_filename = filename.replace(".zip", ".csv")
    csv_path = out_dir / csv_filename
    
    log.info("Unzipping %s...", filename)
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
        
    print(f"✅ Extracted CSV to {csv_path}")
    
    # Clean up zip
    zip_path.unlink()
    
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--date", type=str, default="2024-01-15", help="YYYY-MM-DD format")
    args = parser.parse_args()
    
    download_binance_trades(args.symbol, args.date)
