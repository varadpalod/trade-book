"""
lob_replay.py
-------------
Historical LOB (Limit Order Book) data replay engine.

Reads tick-by-tick CSV data and replays it into the MatchingEngine,
producing real market microstructure signals for RL training.

Supported CSV formats:
    1. Binance trades (default):
       columns: timestamp, price, qty, is_buyer_maker
    2. Generic OHLCV:
       columns: timestamp, open, high, low, close, volume
    3. LOBSTER format (equities):
       columns: time, type, order_id, size, price, direction
       (see data/README.md for download instructions)

Usage:
    from src.market.lob_replay import LOBReplay
    replay = LOBReplay("data/btcusdt_trades.csv", format="binance")
    snapshot = replay.get_snapshot(step=5)
    print(snapshot.mid_price, snapshot.spread, snapshot.imbalance)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.logger import get_logger

log = get_logger(__name__)

# Supported CSV formats
LOBFormat = Literal["binance", "ohlcv", "lobster"]


@dataclass
class LOBSnapshot:
    """
    A single point-in-time snapshot of LOB-derived market state.

    Attributes
    ----------
    timestamp      : original event timestamp (unix seconds)
    mid_price      : (best_bid + best_ask) / 2
    spread         : best_ask - best_bid
    bid_volume     : total quantity at best bid level
    ask_volume     : total quantity at best ask level
    imbalance      : (bid_vol - ask_vol) / (bid_vol + ask_vol)  ∈ [-1, 1]
    last_price     : last traded price
    rolling_vol    : rolling 10-tick return std
    """
    timestamp: float
    mid_price: float
    spread: float
    bid_volume: float
    ask_volume: float
    imbalance: float
    last_price: float
    rolling_vol: float

    @property
    def bid_price(self) -> float:
        return self.mid_price - self.spread / 2

    @property
    def ask_price(self) -> float:
        return self.mid_price + self.spread / 2


class LOBReplay:
    """
    Replays historical trade data as a sequence of LOBSnapshots.

    The replay engine does NOT require a full L2/L3 feed.
    It reconstructs approximate LOB signals from trade data using:
        - Rolling bid/ask estimation from price direction
        - Volume imbalance from buyer/seller initiated trades
        - Spread proxy from average trade tick size

    Parameters
    ----------
    filepath   : path to CSV file
    fmt        : data format ('binance', 'ohlcv', 'lobster')
    vol_window : window size for rolling volatility (number of ticks)
    """

    def __init__(
        self,
        filepath: str | Path,
        fmt: LOBFormat = "binance",
        vol_window: int = 10,
    ) -> None:
        self._path = Path(filepath)
        self._fmt = fmt
        self._vol_window = vol_window
        self._snapshots: list[LOBSnapshot] = []

        if not self._path.exists():
            raise FileNotFoundError(
                f"LOB data file not found: {self._path}\n"
                "See data/README.md for download instructions."
            )

        self._load()
        log.info(
            "LOBReplay loaded | file=%s, format=%s, ticks=%d",
            self._path.name,
            self._fmt,
            len(self._snapshots),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._snapshots)

    def get_snapshot(self, step: int) -> LOBSnapshot:
        """Return the LOB snapshot at index `step`."""
        if step < 0 or step >= len(self._snapshots):
            raise IndexError(f"step={step} out of range [0, {len(self._snapshots)})")
        return self._snapshots[step]

    def price_path(self, start: int = 0, end: int | None = None) -> list[float]:
        """Return the mid-price series as a list."""
        end = end or len(self._snapshots)
        return [s.mid_price for s in self._snapshots[start:end]]

    def slice(self, start: int, length: int) -> list[LOBSnapshot]:
        """Return a contiguous slice of `length` snapshots starting at `start`."""
        return self._snapshots[start : start + length]

    def random_episode_start(
        self,
        episode_length: int,
        rng: np.random.Generator | None = None,
    ) -> int:
        """
        Sample a valid random start index for an episode of given length.
        Used by execution_env_lob.py to pick random segments for training.
        """
        max_start = len(self._snapshots) - episode_length - 1
        if max_start <= 0:
            return 0
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, max_start))

    # ------------------------------------------------------------------
    # CSV loaders
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._fmt == "binance":
            self._load_binance()
        elif self._fmt == "ohlcv":
            self._load_ohlcv()
        elif self._fmt == "lobster":
            self._load_lobster()
        else:
            raise ValueError(f"Unknown format: {self._fmt}")

    def _load_binance(self) -> None:
        """
        Binance historical trade CSV format:
        id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
        or simpler: timestamp, price, qty, is_buyer_maker
        """
        rows = self._read_csv()
        prices: list[float] = []
        buyer_vols: list[float] = []
        seller_vols: list[float] = []
        timestamps: list[float] = []

        for row in rows:
            try:
                # Handle both column name variants
                ts = float(row.get("time", row.get("timestamp", 0)))
                price = float(row.get("price", 0))
                qty = float(row.get("qty", row.get("quantity", 0)))
                is_buyer_maker = str(row.get("is_buyer_maker", "false")).lower() in (
                    "true", "1", "yes"
                )
            except (ValueError, KeyError):
                continue

            if price <= 0 or qty <= 0:
                continue

            prices.append(price)
            timestamps.append(ts / 1000.0 if ts > 1e10 else ts)  # ms -> s

            if is_buyer_maker:
                # Buyer is maker means sell-initiated trade
                seller_vols.append(qty)
                buyer_vols.append(0.0)
            else:
                buyer_vols.append(qty)
                seller_vols.append(0.0)

        self._build_snapshots(prices, timestamps, buyer_vols, seller_vols)

    def _load_ohlcv(self) -> None:
        """Generic OHLCV format: timestamp, open, high, low, close, volume"""
        rows = self._read_csv()
        prices: list[float] = []
        buyer_vols: list[float] = []
        seller_vols: list[float] = []
        timestamps: list[float] = []

        for row in rows:
            try:
                ts = float(row.get("timestamp", row.get("time", 0)))
                close = float(row.get("close", 0))
                volume = float(row.get("volume", 0))
                open_ = float(row.get("open", close))
            except (ValueError, KeyError):
                continue

            if close <= 0:
                continue

            prices.append(close)
            timestamps.append(ts)
            # Approximate buy/sell split from candle direction
            if close >= open_:
                buyer_vols.append(volume * 0.6)
                seller_vols.append(volume * 0.4)
            else:
                buyer_vols.append(volume * 0.4)
                seller_vols.append(volume * 0.6)

        self._build_snapshots(prices, timestamps, buyer_vols, seller_vols)

    def _load_lobster(self) -> None:
        """
        LOBSTER message file format.
        Columns: time, type, order_id, size, price, direction
        type 4 = execution of a visible limit order
        direction: -1=sell, 1=buy
        """
        rows = self._read_csv()
        prices: list[float] = []
        buyer_vols: list[float] = []
        seller_vols: list[float] = []
        timestamps: list[float] = []

        for row in rows:
            try:
                msg_type = int(row.get("type", 0))
                if msg_type != 4:  # Only trade executions
                    continue
                ts = float(row.get("time", 0))
                size = float(row.get("size", 0))
                price = float(row.get("price", 0)) / 10_000.0  # LOBSTER uses 1/10000
                direction = int(row.get("direction", 1))
            except (ValueError, KeyError):
                continue

            if price <= 0 or size <= 0:
                continue

            prices.append(price)
            timestamps.append(ts)
            if direction == 1:
                buyer_vols.append(size)
                seller_vols.append(0.0)
            else:
                buyer_vols.append(0.0)
                seller_vols.append(size)

        self._build_snapshots(prices, timestamps, buyer_vols, seller_vols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_snapshots(
        self,
        prices: list[float],
        timestamps: list[float],
        buyer_vols: list[float],
        seller_vols: list[float],
    ) -> None:
        """Convert raw price/volume arrays into LOBSnapshot objects."""
        if not prices:
            raise ValueError("No valid ticks loaded from CSV.")

        returns = []
        for i in range(1, len(prices)):
            r = (prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-8)
            returns.append(r)

        # Rolling aggregation windows for bid/ask volume
        window = self._vol_window

        for i, price in enumerate(prices):
            # Rolling vol
            ret_window = returns[max(0, i - window) : i]
            rolling_vol = float(np.std(ret_window)) if len(ret_window) >= 2 else 0.0

            # Rolling bid/ask volume windows
            bv = sum(buyer_vols[max(0, i - window) : i + 1])
            sv = sum(seller_vols[max(0, i - window) : i + 1])
            total_vol = bv + sv

            imbalance = (bv - sv) / max(total_vol, 1e-8)

            # Spread proxy: proportional to rolling vol * price
            spread_proxy = max(rolling_vol * price * 2, price * 0.0001)

            snapshot = LOBSnapshot(
                timestamp=timestamps[i] if i < len(timestamps) else float(i),
                mid_price=price,
                spread=spread_proxy,
                bid_volume=bv,
                ask_volume=sv,
                imbalance=imbalance,
                last_price=price,
                rolling_vol=rolling_vol,
            )
            self._snapshots.append(snapshot)

    def _read_csv(self) -> list[dict]:
        """Read CSV with auto-detection of headers."""
        rows = []
        with open(self._path, newline="", encoding="utf-8") as f:
            # Sniff delimiter
            sample = f.read(2048)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel  # fallback

            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                # Strip whitespace from keys
                rows.append({k.strip(): v.strip() for k, v in row.items()})

        log.debug("Read %d rows from %s", len(rows), self._path.name)
        return rows
