"""
multi_asset_replay.py
---------------------
Phase 10: Multi-Asset LOB Replay engine.

Extends LOBReplay to load multiple symbols simultaneously and randomly
sample episodes across all of them during training. This forces the SAC
agent to learn a GENERALISED execution policy that adapts to:
    - Different price scales (BTC at $65000 vs BNB at $400)
    - Different volatility regimes
    - Different liquidity depths
    - Different tick sizes

Asset-aware observation: Each asset appends a 3D context vector:
    [avg_daily_volume_norm, volatility_rank, tick_size_norm]
This allows the agent to distinguish between assets without needing
separate models per symbol.

Usage:
    from src.market.multi_asset_replay import MultiAssetReplay

    replay = MultiAssetReplay("data/", symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    symbol, snapshots = replay.sample_episode(intervals=10)
"""
from __future__ import annotations

import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from src.market.lob_replay import LOBReplay, LOBSnapshot
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class AssetContext:
    """Normalised metadata for a single asset — injected into the observation."""
    symbol: str
    avg_spread_norm: float   # Avg spread / avg price (tick size proxy)
    volatility_rank: float   # Relative volatility rank among all assets [0, 1]
    liquidity_rank: float    # Relative liquidity rank [0, 1] (high = most liquid)


class MultiAssetReplay:
    """
    Loads multiple Binance tick CSVs and samples episodes across all assets.

    The agent sees a 3D asset context appended to its observation:
        [avg_spread_norm, volatility_rank, liquidity_rank]
    This context is FIXED per episode (it is an asset property, not a timestep property).

    Parameters
    ----------
    data_dir   : directory containing CSV files
    symbols    : list of symbol names (e.g., ['BTCUSDT', 'ETHUSDT'])
                 CSV files expected: {data_dir}/{symbol}_trades.csv
                 or: {data_dir}/{symbol}-trades-*.csv
    fmt        : 'binance' | 'ohlcv'
    vol_window : rolling vol window for LOBReplay
    """

    def __init__(
        self,
        data_dir: str,
        symbols: list[str],
        fmt: str = "binance",
        vol_window: int = 10,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._fmt = fmt
        self._replays: dict[str, LOBReplay] = {}
        self._contexts: dict[str, AssetContext] = {}

        log.info("MultiAssetReplay loading %d symbols...", len(symbols))
        for symbol in symbols:
            csv_path = self._find_csv(symbol)
            if csv_path is None:
                log.warning("No CSV found for %s in %s — skipping", symbol, data_dir)
                continue
            try:
                self._replays[symbol] = LOBReplay(str(csv_path), fmt=fmt, vol_window=vol_window)
                log.info("  Loaded %s: %d ticks", symbol, len(self._replays[symbol]))
            except Exception as e:
                log.warning("Failed to load %s: %s — skipping", symbol, e)

        if not self._replays:
            raise ValueError(
                f"No valid data files found in {data_dir}. "
                "Expected files like btcusdt_trades.csv or BTCUSDT-trades-2024-03-01.csv"
            )

        self._compute_contexts()
        log.info(
            "MultiAssetReplay ready | %d assets: %s",
            len(self._replays),
            list(self._replays.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._replays.keys())

    def __len__(self) -> int:
        return len(self._replays)

    def sample_episode(
        self,
        intervals: int = 10,
        rng: np.random.Generator | None = None,
    ) -> tuple[str, list[LOBSnapshot], AssetContext]:
        """
        Randomly select an asset and sample a contiguous episode from its data.

        Returns
        -------
        (symbol, snapshots, asset_context)
        """
        rng = rng or np.random.default_rng()
        symbol = rng.choice(self.symbols)
        replay = self._replays[symbol]
        start = replay.random_episode_start(intervals, rng)
        snapshots = replay.slice(start, intervals + 2)

        if len(snapshots) < intervals:
            snapshots = replay.slice(0, intervals + 2)

        return symbol, snapshots, self._contexts[symbol]

    def get_replay(self, symbol: str) -> LOBReplay:
        return self._replays[symbol]

    def get_context(self, symbol: str) -> AssetContext:
        return self._contexts[symbol]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_csv(self, symbol: str) -> Path | None:
        """Find a CSV file for the given symbol in data_dir."""
        symbol_lower = symbol.lower()
        candidates = [
            self._data_dir / f"{symbol_lower}_trades.csv",
            self._data_dir / f"{symbol.upper()}_trades.csv",
            self._data_dir / f"{symbol.upper()}-trades-*.csv",
            self._data_dir / f"{symbol_lower}.csv",
            self._data_dir / f"{symbol.upper()}.csv",
        ]
        # Exact matches first
        for c in candidates[:4]:
            if c.exists():
                return c
        # Glob for dated files
        glob_results = list(self._data_dir.glob(f"{symbol.upper()}-trades-*.csv"))
        if glob_results:
            return sorted(glob_results)[-1]  # Most recent
        return None

    def _compute_contexts(self) -> None:
        """Compute normalised asset context vectors for all loaded symbols."""
        spread_norms = {}
        volatilities = {}
        avg_volumes = {}

        for symbol, replay in self._replays.items():
            spreads = [s.spread / max(s.mid_price, 1e-8) for s in replay._snapshots[:500]]
            vols = [s.rolling_vol for s in replay._snapshots[:500]]
            volumes = [(s.bid_volume + s.ask_volume) for s in replay._snapshots[:500]]

            spread_norms[symbol] = float(np.mean(spreads)) if spreads else 0.0
            volatilities[symbol] = float(np.mean(vols)) if vols else 0.0
            avg_volumes[symbol] = float(np.mean(volumes)) if volumes else 1.0

        # Rank normalise across all assets
        vol_values = list(volatilities.values())
        liq_values = list(avg_volumes.values())
        n = len(vol_values)

        for symbol in self._replays:
            vol_rank = sorted(vol_values).index(volatilities[symbol]) / max(n - 1, 1)
            liq_rank = sorted(liq_values).index(avg_volumes[symbol]) / max(n - 1, 1)

            self._contexts[symbol] = AssetContext(
                symbol=symbol,
                avg_spread_norm=spread_norms[symbol],
                volatility_rank=float(vol_rank),
                liquidity_rank=float(liq_rank),
            )
            log.debug(
                "Context[%s]: spread_norm=%.5f, vol_rank=%.2f, liq_rank=%.2f",
                symbol, spread_norms[symbol], vol_rank, liq_rank,
            )
