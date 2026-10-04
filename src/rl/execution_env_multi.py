"""
execution_env_multi.py
----------------------
Phase 10: Multi-Asset Gymnasium environment.

Wraps MultiAssetReplay to train a single SAC agent that generalises
across multiple assets (BTC, ETH, BNB, and eventually NSE equities).

Key design:
    - Each episode randomly selects one asset from the loaded universe
    - The 3D AssetContext vector is appended to the observation
    - The agent learns to condition its execution style on the asset type
      (e.g., be more aggressive on illiquid assets, more passive on BTC)

Observation space (15D = 12D LOB + 3D asset context):
    0-11 : standard 12D LOB signals (from ExecutionEnvLO)
    12   : avg_spread_norm      — tick size proxy for this asset
    13   : volatility_rank      — 0=lowest vol, 1=highest vol
    14   : liquidity_rank       — 0=least liquid, 1=most liquid

Action space (3D, same as ExecutionEnvLO):
    [quantity_fraction, price_aggression, order_type_bias]
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.market.multi_asset_replay import MultiAssetReplay, AssetContext
from src.market.lob_replay import LOBSnapshot
from src.execution.fill_simulator import FillSimulator
from src.config.env_config import EnvConfig, DEFAULT_CONFIG
from src.logger import get_logger

log = get_logger(__name__)


class ExecutionEnvMulti(gym.Env):
    """
    Phase 10 Multi-Asset execution environment.

    Randomly samples from multiple assets each episode, injecting
    asset-specific context into the observation so the agent can
    generalise without separate models per symbol.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        multi_replay: MultiAssetReplay,
        config: EnvConfig | None = None,
        random_seed: int = 42,
        fill_model: str = "conservative",
        cancel_after_steps: int = 1,
    ) -> None:
        super().__init__()

        self._multi = multi_replay
        self._cfg = config if config is not None else DEFAULT_CONFIG
        self._rng = np.random.default_rng(random_seed)

        self.total_shares = self._cfg.total_shares
        self.intervals = self._cfg.intervals
        self._lambda_terminal = self._cfg.lambda_terminal

        self._fill_sim = FillSimulator(
            fill_model=fill_model,
            cancel_after_steps=cancel_after_steps,
        )

        # Episode state
        self.current_symbol: str = ""
        self.remaining_shares: int = self.total_shares
        self.filled_shares: int = 0
        self.current_step: int = 0
        self.arrival_price: float = 0.0
        self.total_shortfall: float = 0.0
        self._steps_since_last_fill: int = 0
        self._episode_snapshots: list[LOBSnapshot] = []
        self._current_snapshot: LOBSnapshot | None = None
        self._asset_context: AssetContext | None = None

        # 3D action (same as Phase 8)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 15D observation (12D LOB + 3D asset context)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        log.info(
            "ExecutionEnvMulti (Phase 10) | assets=%s, obs=15D, act=3D",
            self._multi.symbols,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random asset and episode
        symbol, snapshots, context = self._multi.sample_episode(
            intervals=self.intervals, rng=self._rng
        )
        self.current_symbol = symbol
        self._episode_snapshots = snapshots
        self._asset_context = context

        self.remaining_shares = self.total_shares
        self.filled_shares = 0
        self.current_step = 0
        self.total_shortfall = 0.0
        self._steps_since_last_fill = 0
        self._fill_sim.reset()

        self._current_snapshot = self._episode_snapshots[0]
        self.arrival_price = self._current_snapshot.mid_price

        log.debug("Multi Env reset | symbol=%s, arrival=%.4f", symbol, self.arrival_price)
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # Identical execution logic to ExecutionEnvLO
        qty_frac = float(np.clip(action[0], 0.0, 1.0))
        price_agg = float(np.clip(action[1], -1.0, 1.0))
        order_type = float(np.clip(action[2], 0.0, 1.0))

        snap = self._current_snapshot
        mid = snap.mid_price
        spread = snap.spread
        bid = snap.bid_price
        ask = snap.ask_price

        free_inventory = self.remaining_shares - self._fill_sim.get_pending_qty()
        shares_to_attempt = int(qty_frac * free_inventory)
        shares_to_attempt = min(shares_to_attempt, free_inventory)

        reward = 0.0
        info = {"symbol": self.current_symbol}

        if shares_to_attempt > 0:
            use_market_order = order_type > 0.5
            if use_market_order:
                execution_price = ask
                filled = shares_to_attempt
                self.remaining_shares -= filled
                self.filled_shares += filled
                self._steps_since_last_fill = 0
                shortfall = (
                    (self.arrival_price - execution_price) * filled
                    / max(self.arrival_price, 1e-8)
                )
                self.total_shortfall += shortfall
                reward -= shortfall
                info["order_type"] = "market"
            else:
                if price_agg <= 0:
                    limit_price = bid + (mid - bid) * (price_agg + 1.0)
                else:
                    limit_price = mid + (ask - mid) * price_agg
                self._fill_sim.post_limit_order(
                    price=limit_price, qty=shares_to_attempt,
                    side="sell", current_step=self.current_step,
                    ask_volume=snap.ask_volume,
                )
                info["order_type"] = "limit"
                info["limit_price"] = limit_price

        next_idx = min(self.current_step + 1, len(self._episode_snapshots) - 1)
        next_snap = self._episode_snapshots[next_idx]

        filled_qty, cancelled_qty = self._fill_sim.process_tick(
            current_step=self.current_step,
            mid_price=next_snap.mid_price,
            ask_price=next_snap.ask_price,
            bid_price=next_snap.bid_price,
            traded_volume=next_snap.ask_volume + next_snap.bid_volume,
            last_traded_price=next_snap.last_price,
        )

        if filled_qty > 0:
            self.remaining_shares -= filled_qty
            self.filled_shares += filled_qty
            self._steps_since_last_fill = 0
            exec_price = info.get("limit_price", mid)
            shortfall = (
                (self.arrival_price - exec_price) * filled_qty
                / max(self.arrival_price, 1e-8)
            )
            self.total_shortfall += shortfall
            reward -= shortfall

        if cancelled_qty > 0:
            reward -= 0.01 * cancelled_qty / max(self.total_shares, 1)

        time_used = (self.current_step + 1) / self.intervals
        reward -= self._cfg.reward_alpha * (
            (self.remaining_shares / self.total_shares) ** 2
        ) * time_used

        self.current_step += 1
        self._steps_since_last_fill += 1
        if self.current_step < len(self._episode_snapshots):
            self._current_snapshot = self._episode_snapshots[self.current_step]

        terminated = self.current_step >= self.intervals or self.remaining_shares == 0
        truncated = False

        if terminated and self.remaining_shares > 0:
            remaining = self.remaining_shares
            self._fill_sim.cancel_all()
            exec_price = self._current_snapshot.bid_price if self._current_snapshot else mid
            shortfall = (
                (self.arrival_price - exec_price) * remaining
                / max(self.arrival_price, 1e-8)
            )
            self.total_shortfall += shortfall
            reward -= self._lambda_terminal * abs(shortfall)
            self.remaining_shares = 0
            self.filled_shares += remaining

        info.update({
            "filled_qty": filled_qty,
            "remaining_shares": self.remaining_shares,
            "shortfall": self.total_shortfall,
            "fill_rate": self.filled_shares / self.total_shares,
        })
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        snap = self._current_snapshot
        ctx = self._asset_context
        if snap is None or ctx is None:
            return np.zeros(15, dtype=np.float32)

        price_change = (snap.mid_price - self.arrival_price) / max(self.arrival_price, 1e-8)
        steps_left = max((self.intervals - self.current_step) / self.intervals, 1e-6)
        urgency = (self.remaining_shares / self.total_shares) / steps_left
        spread_norm = snap.spread / max(snap.mid_price, 1e-8)
        total_vol = snap.bid_volume + snap.ask_volume + 1e-8

        return np.array([
            # 12D LOB signals
            self.current_step / self.intervals,
            self.remaining_shares / self.total_shares,
            price_change,
            urgency,
            snap.rolling_vol,
            spread_norm,
            snap.imbalance,
            snap.bid_volume / total_vol,
            snap.ask_volume / total_vol,
            self._fill_sim.get_pending_qty() / max(self.total_shares, 1),
            min(self._steps_since_last_fill / self.intervals, 1.0),
            self.filled_shares / max(self.total_shares, 1),
            # 3D asset context
            ctx.avg_spread_norm,
            ctx.volatility_rank,
            ctx.liquidity_rank,
        ], dtype=np.float32)
