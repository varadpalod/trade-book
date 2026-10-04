"""
execution_env_lob.py
--------------------
LOB-backed Gymnasium environment for optimal trade execution.

Unlike execution_env.py (which uses GBM price simulation + analytical
impact model), this environment:
    - Replays real historical tick data from LOBReplay
    - Market impact emerges from the data (no gamma/eta params needed)
    - Observation space includes real LOB microstructure signals
    - Each episode starts at a random position in the historical data

This is the Phase 4 environment for production-grade training.

Observation space (9D):
    0: time_progress       — current_step / intervals          ∈ [0, 1]
    1: inventory_fraction  — remaining / total                 ∈ [0, 1]
    2: price_change        — (P_t - P_0) / P_0                ∈ ℝ
    3: urgency             — inventory_fraction / time_left    ∈ [0, ∞)
    4: rolling_vol         — real tick-by-tick return vol      ∈ [0, ∞)
    5: spread_norm         — spread / mid_price                ∈ [0, ∞)
    6: imbalance           — (bid_vol - ask_vol) / total_vol   ∈ [-1, 1]
    7: bid_vol_norm        — bid_vol / (bid_vol + ask_vol)     ∈ [0, 1]
    8: ask_vol_norm        — ask_vol / (bid_vol + ask_vol)     ∈ [0, 1]

Action space:
    [0, 1] — fraction of remaining shares to sell this step
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.market.lob_replay import LOBReplay, LOBSnapshot
from src.config.env_config import EnvConfig, DEFAULT_CONFIG
from src.logger import get_logger

log = get_logger(__name__)


class ExecutionEnvLOB(gym.Env):
    """
    LOB-backed execution environment.

    Parameters
    ----------
    lob_replay   : pre-loaded LOBReplay instance
    config       : EnvConfig (uses total_shares, intervals, etc.)
    random_seed  : seed for episode start randomisation
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lob_replay: LOBReplay,
        config: EnvConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__()

        self._replay = lob_replay
        self._cfg = config if config is not None else DEFAULT_CONFIG
        self._rng = np.random.default_rng(random_seed)

        self.total_shares = self._cfg.total_shares
        self.intervals = self._cfg.intervals
        self.execution_horizon = self._cfg.execution_horizon

        # Reward shaping
        self._reward_alpha = self._cfg.reward_alpha
        self._reward_beta = self._cfg.reward_beta
        self._reward_scale = self._cfg.reward_scale
        self._lambda_terminal = self._cfg.lambda_terminal

        # Internal state — set in reset()
        self.remaining_shares: int = self.total_shares
        self.current_step: int = 0
        self.arrival_price: float = 0.0
        self.total_shortfall: float = 0.0
        self._episode_snapshots: list[LOBSnapshot] = []
        self._current_snapshot: LOBSnapshot | None = None

        # 9D observation space (real LOB signals)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        log.info(
            "ExecutionEnvLOB initialised | shares=%d, intervals=%d, ticks_available=%d",
            self.total_shares,
            self.intervals,
            len(self._replay),
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random starting position in the historical data
        start = self._replay.random_episode_start(
            episode_length=self.intervals,
            rng=self._rng,
        )
        self._episode_snapshots = self._replay.slice(start, self.intervals + 1)

        # Ensure we have enough snapshots
        if len(self._episode_snapshots) < self.intervals:
            # If near end of data, wrap around from start
            self._episode_snapshots = self._replay.slice(0, self.intervals + 1)

        self.remaining_shares = self.total_shares
        self.current_step = 0
        self.total_shortfall = 0.0

        self._current_snapshot = self._episode_snapshots[0]
        self.arrival_price = self._current_snapshot.mid_price

        log.debug(
            "LOB Env reset | start_step=%d, arrival_price=%.4f",
            start,
            self.arrival_price,
        )
        return self._get_observation(), {}

    def step(self, action):
        action = float(np.clip(action[0], 0.0, 1.0))
        shares_to_sell = int(action * self.remaining_shares)
        shares_to_sell = min(shares_to_sell, self.remaining_shares)
        self.remaining_shares -= shares_to_sell

        snap = self._current_snapshot
        mid = snap.mid_price

        # ------------------------------------------------------------------
        # LOB-based market impact (no gamma/eta needed)
        # Selling into the book: we cross the spread and move the ask
        # Impact = shares_to_sell / total_ask_volume * spread
        # This is a linear book-walking approximation
        # ------------------------------------------------------------------
        ask_depth = max(snap.ask_volume, 1.0)
        participation = shares_to_sell / ask_depth
        spread_impact = participation * snap.spread

        # Execution price = mid - half_spread - additional_spread_impact
        half_spread = snap.spread / 2
        execution_price = mid - half_spread - spread_impact

        # Scale-invariant implementation shortfall
        if self.arrival_price > 0:
            implementation_shortfall = (
                (self.arrival_price - execution_price) * shares_to_sell
            ) / self.arrival_price
        else:
            implementation_shortfall = 0.0

        self.total_shortfall += implementation_shortfall

        # Reward
        inventory_fraction = self.remaining_shares / self.total_shares
        inventory_penalty = self._reward_alpha * (inventory_fraction ** 2)
        impact_penalty = self._reward_beta * spread_impact

        reward = -implementation_shortfall
        reward -= inventory_penalty
        reward -= impact_penalty / self._reward_scale

        info = {
            "shares_sold": shares_to_sell,
            "execution_price": execution_price,
            "mid_price": mid,
            "spread": snap.spread,
            "spread_impact": spread_impact,
            "imbalance": snap.imbalance,
            "implementation_shortfall": implementation_shortfall,
            "shortfall": self.total_shortfall,
        }

        self.current_step += 1

        # Advance to next snapshot
        if self.current_step < len(self._episode_snapshots):
            self._current_snapshot = self._episode_snapshots[self.current_step]

        terminated = (
            self.current_step >= self.intervals or self.remaining_shares == 0
        )
        truncated = False

        # Terminal liquidation penalty
        if self.current_step >= self.intervals and self.remaining_shares > 0:
            snap = self._current_snapshot
            ask_depth = max(snap.ask_volume, 1.0)
            participation = self.remaining_shares / ask_depth
            spread_impact = participation * snap.spread
            execution_price = snap.mid_price - snap.spread / 2 - spread_impact
            if self.arrival_price > 0:
                terminal_shortfall = (
                    (self.arrival_price - execution_price) * self.remaining_shares
                ) / self.arrival_price
            else:
                terminal_shortfall = 0.0
            self.total_shortfall = terminal_shortfall
            reward -= self._lambda_terminal * abs(terminal_shortfall)
            self.remaining_shares = 0

        log.debug(
            "LOB Step %d | sold=%d, exec=%.4f, mid=%.4f, spread=%.5f, reward=%.5f",
            self.current_step, shares_to_sell, execution_price, mid,
            snap.spread, reward,
        )

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """9D observation with real LOB microstructure signals."""
        snap = self._current_snapshot
        if snap is None:
            return np.zeros(9, dtype=np.float32)

        price_change = (snap.mid_price - self.arrival_price) / max(
            self.arrival_price, 1e-8
        )

        steps_left = max(
            (self.intervals - self.current_step) / self.intervals, 1e-6
        )
        urgency = (self.remaining_shares / self.total_shares) / steps_left

        spread_norm = snap.spread / max(snap.mid_price, 1e-8)

        total_vol = snap.bid_volume + snap.ask_volume + 1e-8
        bid_vol_norm = snap.bid_volume / total_vol
        ask_vol_norm = snap.ask_volume / total_vol

        return np.array(
            [
                self.current_step / self.intervals,      # 0: time progress
                self.remaining_shares / self.total_shares,  # 1: inventory fraction
                price_change,                             # 2: price change
                urgency,                                  # 3: urgency
                snap.rolling_vol,                         # 4: rolling vol (real)
                spread_norm,                              # 5: normalised spread
                snap.imbalance,                           # 6: order book imbalance
                bid_vol_norm,                             # 7: bid volume fraction
                ask_vol_norm,                             # 8: ask volume fraction
            ],
            dtype=np.float32,
        )
