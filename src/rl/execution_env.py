"""
execution_env.py
----------------
Gymnasium environment for optimal trade execution.

Phase 1 changes:
    - All magic numbers replaced with EnvConfig values
    - Fixed typo: permenant_impact -> permanent_impact
    - Added structured logging
    - Observation space kept at 4D (expanded to 7D in Phase 2)

Observation space (4D):
    [time_progress, inventory_fraction, price_change, urgency]

Action space:
    [0, 1] — fraction of remaining shares to sell this step
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.config.env_config import EnvConfig, DEFAULT_CONFIG
from src.execution.price_process import PriceProcess
from src.execution.market_impact import MarketImpact
from src.logger import get_logger

log = get_logger(__name__)


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, random_seed: int = 42) -> None:
        super().__init__()

        # Accept a config or fall back to defaults
        cfg = config if config is not None else DEFAULT_CONFIG

        self.total_shares = cfg.total_shares
        self.intervals = cfg.intervals
        self.initial_price = cfg.initial_price
        self.execution_horizon = cfg.execution_horizon

        self.sigma = cfg.sigma
        self.eta = cfg.eta
        self.gamma = cfg.gamma

        # Reward shaping coefficients (from config, no more magic numbers)
        self._reward_alpha = cfg.reward_alpha
        self._reward_beta = cfg.reward_beta
        self._reward_scale = cfg.reward_scale
        self._lambda_inventory = cfg.lambda_inventory
        self._lambda_terminal = cfg.lambda_terminal

        # Internal state
        self.remaining_shares: int = self.total_shares
        self.current_price: float = self.initial_price
        self.current_step: int = 0
        self.arrival_price: float = self.initial_price
        self.total_shortfall: float = 0.0
        self.market_prices: list[float] = []

        self.price_process = PriceProcess(random_seed=random_seed)
        self.market_impact = MarketImpact(eta=self.eta, gamma=self.gamma)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        log.info(
            "ExecutionEnv initialised | shares=%d, intervals=%d, eta=%.2e, gamma=%.2e",
            self.total_shares,
            self.intervals,
            self.eta,
            self.gamma,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.price_process._rng = np.random.default_rng(seed)

        self.remaining_shares = self.total_shares
        self.current_price = self.initial_price
        self.current_step = 0
        self.total_shortfall = 0.0

        tau = self.execution_horizon / self.intervals
        self.market_prices = self.price_process.generate_path(
            initial_price=self.initial_price,
            sigma=self.sigma,
            intervals=self.intervals,
            tau=tau,
        )
        self.current_price = self.market_prices[0]
        self.arrival_price = self.current_price

        log.debug("Env reset | arrival_price=%.4f", self.arrival_price)
        return self._get_observation(), {}

    def step(self, action):
        action = float(np.clip(action[0], 0.0, 1.0))
        shares_to_sell = int(action * self.remaining_shares)
        shares_to_sell = min(shares_to_sell, self.remaining_shares)
        self.remaining_shares -= shares_to_sell

        tau = self.execution_horizon / self.intervals

        # Market impact (BUG FIX: was using "permenant" everywhere)
        temporary_impact = self.market_impact.temporary_impact(
            trade_size=shares_to_sell, tau=tau
        )
        permanent_impact = self.market_impact.permanent_impact(
            trade_size=shares_to_sell
        )

        trade_price = self.current_price
        execution_price = trade_price - (temporary_impact + permanent_impact)
        implementation_shortfall = (
            (self.arrival_price - execution_price) * shares_to_sell
        )
        self.total_shortfall += implementation_shortfall

        # ---------------------------------------------------------------
        # Reward shaping — all coefficients from config (no magic numbers)
        # ---------------------------------------------------------------
        inventory_fraction = self.remaining_shares / self.total_shares
        inventory_penalty = self._reward_alpha * (inventory_fraction ** 2)
        impact_penalty = self._reward_beta * (temporary_impact + permanent_impact)

        reward = -implementation_shortfall
        reward -= inventory_penalty
        reward -= impact_penalty
        reward /= self._reward_scale

        info = {
            "shares_sold": shares_to_sell,
            "temporary_impact": temporary_impact,
            "permanent_impact": permanent_impact,
            "execution_price": execution_price,
            "implementation_shortfall": implementation_shortfall,
            "current_price": trade_price,
            "shortfall": self.total_shortfall,
        }

        self.current_step += 1
        if self.current_step < len(self.market_prices):
            self.current_price = self.market_prices[self.current_step]

        terminated = (
            self.current_step >= self.intervals or self.remaining_shares == 0
        )
        truncated = False

        # Terminal liquidation penalty — force-sell remaining shares
        if self.current_step >= self.intervals and self.remaining_shares > 0:
            temporary_impact = self.market_impact.temporary_impact(
                trade_size=self.remaining_shares, tau=tau
            )
            permanent_impact = self.market_impact.permanent_impact(
                trade_size=self.remaining_shares
            )
            execution_price = self.current_price - temporary_impact - permanent_impact
            terminal_shortfall = (
                (self.arrival_price - execution_price) * self.remaining_shares
            )
            self.total_shortfall = terminal_shortfall
            terminal_cost = (temporary_impact + permanent_impact) * self.remaining_shares
            reward -= terminal_cost / self._reward_scale
            self.remaining_shares = 0

        log.debug(
            "Step %d | action=%.4f, sold=%d, reward=%.4f, remaining=%d",
            self.current_step,
            action,
            shares_to_sell,
            reward,
            self.remaining_shares,
        )

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        price_change = (self.current_price - self.arrival_price) / self.arrival_price
        steps_left = max((self.intervals - self.current_step) / self.intervals, 1e-6)
        urgency = (self.remaining_shares / self.total_shares) / steps_left

        return np.array(
            [
                self.current_step / self.intervals,  # time progress ∈ [0, 1]
                self.remaining_shares / self.total_shares,  # inventory fraction ∈ [0, 1]
                price_change,  # normalised price drift
                urgency,  # execution urgency
            ],
            dtype=np.float32,
        )
