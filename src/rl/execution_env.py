"""
execution_env.py
----------------
Gymnasium environment for optimal trade execution.

Phase 2 changes (Domain Randomization + Contextual RL):
    - eta, gamma, sigma randomised on every reset() from EnvConfig ranges
    - Observation space expanded from 4D to 7D:
        [time_progress, inventory_fraction, price_change, urgency,
         rolling_volatility, eta_norm, gamma_norm]
    - MarketImpact object rebuilt each reset with fresh sampled params
    - Reward normalised by arrival_price for scale invariance
    - Synthetic LOB signals (spread proxy, imbalance proxy) added to info dict

Observation space (7D):
    0: time_progress       — current_step / intervals          ∈ [0, 1]
    1: inventory_fraction  — remaining / total                 ∈ [0, 1]
    2: price_change        — (P_t - P_0) / P_0                ∈ ℝ
    3: urgency             — inventory_fraction / time_left    ∈ [0, ∞)
    4: rolling_volatility  — std of last N price returns       ∈ [0, ∞)
    5: eta_norm            — eta / eta_max (agent knows regime) ∈ [0, 1]
    6: gamma_norm          — gamma / gamma_max                  ∈ [0, 1]

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

# Window size for rolling volatility signal
_VOL_WINDOW = 5

# sigma in EnvConfig is an ANNUALISED volatility (0.5 = 50%/yr), but tau is in SECONDS.
# Using sigma directly per sqrt(second) made the simulated price crash ~99% every episode
# (so any "sell early" policy looked brilliant). Convert once, here.
_TRADING_SECONDS_PER_YEAR = 252 * 6.5 * 3600


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, random_seed: int = 42) -> None:
        super().__init__()

        self._cfg = config if config is not None else DEFAULT_CONFIG
        self._base_rng = np.random.default_rng(random_seed)

        # Static env dimensions (don't change between episodes)
        self.total_shares = self._cfg.total_shares
        self.intervals = self._cfg.intervals
        self.initial_price = self._cfg.initial_price
        self.execution_horizon = self._cfg.execution_horizon

        # Reward shaping (from config)
        self._reward_alpha = self._cfg.reward_alpha
        self._reward_beta = self._cfg.reward_beta
        self._reward_scale = self._cfg.reward_scale
        self._lambda_inventory = self._cfg.lambda_inventory
        self._lambda_terminal = self._cfg.lambda_terminal

        # These will be re-sampled every reset() in Phase 2
        self.sigma = self._cfg.sigma
        self.eta = self._cfg.eta
        self.gamma = self._cfg.gamma

        # Internal state — initialised in reset()
        self.remaining_shares: int = self.total_shares
        self.current_price: float = self.initial_price
        self.current_step: int = 0
        self.arrival_price: float = self.initial_price
        self.total_shortfall: float = 0.0
        self.market_prices: list[float] = []
        self._price_returns: list[float] = []  # for rolling vol

        self.price_process = PriceProcess(random_seed=random_seed)
        self.market_impact = MarketImpact(eta=self.eta, gamma=self.gamma)

        # ------------------------------------------------------------------
        # Phase 2: Expanded observation space (7D)
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        log.info(
            "ExecutionEnv (Phase 2) initialised | shares=%d, intervals=%d, obs_dim=7",
            self.total_shares,
            self.intervals,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        rng = np.random.default_rng(seed) if seed is not None else self._base_rng

        # ------------------------------------------------------------------
        # Phase 2.1 — Domain Randomization: sample market params each episode
        # ------------------------------------------------------------------
        self.sigma = float(rng.uniform(*self._cfg.sigma_range))
        self.eta = float(rng.uniform(*self._cfg.eta_range))
        self.gamma = float(rng.uniform(*self._cfg.gamma_range))

        # Rebuild impact model with newly sampled params
        self.market_impact = MarketImpact(eta=self.eta, gamma=self.gamma)

        # Reset internal state
        self.remaining_shares = self.total_shares
        self.current_price = self.initial_price
        self.current_step = 0
        self.total_shortfall = 0.0
        self._price_returns = []

        if seed is not None:
            self.price_process._rng = np.random.default_rng(seed + 1)

        tau = self.execution_horizon / self.intervals
        self.market_prices = self.price_process.generate_path(
            initial_price=self.initial_price,
            sigma=self.sigma / np.sqrt(_TRADING_SECONDS_PER_YEAR),  # annualised -> per-sqrt(second)
            intervals=self.intervals,
            tau=tau,
        )
        self.current_price = self.market_prices[0]
        self.arrival_price = self.current_price

        log.debug(
            "Env reset | sigma=%.3f, eta=%.2e, gamma=%.2e, arrival=%.4f",
            self.sigma, self.eta, self.gamma, self.arrival_price,
        )
        return self._get_observation(), {}

    def step(self, action):
        action = float(np.clip(action[0], 0.0, 1.0))
        shares_to_sell = int(action * self.remaining_shares)
        shares_to_sell = min(shares_to_sell, self.remaining_shares)
        self.remaining_shares -= shares_to_sell

        tau = self.execution_horizon / self.intervals

        temporary_impact = self.market_impact.temporary_impact(
            trade_size=shares_to_sell, tau=tau
        )
        permanent_impact = self.market_impact.permanent_impact(
            trade_size=shares_to_sell
        )

        trade_price = self.current_price
        execution_price = trade_price - (temporary_impact + permanent_impact)

        # ------------------------------------------------------------------
        # Phase 2.5 — Scale-invariant reward (normalised by arrival price)
        # ------------------------------------------------------------------
        if self.arrival_price > 0:
            implementation_shortfall = (
                (self.arrival_price - execution_price) * shares_to_sell
            ) / self.arrival_price
        else:
            implementation_shortfall = 0.0

        self.total_shortfall += implementation_shortfall

        inventory_fraction = self.remaining_shares / self.total_shares
        inventory_penalty = self._reward_alpha * (inventory_fraction ** 2)
        impact_penalty = self._reward_beta * (temporary_impact + permanent_impact)

        reward = -implementation_shortfall
        reward -= inventory_penalty
        reward -= impact_penalty / self._reward_scale

        info = {
            "shares_sold": shares_to_sell,
            "temporary_impact": temporary_impact,
            "permanent_impact": permanent_impact,
            "execution_price": execution_price,
            "implementation_shortfall": implementation_shortfall,
            "current_price": trade_price,
            "shortfall": self.total_shortfall,
            # Phase 2.4 — synthetic LOB proxy signals (for monitoring/info)
            "spread_proxy": self.eta * tau * 2,          # ~half-spread proxy
            "imbalance_proxy": (action - 0.5) * 2,       # signed pressure
        }

        self.current_step += 1

        # Track price returns for rolling volatility
        if len(self.market_prices) > self.current_step:
            prev = self.market_prices[self.current_step - 1]
            curr = self.market_prices[self.current_step]
            if prev > 0:
                self._price_returns.append((curr - prev) / prev)
        
        if self.current_step < len(self.market_prices):
            self.current_price = self.market_prices[self.current_step]

        terminated = (
            self.current_step >= self.intervals or self.remaining_shares == 0
        )
        truncated = False

        # Terminal liquidation penalty
        if self.current_step >= self.intervals and self.remaining_shares > 0:
            temporary_impact = self.market_impact.temporary_impact(
                trade_size=self.remaining_shares, tau=tau
            )
            permanent_impact = self.market_impact.permanent_impact(
                trade_size=self.remaining_shares
            )
            execution_price = self.current_price - temporary_impact - permanent_impact
            if self.arrival_price > 0:
                terminal_shortfall = (
                    (self.arrival_price - execution_price) * self.remaining_shares
                ) / self.arrival_price
            else:
                terminal_shortfall = 0.0
            # was `=` (overwrote earlier costs) and never reached info["shortfall"], so unsold
            # inventory was never charged. Accumulate and publish it.
            self.total_shortfall += terminal_shortfall
            terminal_cost = (temporary_impact + permanent_impact) * self.remaining_shares
            reward -= terminal_cost / (self._reward_scale * self.arrival_price)
            self.remaining_shares = 0
            info["shortfall"] = self.total_shortfall

        log.debug(
            "Step %d | action=%.3f, sold=%d, reward=%.5f, remaining=%d, sigma=%.3f",
            self.current_step, action, shares_to_sell, reward,
            self.remaining_shares, self.sigma,
        )

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        7D observation vector:
          [time_progress, inventory_fraction, price_change, urgency,
           rolling_volatility, eta_norm, gamma_norm]
        """
        price_change = (self.current_price - self.arrival_price) / max(
            self.arrival_price, 1e-8
        )

        steps_left = max(
            (self.intervals - self.current_step) / self.intervals, 1e-6
        )
        urgency = (self.remaining_shares / self.total_shares) / steps_left

        # Phase 2.3 — Rolling volatility signal
        if len(self._price_returns) >= 2:
            rolling_vol = float(np.std(self._price_returns[-_VOL_WINDOW:]))
        else:
            rolling_vol = 0.0

        # Phase 2.2 — Contextual RL: normalised impact params as observations
        eta_norm = (self.eta - self._cfg.eta_range[0]) / (
            self._cfg.eta_range[1] - self._cfg.eta_range[0] + 1e-12
        )
        gamma_norm = (self.gamma - self._cfg.gamma_range[0]) / (
            self._cfg.gamma_range[1] - self._cfg.gamma_range[0] + 1e-12
        )

        return np.array(
            [
                self.current_step / self.intervals,      # 0: time progress
                self.remaining_shares / self.total_shares,  # 1: inventory fraction
                price_change,                             # 2: normalised price drift
                urgency,                                  # 3: execution urgency
                rolling_vol,                              # 4: rolling volatility
                float(np.clip(eta_norm, 0.0, 1.0)),       # 5: eta in [0,1]
                float(np.clip(gamma_norm, 0.0, 1.0)),     # 6: gamma in [0,1]
            ],
            dtype=np.float32,
        )
