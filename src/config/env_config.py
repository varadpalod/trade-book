"""
env_config.py
-------------
Central configuration dataclass for the trade execution environment.

All magic numbers that were previously scattered across execution_env.py,
train_td3.py, and market_simulator.py are defined here.

Phase 2 will use the *_range fields for Domain Randomization —
each reset() will sample uniformly within those ranges.
"""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # ---------------------------------------------------------------
    # Core execution parameters
    # ---------------------------------------------------------------
    total_shares: int = 1_000
    intervals: int = 10
    initial_price: float = 100.0
    execution_horizon: float = 60.0  # seconds

    # ---------------------------------------------------------------
    # Market impact parameters (Almgren-Chriss)
    # eta   = temporary impact coefficient
    # gamma = permanent impact coefficient
    # sigma = price volatility (annualised daily vol proxy)
    # ---------------------------------------------------------------
    sigma: float = 1.0
    eta: float = 2.5e-6
    gamma: float = 2.5e-7

    # ---------------------------------------------------------------
    # Domain Randomization ranges (used in Phase 2)
    # Each reset() will sample uniformly from [low, high]
    # ---------------------------------------------------------------
    sigma_range: tuple[float, float] = field(default_factory=lambda: (0.5, 2.0))
    eta_range: tuple[float, float] = field(default_factory=lambda: (1e-6, 5e-6))
    gamma_range: tuple[float, float] = field(default_factory=lambda: (1e-7, 5e-7))

    # ---------------------------------------------------------------
    # Reward shaping coefficients
    # Previously hardcoded as alpha=1.0, beta=100.0 in execution_env.py
    # ---------------------------------------------------------------
    reward_alpha: float = 1.0        # inventory penalty weight
    reward_beta: float = 100.0       # impact penalty weight
    reward_scale: float = 100.0      # reward denominator (reward /= reward_scale)

    # ---------------------------------------------------------------
    # Environment penalty terms
    # Previously hardcoded as lambda_inventory=0.001, lambda_terminal=10.0
    # ---------------------------------------------------------------
    lambda_inventory: float = 0.001
    lambda_terminal: float = 10.0

    # ---------------------------------------------------------------
    # Random seed
    # ---------------------------------------------------------------
    random_seed: int = 42


# Singleton default config — import this directly for quick usage
DEFAULT_CONFIG = EnvConfig()
