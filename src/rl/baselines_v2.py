"""
baselines_v2.py
---------------
The classic strategies (TWAP, VWAP, Almgren-Chriss), run INSIDE the v2 environment so they
pay exactly the same costs (sell at bid, temporary + permanent impact) as the RL agent.

Each baseline is a schedule: how many units to sell at each step, via market orders.
Uses the project's own strategy classes in src/execution/.
"""
import numpy as np

from src.execution.twap import TWAPStrategy
from src.execution.vwap import VWAPStrategy
from src.execution.discrete_optimizer import DiscreteOptimizer


def twap_schedule(env) -> list[int]:
    return TWAPStrategy().generate_schedule(env.total_shares, env.intervals)


def vwap_schedule(env) -> list[int]:
    """Sell in proportion to the volume profile observed just before the episode."""
    return VWAPStrategy().generate_schedule(env.total_shares, env.intervals, env.hist_volume_profile)


def ac_schedule(env, risk: float = 1.0) -> list[int]:
    """
    Almgren-Chriss schedule using the env's TRUE impact parameters and the asset's volatility.

    `risk` is a unit-free risk-aversion: 1.0 means the variance penalty equals the expected
    impact cost for a TWAP schedule; larger = sell faster (front-loaded), 0 = TWAP.
    Units: price normalised to 1, one step = tau 1.
    """
    c, N, T = env.cfg, env.total_shares, env.intervals
    eta = c.impact_coef / N               # temp impact per share, per share traded
    gamma = c.perm_coef / N
    sigma = env.step_sigma
    eta_tilde = eta - gamma / 2.0
    expected_scale = eta_tilde * N ** 2 / T
    variance_scale = sigma ** 2 * N ** 2 * T / 3.0
    lam = risk * expected_scale / max(variance_scale, 1e-18)
    return DiscreteOptimizer().optimize(
        total_shares=N, intervals=T, execution_horizon=float(T),
        sigma=sigma, eta=eta, gamma=gamma, lambda_risk=lam,
    )


def run_schedule(env, schedule: list[int]) -> float:
    """Play one full episode (env must already be reset) selling `schedule[k]` units at step k. Returns cost in bps."""
    done, k = False, 0
    info = {}
    while not done:
        want = int(schedule[min(k, len(schedule) - 1)])
        free = env.remaining_shares - env._fill_sim.get_pending_qty()
        frac = 0.0 if want <= 0 or free <= 0 else min(1.0, (want + 0.5) / free)
        _, _, done, _, info = env.step(np.array([frac, 0.0, 1.0], dtype=np.float32))
        k += 1
    return float(info["agent_bps"])
