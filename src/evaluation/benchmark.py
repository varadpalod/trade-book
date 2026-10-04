"""
benchmark.py
------------
Benchmarks SAC vs TD3 vs Almgren-Chriss vs TWAP on shared market paths.

Phase 3 changes:
    - Added SAC evaluation (evaluate_sac)
    - Added TWAP baseline
    - Updated compare() to benchmark all four strategies
    - Updated for 7D observation space (Phase 2 env)
    - Added structured logging

Usage:
    python -m src.evaluation.benchmark
"""
import numpy as np
from stable_baselines3 import SAC, TD3

from src.rl.execution_env import ExecutionEnv
from src.execution.almgren_chriss import AlmgrenChrissStrategy
from src.execution.twap import TWAPStrategy
from src.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual strategy evaluators
# ---------------------------------------------------------------------------

def evaluate_sac(shared_market_prices: list, seed: int, verbose: bool = False) -> dict:
    """Run SAC agent on a pre-generated price path."""
    env = ExecutionEnv(random_seed=seed)
    obs, _ = env.reset(seed=seed)
    env.market_prices = shared_market_prices.copy()
    env.current_price = env.market_prices[0]
    env.arrival_price = env.market_prices[0]

    try:
        model = SAC.load("sac_execution")
    except FileNotFoundError:
        log.warning("sac_execution.zip not found — skipping SAC evaluation.")
        return {"reward": 0.0, "shortfall": 0.0, "actions": [], "remaining_inventory": env.total_shares}

    done = False
    total_reward = 0.0
    actions = []
    step = 1

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action[0]))
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            _print_step("SAC", step, action, info, reward)
        total_reward += reward
        done = terminated or truncated
        step += 1

    return {
        "reward": total_reward,
        "shortfall": info["shortfall"],
        "actions": actions,
        "remaining_inventory": env.remaining_shares,
    }


def evaluate_td3(shared_market_prices: list, seed: int, verbose: bool = False) -> dict:
    """Run TD3 agent on a pre-generated price path."""
    env = ExecutionEnv(random_seed=seed)
    obs, _ = env.reset(seed=seed)
    env.market_prices = shared_market_prices.copy()
    env.current_price = env.market_prices[0]
    env.arrival_price = env.market_prices[0]

    try:
        model = TD3.load("td3_execution_gbm1")
    except FileNotFoundError:
        log.warning("td3_execution_gbm1.zip not found — skipping TD3 evaluation.")
        return {"reward": 0.0, "shortfall": 0.0, "actions": [], "remaining_inventory": env.total_shares}

    done = False
    total_reward = 0.0
    actions = []
    step = 1

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action[0]))
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            _print_step("TD3", step, action, info, reward)
        total_reward += reward
        done = terminated or truncated
        step += 1

    return {
        "reward": total_reward,
        "shortfall": info["shortfall"],
        "actions": actions,
        "remaining_inventory": env.remaining_shares,
    }


def evaluate_almgren(shared_market_prices: list, seed: int, verbose: bool = False) -> dict:
    """Run Almgren-Chriss analytical schedule on a pre-generated price path."""
    env = ExecutionEnv(random_seed=seed)
    env.reset(seed=seed)
    env.market_prices = shared_market_prices.copy()
    env.current_price = env.market_prices[0]
    env.arrival_price = env.market_prices[0]

    strategy = AlmgrenChrissStrategy()
    schedule = strategy.generate_schedule(
        total_shares=env.total_shares,
        execution_horizon=env.execution_horizon,
        intervals=env.intervals,
    )

    total_reward = 0.0
    total_shortfall = 0.0
    remaining_inventory = env.total_shares
    trades = []
    tau = env.execution_horizon / env.intervals

    for i, shares_to_sell in enumerate(schedule):
        current_price = env.market_prices[i]
        temporary_impact = env.market_impact.temporary_impact(trade_size=shares_to_sell, tau=tau)
        permanent_impact = env.market_impact.permanent_impact(trade_size=shares_to_sell)
        execution_price = current_price - temporary_impact - permanent_impact
        is_step = (env.arrival_price - execution_price) * shares_to_sell / max(env.arrival_price, 1e-8)
        total_shortfall += is_step

        reward = -is_step
        total_reward += reward
        remaining_inventory -= shares_to_sell
        trades.append(shares_to_sell)

        if verbose:
            print(f"  [AC] Step {i+1}: price={current_price:.4f}, sold={shares_to_sell}, IS={is_step:.4f}")

    return {
        "reward": total_reward,
        "shortfall": total_shortfall,
        "actions": trades,
        "remaining_inventory": remaining_inventory,
    }


def evaluate_twap(shared_market_prices: list, seed: int, verbose: bool = False) -> dict:
    """Run TWAP baseline on a pre-generated price path."""
    env = ExecutionEnv(random_seed=seed)
    env.reset(seed=seed)
    env.market_prices = shared_market_prices.copy()
    env.current_price = env.market_prices[0]
    env.arrival_price = env.market_prices[0]

    strategy = TWAPStrategy()
    schedule = strategy.generate_schedule(env.total_shares, env.intervals)

    total_reward = 0.0
    total_shortfall = 0.0
    remaining_inventory = env.total_shares
    trades = []
    tau = env.execution_horizon / env.intervals

    for i, shares_to_sell in enumerate(schedule):
        current_price = env.market_prices[i]
        temporary_impact = env.market_impact.temporary_impact(trade_size=shares_to_sell, tau=tau)
        permanent_impact = env.market_impact.permanent_impact(trade_size=shares_to_sell)
        execution_price = current_price - temporary_impact - permanent_impact
        is_step = (env.arrival_price - execution_price) * shares_to_sell / max(env.arrival_price, 1e-8)
        total_shortfall += is_step

        reward = -is_step
        total_reward += reward
        remaining_inventory -= shares_to_sell
        trades.append(shares_to_sell)

        if verbose:
            print(f"  [TWAP] Step {i+1}: price={current_price:.4f}, sold={shares_to_sell}, IS={is_step:.4f}")

    return {
        "reward": total_reward,
        "shortfall": total_shortfall,
        "actions": trades,
        "remaining_inventory": remaining_inventory,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def compare(episodes: int = 100) -> None:
    """
    Compare SAC vs TD3 vs AC vs TWAP across N shared market paths.
    """
    strategies = ["SAC", "TD3", "AC", "TWAP"]
    evaluators = {
        "SAC": evaluate_sac,
        "TD3": evaluate_td3,
        "AC": evaluate_almgren,
        "TWAP": evaluate_twap,
    }

    results = {s: {"rewards": [], "shortfalls": []} for s in strategies}
    wins = {s: 0 for s in strategies}

    log.info("Starting benchmark | episodes=%d, strategies=%s", episodes, strategies)

    for seed in range(episodes):
        # Generate one shared market path
        env = ExecutionEnv(random_seed=seed)
        env.reset(seed=seed)
        shared_prices = env.market_prices.copy()

        verbose = seed in [0, 42, 99]
        if verbose:
            print(f"\n--- Seed {seed} | Prices[:5]={[f'{p:.2f}' for p in shared_prices[:5]]} ---")

        episode_shortfalls = {}
        for name in strategies:
            res = evaluators[name](shared_prices, seed, verbose=verbose)
            results[name]["rewards"].append(res["reward"])
            results[name]["shortfalls"].append(res["shortfall"])
            episode_shortfalls[name] = res["shortfall"]

        # Winner = lowest implementation shortfall
        best = min(episode_shortfalls, key=episode_shortfalls.get)
        wins[best] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<12} {'Mean IS':>12} {'Std IS':>10} {'Win Rate':>10}")
    print("-" * 50)
    for name in strategies:
        sf = results[name]["shortfalls"]
        print(
            f"{name:<12} {np.mean(sf):>12.4f} {np.std(sf):>10.4f} "
            f"{100 * wins[name] / episodes:>9.1f}%"
        )
    print("=" * 70)

    log.info("Benchmark complete | wins=%s", wins)


def _print_step(strategy: str, step: int, action, info: dict, reward: float) -> None:
    print(
        f"  [{strategy}] Step {step}: action={float(action[0]):.3f}, "
        f"price={info['current_price']:.4f}, sold={info['shares_sold']}, "
        f"IS={info['implementation_shortfall']:.4f}, reward={reward:.4f}"
    )


if __name__ == "__main__":
    compare()
