"""
evaluate_sac.py
---------------
SAC agent evaluation script.

Evaluates the SAC model across multiple episodes, printing per-episode
metrics and a summary table.

Usage:
    python -m src.rl.evaluate_sac
    python -m src.rl.evaluate_sac --model sac_execution --episodes 10 --verbose
"""
import argparse
import numpy as np

from stable_baselines3 import SAC
from src.rl.execution_env import ExecutionEnv
from src.logger import get_logger

log = get_logger(__name__)


def evaluate(
    model_path: str = "sac_execution",
    episodes: int = 10,
    verbose: bool = False,
) -> dict:
    """
    Evaluate a trained SAC model.

    Returns
    -------
    dict with keys: mean_reward, std_reward, mean_shortfall, win_rate_vs_passive
    """
    env = ExecutionEnv(random_seed=1000)  # fresh seed not seen during training
    model = SAC.load(model_path)

    log.info("Evaluating SAC model: %s | episodes=%d", model_path, episodes)

    rewards = []
    shortfalls = []
    remaining_invs = []
    episode_actions = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        actions = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(float(action[0]))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        shortfalls.append(info["shortfall"])
        remaining_invs.append(env.remaining_shares)
        episode_actions.append(actions)

        if verbose:
            print(f"\nEpisode {episode + 1}")
            print(f"  eta={env.eta:.2e}, gamma={env.gamma:.2e}, sigma={env.sigma:.3f}")
            print(f"  Total Reward       : {total_reward:.4f}")
            print(f"  Implementation Shortfall: {info['shortfall']:.4f}")
            print(f"  Remaining Inventory: {env.remaining_shares}")
            print(f"  Actions            : {[f'{a:.3f}' for a in actions]}")

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_shortfall = float(np.mean(shortfalls))
    fill_rate = float(
        np.mean([1.0 - r / env.total_shares for r in remaining_invs])
    )

    print("\n" + "=" * 60)
    print("SAC Evaluation Summary")
    print("=" * 60)
    print(f"  Episodes           : {episodes}")
    print(f"  Mean Reward        : {mean_reward:.4f} +/- {std_reward:.4f}")
    print(f"  Mean IS Shortfall  : {mean_shortfall:.4f}")
    print(f"  Fill Rate          : {fill_rate * 100:.1f}%")
    print(f"  Remaining Inventory (mean): {np.mean(remaining_invs):.0f}")
    print("=" * 60)

    log.info(
        "Eval complete | mean_reward=%.4f, mean_IS=%.4f, fill_rate=%.1f%%",
        mean_reward,
        mean_shortfall,
        fill_rate * 100,
    )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_shortfall": mean_shortfall,
        "fill_rate": fill_rate,
        "rewards": rewards,
        "shortfalls": shortfalls,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC execution agent")
    parser.add_argument("--model", type=str, default="sac_execution")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluate(model_path=args.model, episodes=args.episodes, verbose=args.verbose)


if __name__ == "__main__":
    main()
