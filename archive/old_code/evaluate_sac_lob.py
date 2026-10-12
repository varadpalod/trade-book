"""
evaluate_sac_lob.py
-------------------
Evaluates a trained SAC agent on the LOB/OHLCV environment.

Usage:
    python -m src.rl.evaluate_sac_lob --model sac_reliance --data data/RELIANCE.NS_1m.csv --format ohlcv
"""
import argparse
import numpy as np
from stable_baselines3 import SAC

from src.market.lob_replay import LOBReplay
from src.rl.execution_env_lob import ExecutionEnvLOB
from src.evaluation.metrics import ExecutionMetrics
from src.logger import get_logger

log = get_logger(__name__)


def evaluate_lob(model_path: str, data_path: str, fmt: str, episodes: int = 5):
    log.info("Evaluating LOB model=%s on data=%s", model_path, data_path)
    
    replay = LOBReplay(filepath=data_path, fmt=fmt)
    env = ExecutionEnvLOB(lob_replay=replay)
    
    try:
        model = SAC.load(model_path)
    except Exception as e:
        log.error("Failed to load model: %s", e)
        return

    m = ExecutionMetrics()
    
    total_shortfall = 0.0
    total_fill_rate = 0.0
    total_reward = 0.0

    print(f"\n--- EVALUATING {model_path} ---")
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=42 + ep)
        
        done = False
        ep_reward = 0.0
        actions = []
        exec_prices = []
        qtys = []
        market_prices = []
        
        arrival_price = env.arrival_price
        target_shares = env.total_shares

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            actions.append(float(action[0]))
            
            if info["shares_sold"] > 0:
                exec_prices.append(info["execution_price"])
                qtys.append(info["shares_sold"])
                market_prices.append(info["mid_price"])
                
            done = terminated or truncated

        # Metrics for episode
        is_val = m.implementation_shortfall(arrival_price, exec_prices, qtys)
        fill = m.fill_rate(target_shares, env.remaining_shares)
        
        total_shortfall += is_val
        total_fill_rate += fill
        total_reward += ep_reward
        
        print(f"Episode {ep+1}/{episodes} | Arrival: {arrival_price:.2f} | Shortfall: {is_val:.2f} | Fill Rate: {fill:.1%} | Reward: {ep_reward:.2f}")

    print("-" * 50)
    print(f"Average Shortfall : {total_shortfall / episodes:.2f}")
    print(f"Average Fill Rate : {total_fill_rate / episodes:.1%}")
    print(f"Average Reward    : {total_reward / episodes:.2f}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sac_reliance")
    parser.add_argument("--data", type=str, default="data/RELIANCE.NS_1m.csv")
    parser.add_argument("--format", type=str, default="ohlcv")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    evaluate_lob(args.model, args.data, args.format, args.episodes)


if __name__ == "__main__":
    main()
