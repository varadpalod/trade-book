"""
train_multi_asset.py
--------------------
Phase 10: Multi-Asset SAC training.

Trains one shared SAC execution agent across multiple Binance symbols.
The agent learns a universal execution policy that generalises across:
    - Different price scales (BTC vs BNB)
    - Different volatility regimes
    - Different liquidity depths
    - Eventually: different exchanges / asset classes

This is the "Execution Foundation Model" — one model to rule all assets.

Usage:
    # Single data directory with multiple CSVs:
    python -m src.rl.train_multi_asset --data-dir data/ --symbols BTCUSDT,ETHUSDT

    # Download multiple symbols first:
    python -m src.market.download_binance_data --symbol ETHUSDT --date 2024-03-01
    python -m src.market.download_binance_data --symbol BNBUSDT --date 2024-03-01
    python -m src.rl.train_multi_asset --data-dir data/ --symbols BTCUSDT,ETHUSDT,BNBUSDT
"""
import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.market.multi_asset_replay import MultiAssetReplay
from src.rl.execution_env_multi import ExecutionEnvMulti
from src.logger import get_logger

log = get_logger(__name__)


def make_multi_env(multi_replay: MultiAssetReplay, seed: int = 0):
    def _init():
        return ExecutionEnvMulti(multi_replay=multi_replay, random_seed=seed)
    return _init


def train(
    data_dir: str,
    symbols: list[str],
    fmt: str = "binance",
    total_timesteps: int = 500_000,
    save_path: str = "sac_multi",
    eval_freq: int = 20_000,
    checkpoint_freq: int = 100_000,
) -> SAC:
    log.info(
        "Phase 10 Multi-Asset SAC | data_dir=%s, symbols=%s, timesteps=%d",
        data_dir, symbols, total_timesteps,
    )

    multi_replay = MultiAssetReplay(data_dir=data_dir, symbols=symbols, fmt=fmt)
    log.info("Loaded %d assets: %s", len(multi_replay), multi_replay.symbols)

    env = ExecutionEnvMulti(multi_replay=multi_replay, random_seed=42)
    eval_env = DummyVecEnv([make_multi_env(multi_replay, seed=999)])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        device="cuda",
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,  # Larger buffer for multi-asset diversity
        learning_starts=20_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
        tensorboard_log="./logs/sac_multi_tensorboard/",
        policy_kwargs={"net_arch": [256, 256, 128]},  # Deeper for generalisation
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    log.info("Multi-Asset SAC | obs_dim=%d (12D LOB + 3D ctx), act_dim=%d", obs_dim, act_dim)
    print(f"Assets: {multi_replay.symbols}")
    print(f"Observation: {obs_dim}D (12D LOB + 3D asset context)")
    print(f"Action: {act_dim}D (Quantity / Price Level / Order Type)")

    Path("checkpoints").mkdir(exist_ok=True)
    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./checkpoints/",
            name_prefix="sac_multi_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"./{save_path}_best/",
            log_path="./logs/eval_multi/",
            eval_freq=eval_freq,
            n_eval_episodes=30,
            deterministic=True,
        ),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(save_path)
    log.info("Multi-Asset training complete | saved to %s.zip", save_path)
    print(f"\nTraining complete. Model saved: {save_path}.zip")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train multi-asset execution SAC")
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--symbols", type=str, default="BTCUSDT",
                        help="Comma-separated symbol list, e.g. BTCUSDT,ETHUSDT,BNBUSDT")
    parser.add_argument("--format", type=str, default="binance", choices=["binance", "ohlcv"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save", type=str, default="sac_multi")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    train(
        data_dir=args.data_dir,
        symbols=symbols,
        fmt=args.format,
        total_timesteps=args.timesteps,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
