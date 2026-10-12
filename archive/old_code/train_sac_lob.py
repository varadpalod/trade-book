"""
train_sac_lob.py
----------------
SAC training on the LOB-backed execution environment.

Requires real historical tick data in data/ directory.
See data/README.md for download instructions.

Usage:
    python -m src.rl.train_sac_lob --data data/btcusdt_trades.csv
    python -m src.rl.train_sac_lob --data data/btcusdt_trades.csv --format binance --timesteps 200000
"""
import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.market.lob_replay import LOBReplay
from src.rl.execution_env_lob import ExecutionEnvLOB
from src.config.env_config import EnvConfig
from src.logger import get_logger

log = get_logger(__name__)


def make_lob_env(replay: LOBReplay, seed: int = 0):
    """Factory for LOB environment (for DummyVecEnv)."""
    def _init():
        return ExecutionEnvLOB(lob_replay=replay, random_seed=seed)
    return _init


def train(
    data_path: str,
    fmt: str = "binance",
    total_timesteps: int = 200_000,
    save_path: str = "sac_execution_lob",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
) -> SAC:
    """Train SAC on real LOB data."""
    log.info(
        "Starting LOB SAC training | data=%s, format=%s, timesteps=%d",
        data_path, fmt, total_timesteps,
    )

    replay = LOBReplay(filepath=data_path, fmt=fmt)

    if len(replay) < 100:
        raise ValueError(
            f"Insufficient tick data: only {len(replay)} snapshots loaded. "
            "Need at least 100 ticks. Check data/README.md."
        )

    env = ExecutionEnvLOB(lob_replay=replay, random_seed=42)
    eval_env = DummyVecEnv([make_lob_env(replay, seed=999)])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        device="cuda",
        verbose=1,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
        tensorboard_log="./logs/sac_lob_tensorboard/",
        policy_kwargs={
            "net_arch": [256, 256],
        },
    )

    log.info("LOB SAC model created | obs_dim=%d", env.observation_space.shape[0])
    print(f"Observation space: {env.observation_space.shape[0]}D (real LOB signals)")

    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./checkpoints/",
            name_prefix="sac_lob_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"./{save_path}_best/",
            log_path="./logs/eval_lob/",
            eval_freq=eval_freq,
            n_eval_episodes=20,
            deterministic=True,
        ),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(save_path)
    log.info("LOB training complete | saved to %s.zip", save_path)
    print(f"\nTraining complete. Model saved: {save_path}.zip")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SAC on LOB data")
    parser.add_argument("--data", type=str, required=True, help="Path to tick CSV file")
    parser.add_argument("--format", type=str, default="binance", choices=["binance", "ohlcv", "lobster"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save", type=str, default="sac_execution_lob")
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    train(
        data_path=args.data,
        fmt=args.format,
        total_timesteps=args.timesteps,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
