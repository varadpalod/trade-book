"""
train_sac_lo.py
---------------
SAC training on the Phase 8 Limit Order execution environment.

The agent learns to:
    1. Decide how much inventory to attempt per step
    2. Choose whether to post a passive Limit Order or cross the spread
    3. Pick the optimal price level to maximise spread capture

Usage:
    python -m src.rl.train_sac_lo --data data/btcusdt_trades.csv --timesteps 300000
    python -m src.rl.train_sac_lo --data data/btcusdt_trades.csv --save sac_lo_btc --timesteps 500000
"""
import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.market.lob_replay import LOBReplay
from src.rl.execution_env_lo import ExecutionEnvLO
from src.logger import get_logger

log = get_logger(__name__)


def make_lo_env(replay: LOBReplay, seed: int = 0, fill_model: str = "conservative"):
    def _init():
        return ExecutionEnvLO(lob_replay=replay, random_seed=seed, fill_model=fill_model)
    return _init


def train(
    data_path: str,
    fmt: str = "binance",
    total_timesteps: int = 300_000,
    save_path: str = "sac_lo",
    fill_model: str = "conservative",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
) -> SAC:
    log.info(
        "Phase 8 SAC-LO training | data=%s, format=%s, timesteps=%d, fill=%s",
        data_path, fmt, total_timesteps, fill_model,
    )

    replay = LOBReplay(filepath=data_path, fmt=fmt)
    if len(replay) < 50:
        raise ValueError(f"Too few ticks: {len(replay)}. Check your data file.")

    env = ExecutionEnvLO(lob_replay=replay, random_seed=42, fill_model=fill_model)
    eval_env = DummyVecEnv([make_lo_env(replay, seed=999, fill_model=fill_model)])

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
        tensorboard_log="./logs/sac_lo_tensorboard/",
        policy_kwargs={"net_arch": [256, 256]},
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    log.info("Model created | obs_dim=%d, act_dim=%d (Qty/Price/Type)", obs_dim, act_dim)
    print(f"Observation: {obs_dim}D | Action: {act_dim}D (Quantity, Price Level, Order Type)")
    print(f"Fill model: {fill_model}")

    Path("checkpoints").mkdir(exist_ok=True)
    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./checkpoints/",
            name_prefix="sac_lo_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"./{save_path}_best/",
            log_path="./logs/eval_lo/",
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
    log.info("Training complete | saved to %s.zip", save_path)
    print(f"\nTraining complete. Model saved: {save_path}.zip")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SAC with Limit Orders")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--format", type=str, default="binance", choices=["binance", "ohlcv", "lobster"])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save", type=str, default="sac_lo")
    parser.add_argument("--fill-model", type=str, default="conservative", choices=["conservative", "aggressive"])
    args = parser.parse_args()

    train(
        data_path=args.data,
        fmt=args.format,
        total_timesteps=args.timesteps,
        save_path=args.save,
        fill_model=args.fill_model,
    )


if __name__ == "__main__":
    main()
