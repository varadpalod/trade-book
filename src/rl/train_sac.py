"""
train_sac.py
------------
SAC (Soft Actor-Critic) training script for optimal trade execution.

Why SAC over TD3?
    - Entropy regularisation naturally handles the exploration-exploitation
      tradeoff in noisy financial reward landscapes
    - Automatically adjusts exploration temperature via alpha tuning
    - More robust to sparse, delayed rewards than TD3
    - Works well with the expanded 7D contextual observation space

Phase 2 environment is used:
    - Domain Randomization: eta/gamma/sigma randomised on every episode
    - Contextual 7D observations include normalised impact params
    - Scale-invariant reward function

Usage:
    python -m src.rl.train_sac
    python -m src.rl.train_sac --timesteps 200000 --save sac_v2
"""
import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.execution_env import ExecutionEnv
from src.config.env_config import EnvConfig
from src.logger import get_logger

log = get_logger(__name__)


def make_env(seed: int = 0):
    """Factory function for vectorised environment creation."""
    def _init():
        env = ExecutionEnv(random_seed=seed)
        return env
    return _init


def train(
    total_timesteps: int = 100_000,
    save_path: str = "sac_execution",
    eval_freq: int = 5_000,
    checkpoint_freq: int = 25_000,
    tensorboard_log: str = "./logs/sac_tensorboard/",
) -> SAC:
    """
    Train a SAC agent on the contextual execution environment.

    Parameters
    ----------
    total_timesteps  : training steps
    save_path        : model save filename (without extension)
    eval_freq        : evaluate every N steps
    checkpoint_freq  : save checkpoint every N steps
    tensorboard_log  : TensorBoard log directory
    """
    log.info(
        "Starting SAC training | timesteps=%d, save=%s",
        total_timesteps,
        save_path,
    )

    # Training environment
    env = ExecutionEnv(random_seed=42)

    # Separate eval environment with a fixed seed window
    eval_env = DummyVecEnv([make_env(seed=999)])

    # ------------------------------------------------------------------
    # SAC hyperparameters — tuned for financial execution tasks
    # Key choices:
    #   learning_rate  : 3e-4 (Adam default, works well with SAC)
    #   buffer_size    : 500k (larger replay buffer for diverse episodes)
    #   learning_starts: 10k  (more warmup than TD3 since obs space is wider)
    #   batch_size     : 512  (larger batch = more stable gradient estimates)
    #   ent_coef       : 'auto' (entropy auto-tuning — key SAC advantage)
    #   target_entropy : 'auto' (SB3 sets to -action_dim = -1)
    # ------------------------------------------------------------------
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
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        tensorboard_log=tensorboard_log,
        policy_kwargs={
            "net_arch": [256, 256],  # 2-layer MLP: wider than default for 7D obs
        },
    )

    log.info("SAC model created | policy=MlpPolicy, net_arch=[256,256], ent_coef=auto")

    # Callbacks
    callbacks = []

    # Periodic checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="sac_checkpoint",
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    # Evaluation callback — saves best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}_best/",
        log_path="./logs/eval/",
        eval_freq=eval_freq,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    callbacks.append(eval_cb)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(save_path)
    log.info("Training complete | model saved to %s.zip", save_path)
    print(f"\nTraining complete. Model saved: {save_path}.zip")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SAC execution agent")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save", type=str, default="sac_execution")
    parser.add_argument("--eval-freq", type=int, default=5_000)
    args = parser.parse_args()

    # Ensure checkpoint dir exists
    Path("checkpoints").mkdir(exist_ok=True)

    train(
        total_timesteps=args.timesteps,
        save_path=args.save,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    main()
