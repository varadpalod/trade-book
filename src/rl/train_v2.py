"""
train_v2.py
-----------
Train SAC on ExecutionEnvV2 (realistic costs, TWAP-relative reward).

    python -m src.rl.train_v2 --data-dir data/ --symbols BTCUSDT,ETHUSDT,BNBUSDT \
        --timesteps 400000 --save sac_v2

Best model (by eval-split reward, i.e. bps improvement over TWAP) -> {save}_best/best_model.zip
Auto-selects CUDA if available, else CPU.
"""
import argparse
from pathlib import Path

import importlib.util

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.market.multi_asset_replay import MultiAssetReplay
from src.rl.execution_env_v2 import ExecutionEnvV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT")
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--save", default="sac_v2")
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    symbols = [s.strip().upper() for s in a.symbols.split(",")]
    mr = MultiAssetReplay(data_dir=a.data_dir, symbols=symbols)

    env = Monitor(ExecutionEnvV2(mr, random_seed=a.seed, split="train"))
    eval_env = DummyVecEnv([lambda: Monitor(ExecutionEnvV2(mr, random_seed=999, split="eval"))])

    model = SAC(
        "MlpPolicy", env, device=device, verbose=1, seed=a.seed,
        learning_rate=3e-4, buffer_size=300_000, learning_starts=5_000,
        batch_size=256, tau=0.005, gamma=0.99, ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=("./logs/sac_v2_tensorboard/" if importlib.util.find_spec("tensorboard") else None),
    )
    Path("checkpoints").mkdir(exist_ok=True)
    model.learn(
        total_timesteps=a.timesteps,
        callback=[
            CheckpointCallback(50_000, "./checkpoints/", name_prefix=f"{a.save}_ckpt"),
            EvalCallback(eval_env, best_model_save_path=f"./{a.save}_best/",
                         log_path="./logs/eval_v2/", eval_freq=a.eval_freq,
                         n_eval_episodes=60, deterministic=True),
        ],
        progress_bar=False,
    )
    model.save(a.save)
    print("DONE. saved", a.save)


if __name__ == "__main__":
    main()
