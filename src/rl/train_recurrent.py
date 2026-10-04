"""
train_recurrent.py
------------------
Phase 9: Recurrent RL training using RecurrentPPO (LSTM policy).

Trading is highly sequential — the agent needs to REMEMBER what happened
5 steps ago to detect toxic order flow and institutional sweeps.

RecurrentPPO (sb3-contrib) wraps any Gymnasium env with an LSTM that
maintains hidden state across timesteps within an episode, giving the
agent persistent memory without changing the environment at all.

Architecture:
    LOB state sequence (t, t-1, t-2 ... via LSTM hidden state)
          ↓
        LSTM (256 hidden units)
          ↓
    PPO / Recurrent policy
          ↓
    Execution decision

NOTE: RecurrentPPO is an on-policy algorithm (unlike SAC which is
off-policy). It is generally more sample-inefficient but more stable
and naturally handles sequential memory.

Usage:
    # Install sb3-contrib first:
    pip install sb3-contrib

    # Train on Binance LOB data:
    python -m src.rl.train_recurrent --data data/btcusdt_trades.csv --timesteps 500000

    # Train on Limit Order env (Phase 8):
    python -m src.rl.train_recurrent --data data/btcusdt_trades.csv --env lo --timesteps 500000
"""
import argparse
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv

from src.market.lob_replay import LOBReplay
from src.rl.execution_env_lob import ExecutionEnvLOB
from src.rl.execution_env_lo import ExecutionEnvLO
from src.logger import get_logger

log = get_logger(__name__)


def make_env(env_type: str, replay: LOBReplay, seed: int = 0):
    def _init():
        if env_type == "lo":
            return ExecutionEnvLO(lob_replay=replay, random_seed=seed)
        return ExecutionEnvLOB(lob_replay=replay, random_seed=seed)
    return _init


def train(
    data_path: str,
    fmt: str = "binance",
    env_type: str = "lob",
    total_timesteps: int = 500_000,
    save_path: str = "recurrent_ppo",
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 1,
    n_steps: int = 128,
    batch_size: int = 64,
    n_epochs: int = 10,
):
    """
    Train a RecurrentPPO agent with LSTM memory on LOB data.

    Parameters
    ----------
    data_path       : path to Binance tick CSV
    fmt             : 'binance' | 'ohlcv'
    env_type        : 'lob' (9D, market orders) | 'lo' (12D, limit orders)
    total_timesteps : training steps
    lstm_hidden_size: number of LSTM hidden units
    n_lstm_layers   : number of stacked LSTM layers
    """
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        raise ImportError(
            "sb3-contrib not installed.\n"
            "Run: pip install sb3-contrib\n"
            "This provides RecurrentPPO with LSTM policy support."
        )

    log.info(
        "Phase 9 RecurrentPPO training | data=%s, env=%s, timesteps=%d, lstm=%dx%d",
        data_path, env_type, total_timesteps, lstm_hidden_size, n_lstm_layers,
    )

    replay = LOBReplay(filepath=data_path, fmt=fmt)
    vec_env = DummyVecEnv([make_env(env_type, replay, seed=42)])

    # RecurrentPPO requires a single non-vectorized env for eval
    eval_env = DummyVecEnv([make_env(env_type, replay, seed=999)])

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        device="cuda",
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,                 # Encourage exploration
        learning_rate=3e-4,
        tensorboard_log="./logs/recurrent_ppo_tensorboard/",
        policy_kwargs={
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
            "shared_lstm": False,      # Separate LSTM for actor/critic
            "enable_critic_lstm": True,
        },
    )

    obs_dim = vec_env.observation_space.shape[0]
    log.info(
        "RecurrentPPO model created | obs_dim=%d, lstm=%dx%d",
        obs_dim, lstm_hidden_size, n_lstm_layers,
    )
    print(f"Observation: {obs_dim}D | LSTM: {n_lstm_layers}x{lstm_hidden_size} hidden units")

    Path("checkpoints").mkdir(exist_ok=True)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model.save(save_path)
    log.info("RecurrentPPO training complete | saved to %s.zip", save_path)
    print(f"\nTraining complete. Model saved: {save_path}.zip")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO (LSTM) execution agent")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--format", type=str, default="binance", choices=["binance", "ohlcv"])
    parser.add_argument("--env", type=str, default="lob", choices=["lob", "lo"],
                        help="lob=market orders (9D), lo=limit orders (12D)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save", type=str, default="recurrent_ppo")
    parser.add_argument("--lstm-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    args = parser.parse_args()

    train(
        data_path=args.data,
        fmt=args.format,
        env_type=args.env,
        total_timesteps=args.timesteps,
        save_path=args.save,
        lstm_hidden_size=args.lstm_size,
        n_lstm_layers=args.lstm_layers,
    )


if __name__ == "__main__":
    main()
