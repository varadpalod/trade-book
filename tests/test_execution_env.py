import numpy as np
import pytest

from src.rl.execution_env import ExecutionEnv


@pytest.fixture
def env():
    return ExecutionEnv()


def test_reset_returns_valid_observation(env):
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(info, dict)


def test_step_returns_gymnasium_tuple(env):
    env.reset()

    result = env.step(np.array([0.5], dtype=np.float32))

    assert len(result) == 5


def test_inventory_never_negative(env):
    env.reset()

    done = False

    while not done:
        obs, reward, terminated, truncated, info = env.step(
            np.array([1.0], dtype=np.float32)
        )
        done = terminated or truncated

        assert env.remaining_shares >= 0


def test_episode_terminates(env):
    env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(
            np.array([1.0], dtype=np.float32)
        )

    assert terminated


def test_reward_is_numeric(env):
    env.reset()

    _, reward, _, _, _ = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert isinstance(reward, (int, float, np.floating))


def test_observation_shape_after_step(env):
    env.reset()

    obs, _, _, _, _ = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert obs.shape == (4,)


def test_action_clipping(env):
    env.reset()

    env.step(np.array([2.0], dtype=np.float32))

    assert env.remaining_shares >= 0


def test_reset_restores_initial_state(env):
    env.reset()

    env.step(np.array([1.0], dtype=np.float32))

    env.reset()

    assert env.remaining_shares == env.total_shares
    assert env.current_step == 0


def test_episode_finishes_with_zero_inventory(env):
    env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(
            np.array([0.2], dtype=np.float32)
        )

    assert env.remaining_shares == 0