from src.rl.execution_env import ExecutionEnv


def run_schedule(schedule):
    env = ExecutionEnv()

    obs, info = env.reset()

    total_reward = 0

    for shares in schedule:
        if env.remaining_shares == 0:
            break

        action = [shares / env.remaining_shares]

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    return total_reward


if __name__ == "__main__":
    schedules = {
        "Immediate": [1000, 0, 0, 0, 0],
        "Equal": [200, 200, 200, 200, 200],
        "AC": [389, 263, 168, 105, 75],
    }

    for name, schedule in schedules.items():
        reward = run_schedule(schedule)
        print(name, reward)