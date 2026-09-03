from stable_baselines3 import TD3

from src.rl.execution_env import ExecutionEnv


def evaluate(model_path: str, episodes: int = 5):
    env = ExecutionEnv()
    model = TD3.load(model_path)

    for episode in range(episodes):
        obs, _ = env.reset()

        terminated = False
        truncated = False

        total_reward = 0.0
        actions = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            actions.append(float(action[0]))

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

        print(f"Episode {episode + 1}")
        print(f"Total Reward : {total_reward:.2f}")
        print(f"Actions       : {actions}")
        print(f"Remaining Inv : {env.remaining_shares}")
        print("-" * 40)


if __name__ == "__main__":
    evaluate("td3_execution_v1_500")