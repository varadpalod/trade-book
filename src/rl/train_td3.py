from stable_baselines3 import TD3
from .execution_env import ExecutionEnv

def main():
    env=ExecutionEnv()
    model=TD3(policy="MlpPolicy",env=env,device="cuda",verbose=1,learning_rate=1e-3,buffer_size=100_000,learning_starts=1_000,batch_size=256,gamma=0.99,tau=0.005,train_freq=1,gradient_steps=1,)
    model.learn(total_timesteps=10_000)
    model.save("td3_execution")
    print("training complete")

if __name__=="__main__":
    main()