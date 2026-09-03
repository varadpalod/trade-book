from stable_baselines3 import TD3
from .execution_env import ExecutionEnv

def main():
    env=ExecutionEnv()
    ##model = TD3.load("td3_execution_v1", env=env, device="cuda")
    model=TD3(policy="MlpPolicy",env=env,device="cuda",verbose=1,learning_rate=3e-4,buffer_size=100_000,learning_starts=1_000,batch_size=256,gamma=0.99,tau=0.005,train_freq=1,gradient_steps=1,)
    model.learn(total_timesteps=50_000)
    model.save("td3_execution_le3e_50p3")
    print("training complete")

if __name__=="__main__":
    main()