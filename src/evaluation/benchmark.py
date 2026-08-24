from stable_baselines3 import TD3
from src.rl.execution_env import ExecutionEnv
from src.execution.almgren_chriss import AlmgrenChrissStrategy

def evaluate_td3():
    env=ExecutionEnv()
    model=TD3.load("td3_execution")

    obs,_=env.reset()

    done=False
    total_reward=0.0
    actions=[]

    while not done:
        action,_=model.predict(obs,deterministic=True)
        actions.append(float(action[0]))
        obs,reward,terminated,truncated,info=env.step(action)
        total_reward+=reward
        done= terminated or truncated

    print("TD3 Steps:", len(actions))
    print("TD3 Actions:", actions)
    print("TD3 Total Reward:", total_reward)
    return {
        "reward":total_reward,"actions":actions,"remaining_inventory":env.remaining_shares,
    }

def evaluate_almgren():
    env=ExecutionEnv()
    env.reset()
    strategy=AlmgrenChrissStrategy()

    schedule=strategy.generate_schedule(total_shares=env.total_shares,execution_horizon=env.execution_horizon,intervals=env.intervals,)
    total_reward=0.0
    remaining_inventory=env.total_shares
    trades=[]
    tau=env.execution_horizon/env.intervals

    for i, shares_to_sell in enumerate(schedule):
        current_price=env.market_prices[i]
        temporary_impact=env.market_impact.temporary_impact(trade_size=shares_to_sell,tau=tau,)
        execution_price=current_price-temporary_impact
        implementation_shortfall=(env.market_prices[0]-execution_price)*shares_to_sell

        reward= -implementation_shortfall
        total_reward+=reward
        remaining_inventory-=shares_to_sell
        trades.append(shares_to_sell)

    print("AC Schedule:", schedule)
    print("Sum:", sum(schedule))
    return {
        "reward":total_reward,"actions":trades,"remaining_inventory":remaining_inventory,
    }

def compare():
    td3_results=evaluate_td3()
    ac_results = evaluate_almgren()

    print("="*50)
    print("TD3:",td3_results)
    print("AC:", ac_results)

if __name__=="__main__":
    compare()
