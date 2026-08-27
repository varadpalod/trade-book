from stable_baselines3 import TD3
from src.rl.execution_env import ExecutionEnv
from src.execution.almgren_chriss import AlmgrenChrissStrategy

def evaluate_td3(shared_market_prices):
    env=ExecutionEnv()
    obs,_=env.reset()

    env.market_prices = shared_market_prices.copy()
    env.current_price = env.market_prices[0]
    env.arrival_price = env.market_prices[0]

    model=TD3.load("td3_execution")
    done=False
    total_reward=0.0
    actions=[]

    step=1

    while not done:
        action,_=model.predict(obs,deterministic=True)
        actions.append(float(action[0]))
        obs,reward,terminated,truncated,info=env.step(action)

        print("-" * 70)
        print(f"Step: {step}")
        print(f"Action: {float(action[0]):.4f}")
        print(f"Market Price: {info['current_price']:.4f}")
        print(f"Shares Sold: {info['shares_sold']}")
        print(f"Temporary Impact: {info['temporary_impact']:.6f}")
        print(f"Permanent Impact: {info['permanent_impact']:.6f}")
        print(f"Execution Price: {info['execution_price']:.4f}")
        print(f"Implementation Shortfall: {info['implementation_shortfall']:.4f}")
        print(f"Reward: {reward:.4f}")
        total_reward+=reward
        done= terminated or truncated
        step+=1

    print("TD3 Steps:", len(actions))
    print("TD3 Actions:", actions)
    print("TD3 Total Reward:", total_reward)

    return {
        "reward":total_reward,"actions":actions,"remaining_inventory":env.remaining_shares,
    }

def evaluate_almgren(shared_market_prices):
    env=ExecutionEnv()
    env.reset()
    env.market_prices = shared_market_prices.copy()
    env.current_price = env.market_prices[0]
    env.arrival_price = env.market_prices[0]

    strategy=AlmgrenChrissStrategy()

    schedule=strategy.generate_schedule(total_shares=env.total_shares,execution_horizon=env.execution_horizon,intervals=env.intervals,)
    total_reward=0.0
    remaining_inventory=env.total_shares
    trades=[]
    tau=env.execution_horizon/env.intervals

    for i, shares_to_sell in enumerate(schedule):
        current_price=env.market_prices[i]
        temporary_impact=env.market_impact.temporary_impact(trade_size=shares_to_sell,tau=tau,)
        permanent_impact=env.market_impact.permanent_impact(trade_size=shares_to_sell,)
        execution_price=current_price-temporary_impact-permanent_impact
        implementation_shortfall=(env.arrival_price-execution_price)*shares_to_sell

        reward= -implementation_shortfall
        total_reward+=reward
        remaining_inventory-=shares_to_sell
        trades.append(shares_to_sell)

        print("-" * 70)
        print(f"Step: {i+1}")
        print(f"Market Price: {current_price:.4f}") 
        print(f"Shares Sold: {shares_to_sell}")
        print(f"Remaining Inventory: {remaining_inventory}")
        print(f"Temporary Impact: {temporary_impact:.6f}")
        print(f"Permanent Impact: {permanent_impact:.6f}")
        print(f"Execution Price: {execution_price:.4f}")
        print(f"Implementation Shortfall: {implementation_shortfall:.4f}")
        print(f"Reward: {reward:.4f}")

    print("AC Schedule:", schedule)
    print("Sum:", sum(schedule))

    print("=" * 70)
    print("Evaluation Parameters")
    print(f"Sigma: {env.sigma}")
    print(f"Eta: {env.eta}")
    print(f"Gamma: {env.gamma}")
    print(f"Intervals: {env.intervals}")
    print(f"Execution Horizon: {env.execution_horizon}")
    print(f"Arrival Price: {env.arrival_price:.4f}")
    print("=" * 70)
    return {
        "reward":total_reward,"actions":trades,"remaining_inventory":remaining_inventory,
    }

def compare():
    env = ExecutionEnv()
    env.reset()
    shared_market_prices = env.market_prices.copy()
    td3_results=evaluate_td3(shared_market_prices)
    ac_results = evaluate_almgren(shared_market_prices)

    print("="*50)
    print("TD3:",td3_results)
    print("AC:", ac_results)

if __name__=="__main__":
    compare()
