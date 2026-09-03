import gymnasium as gym
from gymnasium import spaces
import numpy as np 

from src.execution.price_process import PriceProcess
from src.execution.market_impact import MarketImpact

class ExecutionEnv(gym.Env):
    metadata={"render_modes":[]}

    def __init__(self,total_shares:int=1000,intervals:int=10,initial_price:float=100.0,execution_horizon:float=60.0,sigma:float=1.0,eta:float=2.5e-6,gamma:float=2.5e-7,random_seed:int=42):
        super().__init__()
        self.total_shares = total_shares
        self.remaining_shares = self.total_shares

        self.initial_price = initial_price
        self.current_price = self.initial_price

        self.intervals = intervals
        self.current_step = 0

        self.execution_horizon = execution_horizon
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.price_process=PriceProcess(random_seed=random_seed)
        self.market_impact=MarketImpact(eta=self.eta,gamma=self.gamma,)
        self.action_space=spaces.Box(low=0.0,high=1.0,shape=(1,),dtype=np.float32,)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(3,),dtype=np.float32,)

        self.lambda_inventory=0.001
        self.lambda_terminal=10.0

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.price_process._rng=np.random.default_rng(seed)
        
        self.remaining_shares=self.total_shares
        self.current_price=self.initial_price
        self.current_step=0
        tau = self.execution_horizon / self.intervals
        self.market_prices=self.price_process.generate_path(initial_price=self.initial_price,sigma=self.sigma,intervals=self.intervals,tau=tau)
        self.current_price = self.market_prices[0]
        self.arrival_price=self.current_price
        self.total_shortfall = 0
        return self._get_observation(),{}

    def step(self,action):
        action = float(np.clip(action[0], 0.0, 1.0))
        shares_to_sell=int(action*self.remaining_shares)
        shares_to_sell=min(shares_to_sell,self.remaining_shares,)
        self.remaining_shares -= shares_to_sell

        tau=self.execution_horizon/self.intervals
        temporary_impact=self.market_impact.temporary_impact(trade_size=shares_to_sell,tau=tau)
        permenant_impact=self.market_impact.permanent_impact(trade_size=shares_to_sell)
        trade_price = self.current_price
        execution_price=trade_price-(temporary_impact+permenant_impact)
        implementation_shortfall=(self.arrival_price-execution_price)*shares_to_sell
        self.total_shortfall += implementation_shortfall
        execution_cost = (temporary_impact + permenant_impact) * shares_to_sell

        alpha=1.0
        beta=100.0
        inventory_penalty=alpha*(self.remaining_shares/self.total_shares)
        impact_penalty=beta*(temporary_impact+permenant_impact)
        reward= -implementation_shortfall
        reward-=inventory_penalty
        reward-=impact_penalty

        info={
                "shares_sold": shares_to_sell,
                "temporary_impact": temporary_impact,
                "permanent_impact": permenant_impact,
                "execution_price": execution_price,
                "implementation_shortfall": implementation_shortfall,
                "current_price": trade_price,
                "shortfall": self.total_shortfall,
                }

        self.current_step+=1
        if self.current_step<len(self.market_prices):
            self.current_price=self.market_prices[self.current_step]
        terminated=(self.current_step>=self.intervals or self.remaining_shares==0)
        truncated=False
        

        if(self.current_step >= self.intervals and self.remaining_shares>0):
            temporary_impact=self.market_impact.temporary_impact(trade_size=self.remaining_shares,tau=tau)
            permenant_impact=self.market_impact.permanent_impact(trade_size=self.remaining_shares)
            execution_price=(self.current_price-temporary_impact-permenant_impact)
            terminal_shortfall=(self.arrival_price-execution_price)*self.remaining_shares
            self.total_shortfall=terminal_shortfall
            terminal_cost=(temporary_impact + permenant_impact)*self.remaining_shares
            reward-=terminal_cost
            self.remaining_shares=0

        reward/=100
        return self._get_observation(),reward,terminated,truncated,info

    def _get_observation(self):
        price_change = (self.current_price - self.arrival_price) / self.arrival_price
        return np.array(
        [
            self.current_step / self.intervals,
            self.remaining_shares / self.total_shares,
            price_change,
        ],
        dtype=np.float32,
    )
    
