from .strategy import ExecutionStrategy
from .cost_model import ExecutionCostModel
from .discrete_optimizer import DiscreteOptimizer


class AlmgrenChrissStrategy(ExecutionStrategy):

    def __init__(self,lambda_risk: float = 2e-6,sigma: float = 0.95,eta: float = 2.5e-6,gamma: float = 2.5e-7,):
        if lambda_risk < 0:
            raise ValueError("lambda_risk cannot be negative.")

        self.lambda_risk = lambda_risk
        self.cost_model = ExecutionCostModel(sigma=sigma,eta=eta,gamma=gamma,)

        self.optimizer = DiscreteOptimizer()

    def generate_schedule(self,total_shares: int,execution_horizon: float,intervals: int,market_data=None,) -> list[int]:

        return self.optimizer.optimize(
            total_shares=total_shares,
            execution_horizon=execution_horizon,
            intervals=intervals,
            sigma=self.cost_model.sigma,
            eta=self.cost_model.eta,
            gamma=self.cost_model.gamma,
            lambda_risk=self.lambda_risk,
        )