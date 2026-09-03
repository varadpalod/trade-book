from src.execution.execution_result import ExecutionResult
from src.execution.market_impact import MarketImpact
from src.execution.price_process import PriceProcess


class MarketSimulator:
    def __init__(self, random_seed: int | None = None):
        self._price_process = PriceProcess(random_seed)
        self._impact = MarketImpact()

    def execute(self,schedule: list[int],initial_price: float,sigma: float,eta: float,gamma: float,execution_horizon: float,) -> ExecutionResult:
        intervals=len(schedule)
        if intervals==0:
            raise ValueError("interval cant be 0")

        tau=execution_horizon/intervals
        market_prices=self._price_process.generate_path(initial_price=initial_price,sigma=sigma,intervals=intervals,tau=tau,)
        execution_prices=[]
        inventory=[]
        cash_spent=0.0
        remaining=sum(schedule)

        for i, trade in enumerate(schedule):
            inventory.append(remaining)
            market_price=market_prices[i]
            permanent=self._impact.permanent_impact(trade_size=trade,gamma=gamma)
            temporary=self._impact.temporary_impact(trade_size=trade,eta=eta)
            execution_price=(market_price+permanent+temporary)
            execution_prices.append(execution_price)
            cash_spent+=execution_price*trade
            remaining-=trade

        inventory.append(remaining)
        arrival_cost=initial_price*sum(schedule)
        implementation_shortfall=(cash_spent-arrival_cost)

        return ExecutionResult(
            schedule=schedule,
            market_prices=market_prices,
            execution_prices=execution_prices,
            inventory=inventory,
            cash_spent=cash_spent,
            implementation_shortfall=implementation_shortfall,
        )