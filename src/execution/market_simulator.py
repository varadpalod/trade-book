"""
market_simulator.py
-------------------
Simulates trade execution of a pre-computed schedule against
a GBM price path, using the Almgren-Chriss impact model.

Bug fixed (Phase 1):
    MarketImpact() was instantiated with no arguments, causing
    a TypeError since eta and gamma were required params.
    Now correctly instantiated with the caller-supplied eta/gamma.
"""
from src.execution.execution_result import ExecutionResult
from src.execution.market_impact import MarketImpact
from src.execution.price_process import PriceProcess
from src.logger import get_logger

log = get_logger(__name__)


class MarketSimulator:
    def __init__(self, random_seed: int | None = None) -> None:
        self._price_process = PriceProcess(random_seed)
        log.debug("MarketSimulator initialised | seed=%s", random_seed)

    def execute(
        self,
        schedule: list[int],
        initial_price: float,
        sigma: float,
        eta: float,
        gamma: float,
        execution_horizon: float,
    ) -> ExecutionResult:
        """
        Execute a trade schedule and return a full ExecutionResult.

        Parameters
        ----------
        schedule          : list of shares to trade at each interval
        initial_price     : arrival price (S0)
        sigma             : volatility parameter for GBM
        eta               : temporary impact coefficient
        gamma             : permanent impact coefficient
        execution_horizon : total execution time in seconds
        """
        intervals = len(schedule)
        if intervals == 0:
            raise ValueError("schedule cannot be empty.")

        # BUG FIX: MarketImpact now receives eta and gamma correctly
        impact = MarketImpact(eta=eta, gamma=gamma)

        tau = execution_horizon / intervals
        market_prices = self._price_process.generate_path(
            initial_price=initial_price,
            sigma=sigma,
            intervals=intervals,
            tau=tau,
        )

        execution_prices: list[float] = []
        inventory: list[int] = []
        cash_spent = 0.0
        remaining = sum(schedule)

        log.info(
            "Executing schedule | intervals=%d, total_shares=%d, tau=%.2fs",
            intervals,
            remaining,
            tau,
        )

        for i, trade in enumerate(schedule):
            inventory.append(remaining)
            market_price = market_prices[i]

            permanent = impact.permanent_impact(trade_size=trade)
            temporary = impact.temporary_impact(trade_size=trade, tau=tau)

            execution_price = market_price + permanent + temporary
            execution_prices.append(execution_price)
            cash_spent += execution_price * trade
            remaining -= trade

            log.debug(
                "Step %d | trade=%d, mkt_price=%.4f, exec_price=%.4f, remaining=%d",
                i,
                trade,
                market_price,
                execution_price,
                remaining,
            )

        inventory.append(remaining)
        arrival_cost = initial_price * sum(schedule)
        implementation_shortfall = cash_spent - arrival_cost

        log.info(
            "Execution complete | IS=%.4f, cash_spent=%.2f",
            implementation_shortfall,
            cash_spent,
        )

        return ExecutionResult(
            schedule=schedule,
            market_prices=market_prices,
            execution_prices=execution_prices,
            inventory=inventory,
            cash_spent=cash_spent,
            implementation_shortfall=implementation_shortfall,
        )