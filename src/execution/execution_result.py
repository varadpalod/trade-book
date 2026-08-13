from dataclasses import dataclass

@dataclass
class ExecutionResult:
    schedule:list[int]
    market_prices:list[float]
    execution_prices:list[float]
    inventory:list[int]
    cash_spent:float
    implementation_shortfall:float
    