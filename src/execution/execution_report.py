from dataclasses import dataclass

@dataclass
class ExecutionReport:
    total_quantity:int
    executed_quantity:int
    remaining_quantity:int
    average_execution_price:float
    implementation_shortfall:float
