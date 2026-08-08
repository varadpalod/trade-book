from dataclasses import dataclass
from .execution_report import ExecutionReport

@dataclass
class BenchmarkResult:
    strategy_name:str
    report:ExecutionReport

class BenchmarkRunner:
    def __init__(self):
        self.results:list[BenchmarkResult]=[]

    def add_result(self,strategy_name:str,report:ExecutionReport,):
        self.results.append(BenchmarkResult(strategy_name,report,))

    def strategy_count(self)->int:
        return len(self.results)
    
    def best_average_price(self):
        if not self.results:
            return None

        return min(self.results,key=lambda r:r.report.average_execution_price,)