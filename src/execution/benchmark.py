from dataclasses import dataclass
from .execution_result import ExecutionReport
from execution.market_simulator import MarketSimulator 
import csv

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

    def run_strategy(self,strategy,strategy_name:str,steps:int,target_quantity:int,arrival_price:float):
        simulator=MarketSimulator()
        simulator.run_with_strategy(strategy,steps)
        report=simulator.generate_execution_report(target_quantity=target_quantity,arrival_price=arrival_price)
        self.add_result(strategy_name,report)
        return report 

    def rank_by_average_price(self):
        return sorted(self.results,key=lambda r:r.report.average_execution_price,)

    def rank_by_shortfall(self):
        return sorted(self.results,key=lambda r:r.report.implementation_shortfall,)

    def summary(self):
        return [
            {
            "strategy": result.strategy_name,
            "average_price": result.report.average_execution_price,
            "shortfall": result.report.implementation_shortfall,
            "executed": result.report.executed_quantity,

        }
        for result in self.results
    ]

    def export_csv(self,filename:str):
        with open(filename,"w",newline="") as file:
            writer=csv.writer(file)

            writer.writerow(
                [
                    "Strategy","Average Price","Implementation Shortfall","Executed Quantity","Remaining Quantity",
                ]
            )

            for result in self.results:
                writer.writerow(
                    [
                        result.strategy_name,result.report.average_execution_price,result.report.implementation_shortfall,
                        result.report.executed_quantity,result.report.remaining_quantity
                    ]
                )