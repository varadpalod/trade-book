from src.execution.benchmark import BenchmarkRunner
from src.execution.execution_report import ExecutionReport


def test_add_result():
    runner = BenchmarkRunner()

    report = ExecutionReport(
        total_quantity=100,
        executed_quantity=100,
        remaining_quantity=0,
        average_execution_price=100,
        implementation_shortfall=0,
    )

    runner.add_result("TWAP", report)

    assert runner.strategy_count() == 1

def test_best_average_price():
    runner = BenchmarkRunner()

    runner.add_result(
        "TWAP",
        ExecutionReport(
            100,
            100,
            0,
            101,
            100,
        ),
    )

    runner.add_result(
        "VWAP",
        ExecutionReport(
            100,
            100,
            0,
            99,
            -100,
        ),
    )

    best = runner.best_average_price()

    assert best.strategy_name == "VWAP"

def test_empty_benchmark():
    runner = BenchmarkRunner()

    assert runner.strategy_count() == 0