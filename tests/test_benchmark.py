from datetime import datetime

from src.execution.benchmark import BenchmarkRunner
from src.execution.execution_report import ExecutionReport
from src.market.enums import OrderSide
from src.market.order import Order
from pathlib import Path


class SingleOrderStrategy:
    def __init__(self):
        self.sent = False

    def next_order(self):
        if self.sent:
            return None

        self.sent = True

        return Order(
            1,
            OrderSide.BUY,
            100,
            10,
            datetime.now(),
        )


def test_empty_benchmark():
    runner = BenchmarkRunner()

    assert runner.strategy_count() == 0


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
            total_quantity=100,
            executed_quantity=100,
            remaining_quantity=0,
            average_execution_price=101,
            implementation_shortfall=100,
        ),
    )

    runner.add_result(
        "VWAP",
        ExecutionReport(
            total_quantity=100,
            executed_quantity=100,
            remaining_quantity=0,
            average_execution_price=99,
            implementation_shortfall=-100,
        ),
    )

    best = runner.best_average_price()

    assert best.strategy_name == "VWAP"


def test_run_strategy_adds_result():
    runner = BenchmarkRunner()
    strategy = SingleOrderStrategy()

    runner.run_strategy(
        strategy=strategy,
        strategy_name="Single",
        steps=5,
        target_quantity=10,
        arrival_price=100,
    )

    assert runner.strategy_count() == 1


def test_run_strategy_returns_report():
    runner = BenchmarkRunner()
    strategy = SingleOrderStrategy()

    report = runner.run_strategy(
        strategy=strategy,
        strategy_name="Single",
        steps=5,
        target_quantity=10,
        arrival_price=100,
    )

    assert report.total_quantity == 10
    assert isinstance(report, ExecutionReport)

def test_rank_by_average_price():
    runner = BenchmarkRunner()

    runner.add_result(
        "TWAP",
        ExecutionReport(100, 100, 0, 102, 200),
    )

    runner.add_result(
        "VWAP",
        ExecutionReport(100, 100, 0, 99, -100),
    )

    ranking = runner.rank_by_average_price()

    assert ranking[0].strategy_name == "VWAP"
    assert ranking[1].strategy_name == "TWAP"

def test_rank_by_shortfall():
    runner = BenchmarkRunner()

    runner.add_result(
        "TWAP",
        ExecutionReport(100, 100, 0, 101, 100),
    )

    runner.add_result(
        "VWAP",
        ExecutionReport(100, 100, 0, 99, -100),
    )

    ranking = runner.rank_by_shortfall()

    assert ranking[0].strategy_name == "VWAP"

def test_summary():
    runner = BenchmarkRunner()

    runner.add_result(
        "TWAP",
        ExecutionReport(100, 90, 10, 100, 0),
    )

    summary = runner.summary()

    assert len(summary) == 1
    assert summary[0]["strategy"] == "TWAP"
    assert summary[0]["executed"] == 90

def test_export_csv(tmp_path):
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

    file = tmp_path / "benchmark.csv"

    runner.export_csv(file)

    assert file.exists()

    content = file.read_text()

    assert "TWAP" in content
    assert "Average Price" in content

def test_export_multiple_results(tmp_path):
    runner = BenchmarkRunner()

    runner.add_result(
        "TWAP",
        ExecutionReport(100, 100, 0, 101, 100),
    )

    runner.add_result(
        "VWAP",
        ExecutionReport(100, 100, 0, 99, -100),
    )

    file = tmp_path / "results.csv"

    runner.export_csv(file)

    content = file.read_text()

    assert "TWAP" in content
    assert "VWAP" in content
