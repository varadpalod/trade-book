from datetime import datetime

from src.market.enums import OrderSide
from src.market.market_simulator import MarketSimulator
from src.market.order import Order
from src.execution.execution_report import ExecutionReport
from src.market.trade import Trade

def test_market_simulator_creation():
    simulator = MarketSimulator()

    assert simulator.order_book is not None
    assert simulator.engine is not None


def test_submit_order():
    simulator = MarketSimulator()

    buy = Order(
        1,
        OrderSide.BUY,
        100,
        10,
        datetime.now(),
    )

    trades = simulator.submit_order(buy)

    assert trades == []
    assert simulator.order_book.bid_depth() == 1


def test_trade_history_property():
    simulator = MarketSimulator()

    simulator.submit_order(
        Order(
            1,
            OrderSide.SELL,
            100,
            10,
            datetime.now(),
        )
    )

    simulator.submit_order(
        Order(
            2,
            OrderSide.BUY,
            100,
            10,
            datetime.now(),
        )
    )

    assert len(simulator.trade_history) == 1

def test_generate_random_order():
    simulator = MarketSimulator()

    order = simulator.generate_random_order()

    assert order.order_id == 1
    assert order.quantity >= 1
    assert order.quantity <= 100
    assert order.side in (
        OrderSide.BUY,
        OrderSide.SELL,
    )

def test_order_id_increments():
    simulator = MarketSimulator()

    first = simulator.generate_random_order()
    second = simulator.generate_random_order()

    assert first.order_id == 1
    assert second.order_id == 2

def test_simulate_step():
    simulator = MarketSimulator()

    simulator.simulate_step()

    assert simulator.order_book.total_depth() >= 0

def test_update_mid_price():
    simulator = MarketSimulator()

    initial = simulator.mid_price

    simulator.update_mid_price()

    assert simulator.mid_price > 0

def test_custom_market_parameters():
    simulator = MarketSimulator(
        initial_price=150,
        volatility=1.0,
        max_quantity=500,
        price_range=5,
    )

    assert simulator.mid_price == 150
    assert simulator.volatility == 1.0
    assert simulator.max_quantity == 500
    assert simulator.price_range == 5

def test_run_simulation():
    simulator = MarketSimulator()

    simulator.run(50)

    assert simulator.total_orders == 50

def test_simulation_summary():
    simulator = MarketSimulator()

    simulator.run(10)

    summary = simulator.simulation_summary()

    assert summary["orders"] == 10
    assert summary["trades"] >= 0
    assert summary["volume"] >= 0
    assert summary["mid_price"] > 0

class MockStrategy:
    def __init__(self):
        self.called = 0

    def next_order(self):
        self.called += 1
        return None

    def test_strategy_called_each_step():
        simulator = MarketSimulator()
        strategy = MockStrategy()

        simulator.run_with_strategy(strategy, 10)

        assert strategy.called == 10

class BuyStrategy:
    def __init__(self):
        self.sent = False

    def next_order(self):
        if self.sent:
            return None

        self.sent = True

        return Order(
            999,
            OrderSide.BUY,
            100,
            10,
            datetime.now(),
        )

    def test_strategy_submits_order():
        simulator = MarketSimulator()
        strategy = BuyStrategy()

        simulator.run_with_strategy(strategy, 5)

        assert simulator.total_orders >= 6

def test_strategy_called_each_step():
    simulator = MarketSimulator()
    strategy = MockStrategy()

    simulator.run_with_strategy(strategy, 10)

    assert strategy.called == 10

def test_strategy_submits_order():
    simulator = MarketSimulator()
    strategy = BuyStrategy()

    simulator.run_with_strategy(strategy, 5)

    assert simulator.total_orders >= 6

def test_empty_execution_report():
    simulator = MarketSimulator()

    report = simulator.generate_execution_report(
        target_quantity=100,
        arrival_price=100,
    )

    assert report.executed_quantity == 0
    assert report.remaining_quantity == 100
    assert report.average_execution_price == 0

def test_execution_report_after_trade():
    simulator = MarketSimulator()

    simulator.trade_history.append(
        Trade(
            buy_order_id=1,
            sell_order_id=2,
            price=101,
            quantity=20,
            timestamp=datetime.now(),
        )
    )

    report = simulator.generate_execution_report(
        target_quantity=100,
        arrival_price=100,
    )

    assert report.executed_quantity == 20
    assert report.remaining_quantity == 80
    assert report.average_execution_price == 101
    assert report.implementation_shortfall == 20