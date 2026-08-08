from datetime import datetime

from src.market.enums import OrderSide
from src.market.market_simulator import MarketSimulator
from src.market.order import Order


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