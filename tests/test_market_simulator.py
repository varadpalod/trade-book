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