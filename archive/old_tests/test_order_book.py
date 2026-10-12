from datetime import datetime

import pytest

from src.market.enums import OrderSide
from src.market.order import Order
from src.market.order_book import LimitOrderBook


@pytest.fixture
def book():
    return LimitOrderBook()


def test_empty_book(book):
    assert book.bids == []
    assert book.asks == []


def test_add_buy_order(book):
    order = Order(
        order_id=1,
        side=OrderSide.BUY,
        price=100,
        quantity=50,
        timestamp=datetime.now(),
    )

    book.add_order(order)

    assert len(book.bids) == 1
    assert len(book.asks) == 0


def test_add_sell_order(book):
    order = Order(
        order_id=2,
        side=OrderSide.SELL,
        price=101,
        quantity=20,
        timestamp=datetime.now(),
    )

    book.add_order(order)

    assert len(book.asks) == 1
    assert len(book.bids) == 0


def test_best_bid(book):
    book.add_order(Order(1, OrderSide.BUY, 100, 10, datetime.now()))
    book.add_order(Order(2, OrderSide.BUY, 105, 10, datetime.now()))
    book.add_order(Order(3, OrderSide.BUY, 103, 10, datetime.now()))

    assert book.best_bid().price == 105


def test_best_ask(book):
    book.add_order(Order(1, OrderSide.SELL, 110, 10, datetime.now()))
    book.add_order(Order(2, OrderSide.SELL, 108, 10, datetime.now()))
    book.add_order(Order(3, OrderSide.SELL, 109, 10, datetime.now()))

    assert book.best_ask().price == 108


def test_spread(book):
    book.add_order(Order(1, OrderSide.BUY, 99, 10, datetime.now()))
    book.add_order(Order(2, OrderSide.SELL, 101, 10, datetime.now()))

    assert book.spread() == 2


def test_mid_price(book):
    book.add_order(Order(1, OrderSide.BUY, 99, 10, datetime.now()))
    book.add_order(Order(2, OrderSide.SELL, 101, 10, datetime.now()))

    assert book.mid_price() == 100


def test_remove_buy_order(book):
    order = Order(
        1,
        OrderSide.BUY,
        100,
        10,
        datetime.now(),
    )

    book.add_order(order)
    book.remove_order(1)

    assert len(book.bids) == 0


def test_remove_sell_order(book):
    order = Order(
        2,
        OrderSide.SELL,
        101,
        10,
        datetime.now(),
    )

    book.add_order(order)
    book.remove_order(2)

    assert len(book.asks) == 0


def test_remove_nonexistent_order(book):
    book.remove_order(999)

    assert len(book.bids) == 0
    assert len(book.asks) == 0


def test_depth(book):
    book.add_order(Order(1, OrderSide.BUY, 100, 10, datetime.now()))
    book.add_order(Order(2, OrderSide.BUY, 99, 10, datetime.now()))
    book.add_order(Order(3, OrderSide.SELL, 101, 10, datetime.now()))

    assert book.bid_depth() == 2
    assert book.ask_depth() == 1
    assert book.total_depth() == 3