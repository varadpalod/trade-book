from datetime import datetime

from src.market.enums import OrderSide
from src.market.matching_engine import MatchingEngine
from src.market.order import Order
from src.market.order_book import LimitOrderBook


def test_buy_order_added_when_no_seller():
    book = LimitOrderBook()
    engine = MatchingEngine(book)

    order = Order(
        1,
        OrderSide.BUY,
        100,
        10,
        datetime.now(),
    )

    trades = engine.process_order(order)

    assert trades ==[]
    assert book.bid_depth() == 1


def test_sell_order_added_when_no_buyer():
    book = LimitOrderBook()
    engine = MatchingEngine(book)

    order = Order(
        1,
        OrderSide.SELL,
        100,
        10,
        datetime.now(),
    )

    trades = engine.process_order(order)

    assert trades==[]
    assert book.ask_depth() == 1


def test_buy_matches_sell():
    book = LimitOrderBook()
    engine = MatchingEngine(book)

    sell = Order(
        1,
        OrderSide.SELL,
        100,
        10,
        datetime.now(),
    )

    engine.process_order(sell)

    buy = Order(
        2,
        OrderSide.BUY,
        101,
        10,
        datetime.now(),
    )

    trades = engine.process_order(buy)

    assert len(trades)==1
    trade=trades[0]
    assert trade.price == 100
    assert trade.quantity == 10

    assert book.bid_depth() == 0
    assert book.ask_depth() == 0


def test_sell_matches_buy():
    book = LimitOrderBook()
    engine = MatchingEngine(book)

    buy = Order(
        1,
        OrderSide.BUY,
        101,
        10,
        datetime.now(),
    )

    engine.process_order(buy)

    sell = Order(
        2,
        OrderSide.SELL,
        100,
        10,
        datetime.now(),
    )

    trades = engine.process_order(sell)

    assert len(trades)==1
    trade=trades[0]
    assert trade.price == 101
    assert trade.quantity == 10

    assert book.bid_depth() == 0
    assert book.ask_depth() == 0

def test_partial_fill_buy_larger():
    book = LimitOrderBook()
    engine = MatchingEngine(book)

    sell = Order(1, OrderSide.SELL, 100, 40, datetime.now())
    engine.process_order(sell)

    buy = Order(2, OrderSide.BUY, 101, 100, datetime.now())

    trades = engine.process_order(buy)
    assert len(trades)==1
    trade=trades[0]
    assert trade.quantity == 40

    assert book.bid_depth() == 1
    assert book.ask_depth() == 0
    assert book.best_bid().quantity == 60

def test_partial_fill_sell_larger():
    book = LimitOrderBook()
    engine = MatchingEngine(book)

    buy = Order(1, OrderSide.BUY, 101, 40, datetime.now())
    engine.process_order(buy)

    sell = Order(2, OrderSide.SELL, 100, 100, datetime.now())

    trades = engine.process_order(sell)
    assert len(trades)==1
    trade=trades[0]
    assert trade.quantity == 40

    assert book.bid_depth() == 0
    assert book.ask_depth() == 1
    assert book.best_ask().quantity == 60

def test_buy_matches_multiple_sell_orders():
    book = LimitOrderBook()
    engine = MatchingEngine(book)

    engine.process_order(
        Order(1, OrderSide.SELL, 100, 20, datetime.now())
    )

    engine.process_order(
        Order(2, OrderSide.SELL, 101, 30, datetime.now())
    )

    engine.process_order(
        Order(3, OrderSide.SELL, 102, 50, datetime.now())
    )

    trades = engine.process_order(
        Order(4, OrderSide.BUY, 103, 100, datetime.now())
    )

    assert len(trades) == 3

    assert trades[0].quantity == 20
    assert trades[0].price == 100

    assert trades[1].quantity == 30
    assert trades[1].price == 101

    assert trades[2].quantity == 50
    assert trades[2].price == 102

    assert book.total_depth() == 0