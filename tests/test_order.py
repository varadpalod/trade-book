from datetime import datetime

import pytest

from src.market.enums import OrderSide
from src.market.order import Order


def test_create_buy_order():
    order = Order(
        order_id=1,
        side=OrderSide.BUY,
        price=100.5,
        quantity=200,
        timestamp=datetime.now(),
    )

    assert order.side == OrderSide.BUY
    assert order.price == 100.5
    assert order.quantity == 200


def test_invalid_price():
    with pytest.raises(ValueError):
        Order(
            order_id=1,
            side=OrderSide.BUY,
            price=0,
            quantity=100,
            timestamp=datetime.now(),
        )


def test_invalid_quantity():
    with pytest.raises(ValueError):
        Order(
            order_id=1,
            side=OrderSide.SELL,
            price=100,
            quantity=0,
            timestamp=datetime.now(),
        )