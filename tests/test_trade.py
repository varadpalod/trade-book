from datetime import datetime

from src.market.trade import Trade


def test_trade_creation():
    trade = Trade(
        buy_order_id=1,
        sell_order_id=2,
        price=100,
        quantity=50,
        timestamp=datetime.now(),
    )

    assert trade.buy_order_id == 1
    assert trade.sell_order_id == 2
    assert trade.price == 100
    assert trade.quantity == 50