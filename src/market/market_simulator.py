from .matching_engine import MatchingEngine
from .order import Order
from .order_book import LimitOrderBook
from .trade import Trade

class MarketSimulator:
    def __init__(self):
        self.order_book=LimitOrderBook()
        self.engine=MatchingEngine(self.order_book)
        self.order_id=1

    def submit_order(self,order:Order)->list[Trade]:
        return self.engine.process_order(order)

    @property
    def trade_history(self)->list[Trade]:
        return self.engine.trade_history

    