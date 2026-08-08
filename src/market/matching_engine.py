from datetime import datetime

from .enums import OrderSide
from .order import Order
from .order_book import LimitOrderBook
from .trade import Trade

class MatchingEngine:
    def __init__(self,order_book:LimitOrderBook):
        self.order_book=order_book
        self.trade_history :list[Trade]=[]

    def process_order(self, order:Order)->Trade|None:
        if order.side==OrderSide.BUY:
            return self._process_buy(order)

        return self._process_sell(order)
#BUY ORDER PROCESSING
    def _process_buy(self, buy_order:Order)->Trade|None:
        best_ask = self.order_book.best_ask()

        if best_ask is None:
            self.order_book.add_order(buy_order)
            return None

        if buy_order.price < best_ask.price:
            self.order_book.add_order(buy_order)
            return None

        trade_quantity=min(buy_order.quantity,best_ask.quantity)
        trade = Trade(buy_order_id=buy_order.order_id,sell_order_id=best_ask.order_id,price=best_ask.price,quantity=trade_quantity,timestamp=datetime.now(),)

        buy_order.quantity=buy_order.quantity-trade_quantity
        best_ask.quantity=best_ask.quantity-trade_quantity

        if best_ask.quantity == 0 :
            self.order_book.remove_order(best_ask.order_id)

        if buy_order.quantity>0:
            self.order_book.add_order(buy_order)

        self.trade_history.append(trade)

        return trade


    
    def _process_sell(self, sell_order:Order)->Trade|None:
        best_bid = self.order_book.best_bid()

        if best_bid is None:
            self.order_book.add_order(sell_order)
            return None

        if sell_order.price > best_bid.price:
            self.order_book.add_order(sell_order)
            return None

        trade_quantity=min(sell_order.quantity,best_bid.quantity)
        trade = Trade(buy_order_id=best_bid.order_id,sell_order_id=sell_order.order_id,price=best_bid.price,quantity=trade_quantity,timestamp=datetime.now(),)

        sell_order.quantity-=trade_quantity
        best_bid.quantity-=trade_quantity

        if best_bid.quantity==0:
            self.order_book.remove_order(best_bid.order_id)

        if sell_order.quantity>0:
            self.order_book.add_order(sell_order)
        
        self.trade_history.append(trade)

        return trade