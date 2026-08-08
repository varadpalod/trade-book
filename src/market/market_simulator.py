import random
from datetime import datetime
from .matching_engine import MatchingEngine
from .enums import OrderSide
from .order import Order
from .order_book import LimitOrderBook
from .trade import Trade

class MarketSimulator:
    def __init__(self,initial_price:float=100.0,volatility:float=0.5,max_quantity:int=100,price_range:float=2.0,):
        self.order_book=LimitOrderBook()
        self.engine=MatchingEngine(self.order_book)
        self.order_id=1
        self.mid_price=initial_price
        self.volatility=volatility
        self.max_quantity=max_quantity
        self.price_range=price_range

        self.total_orders=0
        self.total_volume=0
        self.total_trades=0

    def submit_order(self,order:Order)->list[Trade]:
        return self.engine.process_order(order)

    @property
    def trade_history(self)->list[Trade]:
        return self.engine.trade_history

    def generate_random_order(self) -> Order:
        side = random.choice([OrderSide.BUY, OrderSide.SELL])
        price = round( random.uniform(self.mid_price - self.price_range,self.mid_price + self.price_range,),2, )
        quantity = random.randint(1, self.max_quantity)
        order = Order(order_id=self.order_id,side=side,price=price,quantity=quantity,timestamp=datetime.now(),)
        self.order_id += 1
        return order

    def simulate_step(self)->list[Trade]:
        self.update_mid_price()
        order = self.generate_random_order()
        trades=self.submit_order(order)

        self.total_orders+=1
        self.total_trades+=len(trades)

        for trade in trades:
            self.total_volume+=trade.quantity

        return self.submit_order(order)

    def update_mid_price(self):
        change=random.uniform(-self.volatility,self.volatility)

        self.mid_price=round(max(1,self.mid_price+change),2,)

    def run(self,steps:int):
        for _ in range(steps):
            self.simulate_step()

    def simulation_summary(self) -> dict:
        return {
            "orders": self.total_orders,
            "trades": self.total_trades,
            "volume": self.total_volume,
            "mid_price": self.mid_price,
        }