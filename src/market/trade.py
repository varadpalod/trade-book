from datetime import datetime
from dataclasses import dataclass

@dataclass
class Trade:
    buy_order_id:int
    sell_order_id:int
    price:float
    quantity:int
    timestamp:datetime

