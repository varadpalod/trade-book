from dataclasses import dataclass
from datetime import datetime

from .enums import OrderSide

@dataclass
class Order:
    order_id: int
    side: OrderSide
    price: float
    quantity: int
    timestamp: datetime

    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive.")

        if self.quantity <= 0:
            raise ValueError("Quantity must be positive.")