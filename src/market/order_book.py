from .enums import OrderSide
from .order import Order


class LimitOrderBook:
    """
    Simple in-memory Limit Order Book.
    Stores buy and sell orders and provides
    basic market statistics.
    """

    def __init__(self):
        self.bids: list[Order] = []
        self.asks: list[Order] = []

    def add_order(self, order: Order) -> None:
        """
        Add an order to the appropriate side of the book.
        """
        if order.side == OrderSide.BUY:
            self.bids.append(order)
        else:
            self.asks.append(order)

    def remove_order(self, order_id: int) -> None:
        """
        Remove an order from the order book.
        """

        self.bids = [
            order
            for order in self.bids
            if order.order_id != order_id
        ]

        self.asks = [
            order
            for order in self.asks
            if order.order_id != order_id
        ]

    def best_bid(self) -> Order | None:
        """
        Returns the highest priced buy order.
        """

        if not self.bids:
            return None

        return max(self.bids, key=lambda order: order.price)

    def best_ask(self) -> Order | None:
        """
        Returns the lowest priced sell order.
        """

        if not self.asks:
            return None

        return min(self.asks, key=lambda order: order.price)

    def spread(self) -> float | None:
        """
        Returns the bid-ask spread.
        """

        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return ask.price - bid.price

    def mid_price(self) -> float | None:
        """
        Returns the market mid price.
        """

        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return (bid.price + ask.price) / 2

    def bid_depth(self) -> int:
        """
        Returns the number of buy orders.
        """
        return len(self.bids)

    def ask_depth(self) -> int:
        """
        Returns the number of sell orders.
        """
        return len(self.asks)

    def total_depth(self) -> int:
        """
        Returns the total number of orders.
        """
        return self.bid_depth() + self.ask_depth()

    def __repr__(self) -> str:
        return (
            f"LimitOrderBook("
            f"bids={len(self.bids)}, "
            f"asks={len(self.asks)})"
        )