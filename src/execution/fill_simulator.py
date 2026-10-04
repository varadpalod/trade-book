"""
fill_simulator.py
-----------------
Probabilistic Limit Order Fill Simulator for Phase 8.

Simulates the queue mechanics of a Central Limit Order Book (CLOB).
When the SAC agent posts a Limit Order at a given price level, this
simulator determines whether that order gets filled based on:

    1. Price priority  — order executes if market trades AT or THROUGH our price
    2. Queue position  — estimated fill probability based on depth at that level
    3. Time in force   — unfilled orders are cancelled at the next decision step

This mirrors the real-world mechanics of both:
    - Crypto (Binance): Sub-second fills on liquid markets
    - Indian Equities (NSE): Same CLOB rules, T+1 settlement

Fill Model:
    P(fill | price_level, qty, ask_vol) = min(1, (ticks_through * avg_tick_vol) / (queue_ahead + qty))

Design decisions:
    - We do NOT simulate the full L3 order book (too complex, not needed for RL)
    - We use a conservative queue-fraction model based on aggregated tick volume
    - Unfilled fraction gets carried forward for re-decision next step
"""
from __future__ import annotations

from dataclasses import dataclass, field
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class PendingOrder:
    """Represents a single outstanding Limit Order in the queue."""
    price: float          # Limit price posted
    qty: int              # Shares/units remaining unfilled
    side: str             # 'sell' or 'buy'
    step_posted: int      # Environment step when this was posted
    queue_fraction: float = 0.5  # Estimated fraction of depth ahead of us in queue


class FillSimulator:
    """
    Simulates limit order fills from historical tick data.

    The simulator is stateful — it holds pending orders across environment
    steps and checks each incoming market snapshot for a fill event.

    Parameters
    ----------
    fill_model          : 'conservative' | 'aggressive'
                          conservative = only fills when market trades THROUGH our price
                          aggressive   = fills if market trades AT our price (optimistic)
    min_fill_fraction   : minimum fraction to fill per tick (prevents tiny fills)
    cancel_after_steps  : cancel any unfilled order after N steps (re-decision)
    """

    def __init__(
        self,
        fill_model: str = "conservative",
        min_fill_fraction: float = 0.0,
        cancel_after_steps: int = 1,
    ) -> None:
        self._fill_model = fill_model
        self._min_fill_fraction = min_fill_fraction
        self._cancel_after_steps = cancel_after_steps
        self._pending: list[PendingOrder] = []
        self._total_filled = 0
        self._total_cancelled = 0

        log.debug(
            "FillSimulator created | model=%s, cancel_after=%d",
            fill_model, cancel_after_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post_limit_order(
        self,
        price: float,
        qty: int,
        side: str,
        current_step: int,
        ask_volume: float = 0.0,
    ) -> None:
        """
        Add a new limit order to the pending queue.

        Parameters
        ----------
        price        : limit price (e.g., best_bid, mid, best_ask)
        qty          : number of units to execute
        side         : 'sell' to liquidate inventory
        current_step : current env step (for cancel_after tracking)
        ask_volume   : current ask depth (used for queue estimation)
        """
        if qty <= 0:
            return

        # Estimate queue fraction: assume we're behind half the current depth
        queue_fraction = min(0.9, ask_volume / max(ask_volume + qty, 1.0))

        order = PendingOrder(
            price=price,
            qty=qty,
            side=side,
            step_posted=current_step,
            queue_fraction=queue_fraction,
        )
        self._pending.append(order)
        log.debug(
            "Limit order posted | price=%.4f, qty=%d, side=%s, queue_frac=%.2f",
            price, qty, side, queue_fraction,
        )

    def process_tick(
        self,
        current_step: int,
        mid_price: float,
        ask_price: float,
        bid_price: float,
        traded_volume: float,
        last_traded_price: float,
    ) -> tuple[int, int]:
        """
        Advance the simulator by one market tick.

        Checks all pending orders against the current market state and
        determines which fraction of each order gets filled.

        Returns
        -------
        (filled_qty, cancelled_qty) for this tick
        """
        filled_this_tick = 0
        cancelled_this_tick = 0

        still_pending = []
        for order in self._pending:
            # --- Cancel check ---
            steps_alive = current_step - order.step_posted
            if steps_alive >= self._cancel_after_steps:
                log.debug(
                    "Limit order CANCELLED | price=%.4f, qty=%d, alive=%d steps",
                    order.price, order.qty, steps_alive,
                )
                cancelled_this_tick += order.qty
                self._total_cancelled += order.qty
                continue

            # --- Fill check (SELL order logic) ---
            filled = 0
            if order.side == "sell":
                if self._fill_model == "conservative":
                    # Fill only if market trades AT or ABOVE our limit price
                    should_fill = last_traded_price >= order.price
                else:
                    # Aggressive: fill if market is anywhere near our price
                    should_fill = last_traded_price >= order.price * 0.9995

                if should_fill:
                    # Estimate fill based on traded volume vs queue depth
                    queue_ahead = order.queue_fraction * traded_volume
                    available_to_fill = max(0.0, traded_volume - queue_ahead)
                    fill_fraction = min(1.0, available_to_fill / max(order.qty, 1))
                    fill_fraction = max(self._min_fill_fraction, fill_fraction)

                    filled = int(fill_fraction * order.qty)
                    order.qty -= filled
                    filled_this_tick += filled
                    self._total_filled += filled

                    log.debug(
                        "Limit FILL | price=%.4f, filled=%d, remaining=%d",
                        order.price, filled, order.qty,
                    )

            if order.qty > 0:
                still_pending.append(order)

        self._pending = still_pending
        return filled_this_tick, cancelled_this_tick

    def get_pending_qty(self) -> int:
        """Total quantity still sitting in the limit order queue."""
        return sum(o.qty for o in self._pending)

    def cancel_all(self) -> int:
        """Force-cancel all pending orders. Returns qty cancelled."""
        cancelled = self.get_pending_qty()
        self._total_cancelled += cancelled
        self._pending.clear()
        log.debug("All pending orders force-cancelled | qty=%d", cancelled)
        return cancelled

    def reset(self) -> None:
        """Reset simulator state for a new episode."""
        self._pending.clear()
        self._total_filled = 0
        self._total_cancelled = 0

    @property
    def stats(self) -> dict:
        return {
            "total_filled": self._total_filled,
            "total_cancelled": self._total_cancelled,
            "pending_qty": self.get_pending_qty(),
        }
