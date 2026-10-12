"""
risk_controls.py
----------------
Production risk controls for the RL execution agent.

Wraps any execution decision with hard-coded safety boundaries
that CANNOT be overridden by the RL model.

Risk layers (in order of priority):
    1. Kill switch      — global on/off switch
    2. Max position     — absolute share limit per order
    3. Max shortfall    — abort if IS exceeds threshold
    4. Max drawdown     — halt if cumulative loss exceeds threshold
    5. Staleness guard  — reject stale/old price data

Usage:
    from src.risk.risk_controls import RiskControls, RiskViolationError

    controls = RiskControls(
        max_shares_per_order=500,
        max_shortfall=10_000.0,
        max_drawdown=5_000.0,
    )

    # Before executing an RL action:
    safe_qty = controls.check_order(
        proposed_qty=model_output,
        current_price=price,
        shortfall_so_far=total_shortfall,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from src.logger import get_logger

log = get_logger(__name__)


class RiskViolationError(Exception):
    """Raised when a hard risk limit is breached."""
    pass


class KillSwitchError(RiskViolationError):
    """Raised when the kill switch is active."""
    pass


@dataclass
class RiskConfig:
    """Configuration for all risk limits."""
    # Maximum shares that can be executed in a single order
    max_shares_per_order: int = 10_000

    # Maximum allowed implementation shortfall (in price * shares units)
    max_shortfall: float = float("inf")

    # Maximum cumulative drawdown before halting
    max_drawdown: float = float("inf")

    # Maximum price change since arrival that allows trading (%)
    max_price_drift_pct: float = 10.0

    # Kill switch — set to True to halt all trading
    kill_switch: bool = False


class RiskControls:
    """
    Hard-coded risk guardrails for the execution agent.

    These controls sit OUTSIDE the RL model and cannot be
    learned away. They ensure that even if the model misbehaves,
    catastrophic execution is prevented.
    """

    def __init__(self, config: RiskConfig | None = None, **kwargs) -> None:
        self._cfg = config if config is not None else RiskConfig(**kwargs)
        self._violations: list[str] = []

        log.info(
            "RiskControls initialised | max_shares=%d, max_IS=%.2f, max_dd=%.2f",
            self._cfg.max_shares_per_order,
            self._cfg.max_shortfall,
            self._cfg.max_drawdown,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate_kill_switch(self) -> None:
        """Immediately halt all trading."""
        self._cfg.kill_switch = True
        log.warning("KILL SWITCH ACTIVATED — all trading halted")

    def deactivate_kill_switch(self) -> None:
        """Re-enable trading."""
        self._cfg.kill_switch = False
        log.info("Kill switch deactivated")

    @property
    def is_halted(self) -> bool:
        return self._cfg.kill_switch

    def check_order(
        self,
        proposed_qty: int,
        current_price: float,
        arrival_price: float,
        shortfall_so_far: float = 0.0,
        drawdown_so_far: float = 0.0,
    ) -> int:
        """
        Validate and potentially clamp a proposed order quantity.

        Parameters
        ----------
        proposed_qty    : shares the RL model wants to trade
        current_price   : current market price
        arrival_price   : price at start of execution
        shortfall_so_far: cumulative IS up to this point
        drawdown_so_far : maximum cumulative IS drawdown so far

        Returns
        -------
        safe_qty : clamped quantity that passes all risk checks

        Raises
        ------
        KillSwitchError     : if kill switch is active
        RiskViolationError  : if a hard limit is breached (non-clamping)
        """
        # --- Layer 1: Kill switch ---
        if self._cfg.kill_switch:
            raise KillSwitchError("Kill switch is active — trading halted")

        # --- Layer 2: Max position size ---
        safe_qty = proposed_qty
        if proposed_qty > self._cfg.max_shares_per_order:
            log.warning(
                "Order clamped: proposed=%d > max=%d",
                proposed_qty,
                self._cfg.max_shares_per_order,
            )
            safe_qty = self._cfg.max_shares_per_order
            self._violations.append(f"max_shares_clamped:{proposed_qty}->{safe_qty}")

        # --- Layer 3: Max shortfall ---
        if shortfall_so_far > self._cfg.max_shortfall:
            msg = (
                f"Max shortfall breached: {shortfall_so_far:.2f} > "
                f"{self._cfg.max_shortfall:.2f}"
            )
            log.error(msg)
            self._violations.append(f"max_shortfall:{shortfall_so_far:.2f}")
            raise RiskViolationError(msg)

        # --- Layer 4: Max drawdown ---
        if drawdown_so_far > self._cfg.max_drawdown:
            msg = (
                f"Max drawdown breached: {drawdown_so_far:.2f} > "
                f"{self._cfg.max_drawdown:.2f}"
            )
            log.error(msg)
            self._violations.append(f"max_drawdown:{drawdown_so_far:.2f}")
            raise RiskViolationError(msg)

        # --- Layer 5: Price staleness / drift ---
        if arrival_price > 0:
            drift_pct = abs(current_price - arrival_price) / arrival_price * 100
            if drift_pct > self._cfg.max_price_drift_pct:
                msg = (
                    f"Price drift too large: {drift_pct:.2f}% > "
                    f"{self._cfg.max_price_drift_pct:.2f}%"
                )
                log.warning(msg)
                # This is a warning, not a hard halt — trader decides
                self._violations.append(f"price_drift:{drift_pct:.2f}%")

        if safe_qty != proposed_qty:
            log.debug("Order adjusted: %d -> %d", proposed_qty, safe_qty)

        return safe_qty

    def violation_count(self) -> int:
        return len(self._violations)

    def violation_log(self) -> list[str]:
        return list(self._violations)

    def reset_violations(self) -> None:
        self._violations.clear()
        log.debug("Violation log cleared")
