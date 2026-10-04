"""
market_impact.py
----------------
Almgren-Chriss market impact model.

Temporary impact: eta * (n / tau)^2
    - Proportional to the square of the participation rate
    - Decays after the trade (transient)

Permanent impact: gamma * n
    - Proportional to trade size
    - Persists in the market price
"""
from src.logger import get_logger

log = get_logger(__name__)


class MarketImpact:
    """
    Computes temporary and permanent market impact
    using the linear Almgren-Chriss impact model.

    Parameters
    ----------
    eta   : temporary impact coefficient (default: 2.5e-6)
    gamma : permanent impact coefficient (default: 2.5e-7)
    """

    def __init__(self, eta: float = 2.5e-6, gamma: float = 2.5e-7) -> None:
        if eta < 0:
            raise ValueError("eta cannot be negative.")
        if gamma < 0:
            raise ValueError("gamma cannot be negative.")

        self.eta = eta
        self.gamma = gamma
        log.debug("MarketImpact initialised | eta=%.2e, gamma=%.2e", eta, gamma)

    def permanent_impact(self, trade_size: float) -> float:
        """
        Permanent price impact: gamma * n

        Parameters
        ----------
        trade_size : number of shares traded in this interval
        """
        if trade_size < 0:
            raise ValueError("trade_size cannot be negative.")

        return self.gamma * trade_size

    def temporary_impact(self, trade_size: float, tau: float) -> float:
        """
        Temporary price impact: eta * (n / tau)^2

        Parameters
        ----------
        trade_size : number of shares traded in this interval
        tau        : duration of the interval in seconds
        """
        if trade_size < 0:
            raise ValueError("trade_size cannot be negative.")
        if tau <= 0:
            raise ValueError("tau must be positive.")

        participation_rate = trade_size / tau
        return self.eta * (participation_rate ** 2)

    def total_impact(self, trade_size: float, tau: float) -> float:
        """Convenience: permanent + temporary impact."""
        return self.permanent_impact(trade_size) + self.temporary_impact(trade_size, tau)
