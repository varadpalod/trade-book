"""
metrics.py
----------
Execution quality metrics for benchmarking trade execution strategies.

Phase 5 additions:
    - Sharpe ratio of per-step P&L
    - Maximum drawdown of cumulative shortfall
    - VWAP slippage (execution price vs VWAP)
    - Fill rate (fraction of target shares executed)
    - Market participation rate per interval
"""
from __future__ import annotations

import numpy as np


class ExecutionMetrics:
    """
    Static utility class for computing execution quality metrics.

    All methods accept numpy arrays or lists and return floats.
    """

    # ---------------------------------------------------------------
    # Core execution metrics (existing, unchanged)
    # ---------------------------------------------------------------

    @staticmethod
    def implementation_shortfall(
        arrival_price: float,
        execution_prices: list[float],
        quantities: list[float],
    ) -> float:
        """
        Implementation Shortfall (IS) — the primary metric.

        IS = Sum[(arrival_price - exec_price_i) * qty_i]
        Positive IS = worse than arrival (cost).
        Negative IS = better than arrival (gain).
        """
        execution_prices = np.array(execution_prices)
        quantities = np.array(quantities)
        return float(np.sum((arrival_price - execution_prices) * quantities))

    @staticmethod
    def average_execution_price(
        execution_prices: list[float],
        quantities: list[float],
    ) -> float:
        """Volume-weighted average execution price."""
        execution_prices = np.array(execution_prices)
        quantities = np.array(quantities)
        total_qty = quantities.sum()
        if total_qty == 0:
            return 0.0
        return float(np.sum(execution_prices * quantities) / total_qty)

    @staticmethod
    def execution_revenue(
        execution_prices: list[float],
        quantities: list[float],
    ) -> float:
        """Total cash received from selling all shares."""
        execution_prices = np.array(execution_prices)
        quantities = np.array(quantities)
        return float(np.sum(execution_prices * quantities))

    @staticmethod
    def fill_rate(total_shares: int, remaining_shares: int) -> float:
        """Fraction of target shares successfully executed ∈ [0, 1]."""
        if total_shares == 0:
            return 0.0
        return float((total_shares - remaining_shares) / total_shares)

    @staticmethod
    def participation_rate(
        total_shares: int,
        quantities: list[float],
    ) -> np.ndarray:
        """Per-interval share of total volume traded."""
        quantities = np.array(quantities)
        if total_shares == 0:
            return np.zeros_like(quantities)
        return quantities / total_shares

    # ---------------------------------------------------------------
    # Phase 5 additions — advanced metrics
    # ---------------------------------------------------------------

    @staticmethod
    def vwap_slippage(
        execution_prices: list[float],
        quantities: list[float],
        market_prices: list[float],
        market_volumes: list[float] | None = None,
    ) -> float:
        """
        VWAP Slippage — execution VWAP vs market VWAP.

        Measures how much better/worse the strategy did vs
        a passive VWAP benchmark.

        Negative = execution was better than market VWAP (good).
        Positive = execution was worse (bad).

        Parameters
        ----------
        execution_prices : prices at which the strategy executed
        quantities       : shares executed at each step
        market_prices    : market mid prices at each step
        market_volumes   : market volume at each step (optional, uniform if None)
        """
        exec_prices = np.array(execution_prices)
        qtys = np.array(quantities)
        mkt_prices = np.array(market_prices)

        if market_volumes is not None:
            mkt_vols = np.array(market_volumes)
        else:
            mkt_vols = np.ones_like(mkt_prices)

        total_qty = qtys.sum()
        total_mkt_vol = mkt_vols.sum()

        if total_qty == 0 or total_mkt_vol == 0:
            return 0.0

        exec_vwap = float(np.sum(exec_prices * qtys) / total_qty)
        market_vwap = float(np.sum(mkt_prices * mkt_vols) / total_mkt_vol)

        return exec_vwap - market_vwap

    @staticmethod
    def sharpe_ratio(
        step_pnl: list[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Sharpe ratio of per-step P&L.

        Sharpe = (mean(pnl) - rf) / std(pnl)

        Useful for comparing the risk-adjusted performance of strategies
        across episodes.
        """
        pnl = np.array(step_pnl)
        if len(pnl) < 2:
            return 0.0
        std = float(np.std(pnl))
        if std < 1e-10:
            return 0.0
        return float((np.mean(pnl) - risk_free_rate) / std)

    @staticmethod
    def max_drawdown(cumulative_shortfall: list[float]) -> float:
        """
        Maximum drawdown of cumulative implementation shortfall.

        Measures the worst peak-to-trough deterioration in shortfall
        during execution. Higher drawdown = more risky/erratic execution.
        """
        cumulative = np.array(cumulative_shortfall)
        if len(cumulative) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return float(np.max(drawdown))

    @staticmethod
    def shortfall_volatility(episode_shortfalls: list[float]) -> float:
        """
        Standard deviation of implementation shortfall across episodes.
        Lower is better — indicates consistent execution.
        """
        return float(np.std(episode_shortfalls)) if len(episode_shortfalls) > 1 else 0.0

    @staticmethod
    def execution_summary(
        arrival_price: float,
        execution_prices: list[float],
        quantities: list[float],
        market_prices: list[float],
        remaining_shares: int,
        total_shares: int,
        episode_rewards: list[float] | None = None,
    ) -> dict:
        """
        Compute and return all metrics as a single summary dict.

        Useful for logging and benchmark comparison tables.
        """
        is_val = ExecutionMetrics.implementation_shortfall(
            arrival_price, execution_prices, quantities
        )
        avg_price = ExecutionMetrics.average_execution_price(
            execution_prices, quantities
        )
        vwap_slip = ExecutionMetrics.vwap_slippage(
            execution_prices, quantities, market_prices
        )
        fill = ExecutionMetrics.fill_rate(total_shares, remaining_shares)

        summary = {
            "implementation_shortfall": is_val,
            "average_execution_price": avg_price,
            "vwap_slippage": vwap_slip,
            "fill_rate": fill,
            "executed_shares": total_shares - remaining_shares,
            "remaining_shares": remaining_shares,
        }

        if episode_rewards:
            summary["sharpe_ratio"] = ExecutionMetrics.sharpe_ratio(episode_rewards)
            summary["shortfall_volatility"] = ExecutionMetrics.shortfall_volatility(
                [is_val]
            )

        return summary