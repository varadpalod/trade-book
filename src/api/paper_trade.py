"""
paper_trade.py
--------------
Paper trading adapter that connects the SAC execution agent to a
live brokerage API for forward testing.

Currently supports:
    - Alpaca Markets (US equities + crypto)
    - CCXT (crypto exchanges: Binance, Coinbase, etc.)

This is a STUB implementation — the SAC agent logic is complete,
but the broker API calls are structured for easy replacement with
real API credentials.

Architecture:
    SAC agent -> PaperTrader.run_episode() -> BrokerAdapter.execute()
                                           -> RiskControls.check_order()

Usage:
    from src.api.paper_trade import PaperTrader, AlpacaAdapter

    adapter = AlpacaAdapter(api_key="...", api_secret="...", paper=True)
    trader = PaperTrader(model_path="sac_execution", adapter=adapter)
    trader.run_episode(symbol="AAPL", total_shares=1000, intervals=10)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.risk.risk_controls import RiskControls, RiskConfig, RiskViolationError
from src.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MarketQuote:
    """Live market quote."""
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: float
    ask_size: float
    timestamp: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class OrderResult:
    """Result of submitting an order."""
    order_id: str
    symbol: str
    qty: int
    filled_qty: int
    avg_fill_price: float
    status: str
    timestamp: float
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract broker adapter interface
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """Abstract interface for any broker API."""

    @abstractmethod
    def get_quote(self, symbol: str) -> MarketQuote:
        """Fetch the current best bid/ask for `symbol`."""
        ...

    @abstractmethod
    def submit_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        """Submit a market order. side = 'sell' or 'buy'."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> int:
        """Return current position (shares held) for `symbol`."""
        ...


# ---------------------------------------------------------------------------
# Alpaca adapter (paper trading)
# ---------------------------------------------------------------------------

class AlpacaAdapter(BrokerAdapter):
    """
    Alpaca Markets adapter for paper trading.

    Requires: pip install alpaca-trade-api
    Get API keys at: https://alpaca.markets (free paper account)

    Parameters
    ----------
    api_key    : Alpaca API key
    api_secret : Alpaca API secret
    paper      : if True, connects to paper trading endpoint
    """

    def __init__(
        self,
        api_key: str = "YOUR_ALPACA_API_KEY",
        api_secret: str = "YOUR_ALPACA_SECRET_KEY",
        paper: bool = True,
    ) -> None:
        self._key = api_key
        self._secret = api_secret
        self._paper = paper
        self._api = None  # Lazy import to avoid hard dependency

        log.info(
            "AlpacaAdapter created | paper=%s (keys not validated until first call)",
            paper,
        )

    def _connect(self) -> Any:
        """Lazy connection — imports alpaca-trade-api only when needed."""
        if self._api is not None:
            return self._api

        try:
            import alpaca_trade_api as tradeapi

            base_url = (
                "https://paper-api.alpaca.markets"
                if self._paper
                else "https://api.alpaca.markets"
            )
            self._api = tradeapi.REST(self._key, self._secret, base_url)
            log.info("Connected to Alpaca %s", "paper" if self._paper else "live")
        except ImportError:
            raise ImportError(
                "alpaca-trade-api not installed. Run: pip install alpaca-trade-api"
            )

        return self._api

    def get_quote(self, symbol: str) -> MarketQuote:
        api = self._connect()
        quote = api.get_last_quote(symbol)
        return MarketQuote(
            symbol=symbol,
            bid=float(quote.bidprice),
            ask=float(quote.askprice),
            last=float(quote.askprice),
            bid_size=float(quote.bidsize),
            ask_size=float(quote.asksize),
            timestamp=float(quote.timestamp.timestamp()),
        )

    def submit_market_order(self, symbol: str, qty: int, side: str = "sell") -> OrderResult:
        if qty <= 0:
            log.debug("Skipping zero-qty order")
            return OrderResult(
                order_id="SKIP", symbol=symbol, qty=0, filled_qty=0,
                avg_fill_price=0.0, status="skipped", timestamp=time.time()
            )

        api = self._connect()
        order = api.submit_order(
            symbol=symbol,
            qty=str(qty),
            side=side,
            type="market",
            time_in_force="day",
        )
        log.info("Alpaca order submitted | id=%s, qty=%d", order.id, qty)

        return OrderResult(
            order_id=order.id,
            symbol=symbol,
            qty=qty,
            filled_qty=int(order.filled_qty or 0),
            avg_fill_price=float(order.filled_avg_price or 0),
            status=order.status,
            timestamp=time.time(),
            raw={"order_id": order.id},
        )

    def get_position(self, symbol: str) -> int:
        api = self._connect()
        try:
            pos = api.get_position(symbol)
            return int(pos.qty)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# CCXT adapter (crypto exchanges)
# ---------------------------------------------------------------------------

class CCXTAdapter(BrokerAdapter):
    """
    CCXT adapter for crypto paper/live trading.

    Supports any exchange in the CCXT library (Binance, Coinbase, etc.)

    Requires: pip install ccxt

    Parameters
    ----------
    exchange_id : CCXT exchange ID (e.g., 'binance', 'coinbasepro')
    api_key     : exchange API key
    api_secret  : exchange API secret
    sandbox     : if True, use sandbox/paper endpoint
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "YOUR_API_KEY",
        api_secret: str = "YOUR_API_SECRET",
        sandbox: bool = True,
    ) -> None:
        self._exchange_id = exchange_id
        self._key = api_key
        self._secret = api_secret
        self._sandbox = sandbox
        self._exchange = None

        log.info("CCXTAdapter created | exchange=%s, sandbox=%s", exchange_id, sandbox)

    def _connect(self) -> Any:
        if self._exchange is not None:
            return self._exchange

        try:
            import ccxt
            exchange_class = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_class({
                "apiKey": self._key,
                "secret": self._secret,
            })
            if self._sandbox:
                self._exchange.set_sandbox_mode(True)
            log.info("Connected to %s (sandbox=%s)", self._exchange_id, self._sandbox)
        except ImportError:
            raise ImportError("ccxt not installed. Run: pip install ccxt")

        return self._exchange

    def get_quote(self, symbol: str) -> MarketQuote:
        ex = self._connect()
        ob = ex.fetch_order_book(symbol, limit=1)
        bid = ob["bids"][0][0] if ob["bids"] else 0.0
        ask = ob["asks"][0][0] if ob["asks"] else 0.0
        bid_size = ob["bids"][0][1] if ob["bids"] else 0.0
        ask_size = ob["asks"][0][1] if ob["asks"] else 0.0

        return MarketQuote(
            symbol=symbol, bid=bid, ask=ask, last=(bid + ask) / 2,
            bid_size=bid_size, ask_size=ask_size, timestamp=time.time(),
        )

    def submit_market_order(self, symbol: str, qty: int, side: str = "sell") -> OrderResult:
        if qty <= 0:
            return OrderResult(
                order_id="SKIP", symbol=symbol, qty=0, filled_qty=0,
                avg_fill_price=0.0, status="skipped", timestamp=time.time()
            )

        ex = self._connect()
        order = ex.create_market_order(symbol, side, qty)
        return OrderResult(
            order_id=order["id"],
            symbol=symbol,
            qty=qty,
            filled_qty=int(order.get("filled", 0)),
            avg_fill_price=float(order.get("average", 0)),
            status=order.get("status", "unknown"),
            timestamp=time.time(),
            raw=order,
        )

    def get_position(self, symbol: str) -> int:
        ex = self._connect()
        balance = ex.fetch_balance()
        base = symbol.split("/")[0]
        return int(balance.get(base, {}).get("free", 0))


# ---------------------------------------------------------------------------
# PaperTrader — wires SAC agent + risk controls + broker adapter
# ---------------------------------------------------------------------------

class PaperTrader:
    """
    Connects the trained SAC agent to a live broker via an adapter.

    Applies RiskControls before every order submission.

    Parameters
    ----------
    model_path : path to saved SAC model (without .zip)
    adapter    : broker adapter (AlpacaAdapter or CCXTAdapter)
    risk_cfg   : risk configuration (defaults to conservative limits)
    """

    def __init__(
        self,
        model_path: str = "sac_execution",
        adapter: BrokerAdapter | None = None,
        risk_cfg: RiskConfig | None = None,
    ) -> None:
        from stable_baselines3 import SAC

        self._model = SAC.load(model_path)
        self._adapter = adapter
        self._risk = RiskControls(risk_cfg or RiskConfig())

        log.info("PaperTrader initialised | model=%s", model_path)

    def run_episode(
        self,
        symbol: str,
        total_shares: int = 1_000,
        intervals: int = 10,
        interval_seconds: float = 60.0,
        dry_run: bool = True,
    ) -> list[OrderResult]:
        """
        Execute one liquidation episode.

        Parameters
        ----------
        symbol           : ticker (e.g., 'AAPL', 'BTC/USDT')
        total_shares     : total shares/units to sell
        intervals        : number of time steps
        interval_seconds : seconds between each step
        dry_run          : if True, log orders but don't submit to broker
        """
        from src.rl.execution_env import ExecutionEnv
        from src.config.env_config import EnvConfig

        cfg = EnvConfig(
            total_shares=total_shares,
            intervals=intervals,
            execution_horizon=intervals * interval_seconds,
        )
        env = ExecutionEnv(config=cfg)
        obs, _ = env.reset()

        results = []
        arrival_price = None

        log.info(
            "PaperTrader episode | symbol=%s, shares=%d, intervals=%d, dry_run=%s",
            symbol, total_shares, intervals, dry_run,
        )

        for step in range(intervals):
            # Get live quote
            if self._adapter and not dry_run:
                quote = self._adapter.get_quote(symbol)
                current_price = quote.mid
            else:
                current_price = env.current_price  # simulation fallback

            if arrival_price is None:
                arrival_price = current_price

            # SAC model decision
            action, _ = self._model.predict(obs, deterministic=True)
            proposed_qty = int(float(action[0]) * env.remaining_shares)

            # Risk check
            try:
                safe_qty = self._risk.check_order(
                    proposed_qty=proposed_qty,
                    current_price=current_price,
                    arrival_price=arrival_price,
                    shortfall_so_far=env.total_shortfall,
                )
            except RiskViolationError as e:
                log.error("Risk limit breached at step %d: %s — halting", step, e)
                break

            # Submit order
            if self._adapter and not dry_run:
                result = self._adapter.submit_market_order(symbol, safe_qty, "sell")
            else:
                result = OrderResult(
                    order_id=f"SIM-{step}",
                    symbol=symbol,
                    qty=safe_qty,
                    filled_qty=safe_qty,
                    avg_fill_price=current_price,
                    status="filled",
                    timestamp=time.time(),
                )

            results.append(result)
            log.info(
                "Step %d/%d | qty=%d, price=%.4f, status=%s",
                step + 1, intervals, safe_qty, current_price, result.status,
            )

            # Step the simulation environment forward
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

            if not dry_run:
                time.sleep(interval_seconds)

        log.info(
            "Episode complete | %d orders, violations=%d",
            len(results), self._risk.violation_count(),
        )
        return results
