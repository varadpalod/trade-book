"""
execution_env_lo.py  (LO = Limit Order)
----------------------------------------
Phase 8 Gymnasium environment: Limit Order + Market Order execution.

The agent now controls THREE dimensions simultaneously:
    - HOW MUCH to sell     : fraction of remaining inventory [0, 1]
    - WHERE to post        : price aggressiveness [-1, 1]
    - HOW URGENT           : limit (passive) vs market (aggressive) [0, 1]

Action space (3D continuous):
    action[0] : quantity_fraction  [0.0, 1.0]   — fraction of remaining shares
    action[1] : price_aggression   [-1.0, 1.0]  — price level
                  -1.0 = passive deep bid (earn most spread, lowest fill prob)
                   0.0 = mid price
                  +1.0 = cross the spread (market order, guaranteed fill)
    action[2] : order_type_bias    [0.0, 1.0]   — if > 0.5, send as market order
                                                   regardless of price_aggression

Observation space (12D):
    0 : time_progress          current_step / intervals            [0, 1]
    1 : inventory_fraction     remaining / total                   [0, 1]
    2 : price_change           (P_t - P_0) / P_0                  ℝ
    3 : urgency                inventory_fraction / steps_left     [0, ∞)
    4 : rolling_vol            real tick vol                       [0, ∞)
    5 : spread_norm            spread / mid_price                  [0, ∞)
    6 : imbalance              (bid_vol - ask_vol) / total_vol     [-1, 1]
    7 : bid_vol_norm           bid_vol / (bid + ask)               [0, 1]
    8 : ask_vol_norm           ask_vol / (bid + ask)               [0, 1]
    9 : pending_qty_norm       pending_qty / total_shares          [0, 1]
    10: steps_since_last_fill  steps without a fill / intervals    [0, 1]
    11: fill_rate_so_far       filled_so_far / total_shares        [0, 1]
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.market.lob_replay import LOBReplay, LOBSnapshot
from src.execution.fill_simulator import FillSimulator
from src.config.env_config import EnvConfig, DEFAULT_CONFIG
from src.logger import get_logger

log = get_logger(__name__)


class ExecutionEnvLO(gym.Env):
    """
    Phase 8 Limit Order execution environment.

    Extends ExecutionEnvLOB with:
        - 3D action space (quantity, price level, order type)
        - FillSimulator for realistic limit order queue mechanics
        - 12D observation with fill tracking signals
        - Spread capture reward (agent earns when posting passive orders)

    Parameters
    ----------
    lob_replay        : pre-loaded LOBReplay instance (Binance tick data)
    config            : EnvConfig for shares/intervals/reward params
    random_seed       : seed for episode randomisation
    fill_model        : 'conservative' | 'aggressive' — fill simulator mode
    cancel_after_steps: steps before unfilled limit orders are cancelled
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lob_replay: LOBReplay,
        config: EnvConfig | None = None,
        random_seed: int = 42,
        fill_model: str = "conservative",
        cancel_after_steps: int = 1,
    ) -> None:
        super().__init__()

        self._replay = lob_replay
        self._cfg = config if config is not None else DEFAULT_CONFIG
        self._rng = np.random.default_rng(random_seed)

        self.total_shares = self._cfg.total_shares
        self.intervals = self._cfg.intervals
        self.execution_horizon = self._cfg.execution_horizon
        self._lambda_terminal = self._cfg.lambda_terminal

        # Fill simulator
        self._fill_sim = FillSimulator(
            fill_model=fill_model,
            cancel_after_steps=cancel_after_steps,
        )

        # Episode state
        self.remaining_shares: int = self.total_shares
        self.filled_shares: int = 0
        self.current_step: int = 0
        self.arrival_price: float = 0.0
        self.total_shortfall: float = 0.0
        self.total_spread_capture: float = 0.0
        self._steps_since_last_fill: int = 0
        self._episode_snapshots: list[LOBSnapshot] = []
        self._current_snapshot: LOBSnapshot | None = None

        # 3D action: [quantity_fraction, price_aggression, order_type_bias]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 12D observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        log.info(
            "ExecutionEnvLO (Phase 8) initialised | shares=%d, intervals=%d, "
            "ticks=%d, fill_model=%s",
            self.total_shares, self.intervals, len(self._replay), fill_model,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        start = self._replay.random_episode_start(
            episode_length=self.intervals, rng=self._rng
        )
        self._episode_snapshots = self._replay.slice(start, self.intervals + 2)
        if len(self._episode_snapshots) < self.intervals:
            self._episode_snapshots = self._replay.slice(0, self.intervals + 2)

        self.remaining_shares = self.total_shares
        self.filled_shares = 0
        self.current_step = 0
        self.total_shortfall = 0.0
        self.total_spread_capture = 0.0
        self._steps_since_last_fill = 0
        self._fill_sim.reset()

        self._current_snapshot = self._episode_snapshots[0]
        self.arrival_price = self._current_snapshot.mid_price

        log.debug("LO Env reset | arrival_price=%.4f", self.arrival_price)
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        qty_frac = float(np.clip(action[0], 0.0, 1.0))
        price_agg = float(np.clip(action[1], -1.0, 1.0))
        order_type = float(np.clip(action[2], 0.0, 1.0))

        snap = self._current_snapshot
        mid = snap.mid_price
        spread = snap.spread
        bid = snap.bid_price
        ask = snap.ask_price

        # Shares to attempt this step (from still-free inventory)
        free_inventory = self.remaining_shares - self._fill_sim.get_pending_qty()
        shares_to_attempt = int(qty_frac * free_inventory)
        shares_to_attempt = min(shares_to_attempt, free_inventory)

        reward = 0.0
        info = {}

        if shares_to_attempt > 0:
            # --- Determine order type ---
            use_market_order = order_type > 0.5

            if use_market_order:
                # Market order: guaranteed fill at cost of crossing the spread
                execution_price = ask  # We pay the ask when selling
                filled = shares_to_attempt
                self.remaining_shares -= filled
                self.filled_shares += filled
                self._steps_since_last_fill = 0

                shortfall = (
                    (self.arrival_price - execution_price) * filled
                    / max(self.arrival_price, 1e-8)
                )
                self.total_shortfall += shortfall
                reward -= shortfall

                log.debug(
                    "MARKET ORDER | step=%d, qty=%d, price=%.4f, shortfall=%.4f",
                    self.current_step, filled, execution_price, shortfall,
                )
                info["order_type"] = "market"

            else:
                # Limit order: post at a level derived from price_aggression
                # price_agg=-1 → deep bid (passive), price_agg=0 → mid, price_agg=+1 → ask (aggressive)
                # We interpolate between bid and ask
                if price_agg <= 0:
                    # Below mid: between bid and mid
                    limit_price = bid + (mid - bid) * (price_agg + 1.0)
                else:
                    # Above mid: between mid and ask
                    limit_price = mid + (ask - mid) * price_agg

                # Post to the fill simulator
                self._fill_sim.post_limit_order(
                    price=limit_price,
                    qty=shares_to_attempt,
                    side="sell",
                    current_step=self.current_step,
                    ask_volume=snap.ask_volume,
                )
                info["order_type"] = "limit"
                info["limit_price"] = limit_price

        # --- Process the fill simulator against this tick ---
        next_snap_idx = min(self.current_step + 1, len(self._episode_snapshots) - 1)
        next_snap = self._episode_snapshots[next_snap_idx]

        filled_qty, cancelled_qty = self._fill_sim.process_tick(
            current_step=self.current_step,
            mid_price=next_snap.mid_price,
            ask_price=next_snap.ask_price,
            bid_price=next_snap.bid_price,
            traded_volume=next_snap.ask_volume + next_snap.bid_volume,
            last_traded_price=next_snap.last_price,
        )

        if filled_qty > 0:
            self.remaining_shares -= filled_qty
            self.filled_shares += filled_qty
            self._steps_since_last_fill = 0

            # Spread capture: reward for posting passively and getting filled
            exec_price = info.get("limit_price", mid)
            shortfall = (
                (self.arrival_price - exec_price) * filled_qty
                / max(self.arrival_price, 1e-8)
            )
            self.total_shortfall += shortfall
            # Reward = negative shortfall BUT if exec_price > arrival_price, it's a gain
            reward -= shortfall

        if cancelled_qty > 0:
            # Penalty for cancelled unexecuted orders (they add urgency risk)
            reward -= 0.01 * cancelled_qty / max(self.total_shares, 1)

        # Urgency penalty: penalise holding too much inventory too late
        time_used = (self.current_step + 1) / self.intervals
        inventory_urgency = self._cfg.reward_alpha * (
            (self.remaining_shares / self.total_shares) ** 2
        ) * time_used
        reward -= inventory_urgency

        # Advance step
        self.current_step += 1
        self._steps_since_last_fill += 1

        if self.current_step < len(self._episode_snapshots):
            self._current_snapshot = self._episode_snapshots[self.current_step]

        terminated = (
            self.current_step >= self.intervals or self.remaining_shares == 0
        )
        truncated = False

        # Terminal: force-sell remaining inventory at market
        if terminated and self.remaining_shares > 0:
            remaining = self.remaining_shares
            # Cancel all pending limit orders
            self._fill_sim.cancel_all()
            # Force market sell at current bid (worst case)
            exec_price = self._current_snapshot.bid_price if self._current_snapshot else mid
            shortfall = (
                (self.arrival_price - exec_price) * remaining
                / max(self.arrival_price, 1e-8)
            )
            self.total_shortfall += shortfall
            reward -= self._lambda_terminal * abs(shortfall)
            self.remaining_shares = 0
            self.filled_shares += remaining

        info.update({
            "filled_qty": filled_qty,
            "cancelled_qty": cancelled_qty,
            "remaining_shares": self.remaining_shares,
            "filled_shares": self.filled_shares,
            "shortfall": self.total_shortfall,
            "mid_price": mid,
            "spread": spread,
            "fill_rate": self.filled_shares / self.total_shares,
        })

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        snap = self._current_snapshot
        if snap is None:
            return np.zeros(12, dtype=np.float32)

        price_change = (snap.mid_price - self.arrival_price) / max(self.arrival_price, 1e-8)
        steps_left = max((self.intervals - self.current_step) / self.intervals, 1e-6)
        urgency = (self.remaining_shares / self.total_shares) / steps_left
        spread_norm = snap.spread / max(snap.mid_price, 1e-8)
        total_vol = snap.bid_volume + snap.ask_volume + 1e-8
        bid_vol_norm = snap.bid_volume / total_vol
        ask_vol_norm = snap.ask_volume / total_vol
        pending_qty_norm = self._fill_sim.get_pending_qty() / max(self.total_shares, 1)
        steps_since_fill_norm = min(self._steps_since_last_fill / self.intervals, 1.0)
        fill_rate = self.filled_shares / max(self.total_shares, 1)

        return np.array([
            self.current_step / self.intervals,          # 0: time progress
            self.remaining_shares / self.total_shares,   # 1: inventory fraction
            price_change,                                # 2: price change
            urgency,                                     # 3: urgency
            snap.rolling_vol,                            # 4: rolling volatility
            spread_norm,                                 # 5: normalised spread
            snap.imbalance,                              # 6: order book imbalance
            bid_vol_norm,                                # 7: bid volume fraction
            ask_vol_norm,                                # 8: ask volume fraction
            pending_qty_norm,                            # 9: pending limit order qty
            steps_since_fill_norm,                       # 10: steps since last fill
            fill_rate,                                   # 11: cumulative fill rate
        ], dtype=np.float32)
