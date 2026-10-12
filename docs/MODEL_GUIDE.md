# What the model is actually doing

## Short version
It **is** a real model: a Soft Actor-Critic (SAC) neural network, queried every tick in `src/api/server.py` (`rl_model.predict`). But what it learned is effectively **"sell ~70% of whatever is left with market orders, front-loaded"**. That is a sell-off with a neural net on top, which is what you felt.

## The pipeline, per tick
1. **Input (15 numbers):** time elapsed, inventory left, price move vs arrival, urgency, volatility, spread, order-flow imbalance, buy/sell volume share, resting limit qty, ticks since fill, inventory sold, + 3 numbers identifying the asset (BTC/ETH/BNB).
2. **Output (3 numbers):** `qty_fraction` (how much of free inventory), `price_aggression` (limit price, -1 bid .. +1 ask), `order_type_bias` (> 0.5 = market order).
3. **Env** (`execution_env_multi.py`) turns that into a market sell or a limit order, fills it, pays reward.

## What I measured (200-300 episodes, `sac_multi_best`)
| Finding | Number |
|---|---|
| Ticks where it chose a market order | **97%** |
| Avg units sold on tick 1 (of 1000) | **~430** (569 left, then 364, 111, 43...) |
| Share of each decision from clock/inventory | **~64%** |
| ...from "which asset is this" | ~20% |
| ...from live market signals | ~16% (almost all order-flow imbalance) |
| Influence of price move / volatility / spread | **~0%** |
| Real cost, sells at bid: model / TWAP / dump | **+8.9 / +8.9 / +9.2 bps** (no edge) |
| What the sim reported for the model | **-9.5 bps (a "gain")** |

## Why it collapsed into a sell-off (root causes, all in the env)
1. **Market sells fill at the ask** (`execution_env_multi.py`, `execution_price = ask`). A seller gets the bid. This books the half-spread as profit, so dumping *looks* profitable (-9.5 bps vs +8.9 real). The old "Spread Capture" card showed this as outperformance.
2. **No market impact.** `market_impact.py` / Almgren-Chriss exist but the env never uses them, so selling 1000 units at once costs the same as selling 1.
3. **Limit orders can't win:** they are cancelled after 1 tick and fill only if price trades through; market orders always fill. Market strictly dominates.
4. **Nothing to react to:** only 1,000 ticks per asset, which look generated, and a 10-tick horizon. Price/volatility carry no learnable signal, so the net falls back on the clock.
5. **Reward** penalises remaining inventory every tick, pushing front-loading.

## UI fixes made
- `recharts` was imported but missing from `package.json`, so the app could not build. Added.
- Charts mixed `Line` inside `AreaChart`, so bid/ask/fill dots never rendered. Now `ComposedChart`.
- Market fills showed "0 shares" and no dot (server only reported limit fills). Fixed.
- "Spread Capture" card replaced with real cost in bps (sell at bid), plus the sim's number for contrast.
- New panels: what it decided, why (input influence), model vs TWAP vs dump on the same ticks, auto verdict.
- WebSocket auto-reconnect, fixed 1-10 tick axes, responsive layout.

## Retraining (done, v2) - result
New files: `src/rl/execution_env_v2.py`, `train_v2.py`, `eval_v2.py`, `src/market/generate_microstructure_data.py`. The old env is untouched.

Held-out scorecard (1,200 episodes, same ticks, `sac_v2.zip`, generated data with order-flow structure):
| | v1 model | v2 model |
|---|---|---|
| Market-order share of ticks | 97% | ~30% (plus limit + hold) |
| Sold on tick 1 | ~430 | ~170-260 |
| Decisions driven by live market signals | ~16% | ~50% |
| Edge vs TWAP (bps, + = better) | none (sim bug) | **+0.2 +/- 2.9, i.e. no demonstrable edge** |

So it stopped being a sell-off and now trades like a TWAP that reads the market, but it does not beat TWAP yet. The data is still simulated with weak structure; real Binance data is the real test.

## What was originally listed as not done
Sell at bid; add impact cost (`eta*q^2` temporary + permanent); scale order size to traded volume; hold limit orders >1 tick; use real/longer tick data; then retrain and re-run the dashboard. The verdict panel will tell you if it finally beats TWAP.

Run: `cd frontend && npm install`, then start the API (`python -m src.api.server`) and `npm run dev`.


## Why the original TD3/SAC "wins" over TWAP and Almgren-Chriss were not real
Found by running the repo's own benchmark logic with two deliberately dumb policies ("sell 50% of what's left each tick") on the ORIGINAL `ExecutionEnv`: they beat both TWAP and Almgren-Chriss on 99% of 300 paths.
1. **Volatility units bug.** `sigma` is an annualised vol (0.5-2.0) but was applied per sqrt(second) with 6-second ticks, so the simulated price fell ~93% in 3 ticks and ~99% per episode. Any front-loaded policy "won". Fixed in `src/rl/execution_env.py` (sigma / sqrt(trading seconds per year)); mean price change is now ~0.
2. **Unsold inventory was never charged.** At the horizon, `total_shortfall = terminal_shortfall` overwrote earlier costs (`=` not `+=`) and the `info["shortfall"]` the benchmark reads was captured before it. "Hold everything" scored 0.00 but honestly costs +0.55. Fixed (accumulate and publish).
After both fixes the dumb policies win only 42-46% of paths (i.e. they lose). Pre-existing: 11 unit tests already failed before these changes and fail identically after; no test changed status. TD3/SAC models trained in the broken environment must be retrained.

## v2 scoreboard now includes VWAP and Almgren-Chriss
`python -m src.rl.eval_v2` replays SAC, TWAP, VWAP, Almgren-Chriss (risk 1x and 10x) and dump-everything on identical episodes. On the generated held-out data (300 episodes) nothing beat TWAP by more than the noise (std err ~5 bps for the risk-aversion variants); only dump-everything was clearly worse. More episodes (1000+) or real data are needed to separate the strategies.
