# Results

One question: **can a reinforcement-learning agent sell a 1,000-unit order for less than the boring strategies?**

Setup: SAC agent (`sac_real.zip`), trained 400k steps on real Binance trades (BTC, ETH, BNB), scored on the held-out 30% of the data it never saw.
300 episodes, every strategy replayed on the identical ticks. Cost = implementation shortfall vs arrival price in bps (lower is better; sells hit the bid, with market impact).

## Headline (default fill rules)

| Strategy | Cost (bps) | Edge vs TWAP | Cheaper than TWAP |
|---|---|---|---|
| **SAC agent** | **+2.03** | **+1.70 ± 0.10** | **71%** |
| TWAP | +3.73 | baseline | |
| VWAP | +4.54 | -0.81 ± 0.04 | 7% |
| Almgren-Chriss (risk 1x) | +3.83 | -0.10 ± 0.02 | 36% |
| Almgren-Chriss (risk 10x) | +4.84 | -1.11 ± 0.07 | 14% |
| Dump everything | +20.72 | -17.00 ± 0.12 | 0% |

The agent is not dumping: it sells 36 of 1000 units on tick 1, with order mix of 35% market / 50% limit / 14% waiting.
What drives its decisions: clock and inventory 48%, live market signals 38%, which asset 14%.

## Stress test: harder limit-order fills

The agent was trained with fairly generous limit-order fills, so the edge was re-measured with harsher rules (same model, no retraining).

| Fill rule | SAC edge vs TWAP | Cheaper than TWAP |
|---|---|---|
| Default | +1.70 ± 0.10 bps | 71% |
| Half the fill volume | +1.50 ± 0.10 | 69% |
| Price must trade 1 bp through our limit | +0.83 ± 0.10 | 47% |
| Half volume + 1 bp through + faster expiry (ttl 2) | +0.37 ± 0.09 | 43% |

Reproduce with `--limit-vol-scale`, `--limit-edge-bps`, `--limit-ttl` on `eval_v2`.

## How to read this
- The edge is statistically real in every scenario, and it is **modest**: roughly 0.4 to 1.7 bps depending on how generous limit-order fills are.
- Volume of fills matters little. Whether price has to trade *through* our limit matters a lot, so most of the saving comes from resting limit orders filling at good prices.
- Almgren-Chriss ties TWAP and VWAP is slightly worse on this data. Beating TWAP is the real bar.
- Results are from a simulator on 3 assets and one data period. They do not prove live-trading profit.

## What this replaced
The original TD3 "beats TWAP and Almgren-Chriss comfortably" result came from simulator bugs (volatility in the wrong units, unsold shares never charged). See `PROJECT_MAP.md` section 4.

## Possible next step
Retrain with harsher fills switched on in training (~3.7 h at 30 fps) to see whether the agent recovers some of the lost edge.
