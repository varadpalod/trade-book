# Project Map (explained like you're 16)

## 1. The one-paragraph version
Imagine you have to sell **1,000 cookies** at the school canteen, and you want the best total price.
If you dump all 1,000 at once, everyone sees a flood of cookies and the price drops.
If you sell one per minute, the price might wander off while you wait.
This project builds **a practice canteen (a simulator)** and **a robot student (an AI)** that learns, by trial and error, how to sell a big pile smartly. Then it checks whether the robot actually beats the simple, boring strategies.

In the real world the "cookies" are shares or crypto coins, and the people doing this are big investors (mutual funds, banks). This job is called **optimal trade execution**.

---

## 2. Words you need (just 8)

| Word | Meaning |
|---|---|
| **Bid** | The highest price a buyer will pay *right now*. If you sell instantly, you get this. |
| **Ask** | The lowest price a seller will accept. Buyers pay this. |
| **Spread** | Ask minus bid. The "toll" for being in a hurry. |
| **Market order** | "Sell NOW at whatever the buyers offer." Fast, but you pay the spread. |
| **Limit order** | "Sell only if I get at least THIS price." You wait in line; it might not happen. |
| **Market impact** | Selling a lot pushes the price down against you. Bigger and faster means worse. |
| **bps (basis points)** | A tiny unit of cost. 100 bps = 1%. So "+9 bps" means you lost 0.09% versus the starting price. |
| **TWAP** | The boring strategy: cut the pile into equal chunks and sell one chunk every tick. |

Two other boring strategies live in the project: **VWAP** (sell more when the market is busy) and **Almgren-Chriss** (a maths formula that balances "price drops if I rush" against "price might move while I wait").

---

## 3. The four parts of the project

```
  [ 1. DATA ]  ->  [ 2. SIMULATOR ]  ->  [ 3. THE ROBOT ]  ->  [ 4. SCOREBOARD ]
  fake/real        practice canteen       learns by trial        compares robot to
  price ticks      with real costs        and error              boring strategies
                                                                       |
                                                              [ DASHBOARD shows it live ]
```

1. **Data**: a long list of trades (price + size + time). The practice canteen replays these.
2. **Simulator** (`execution_env_v2.py`): a game. Each "turn" the robot decides what to do with the cookies it still has. The game applies the rules: sell at the bid, price impact, limit orders wait in line.
3. **The robot** (SAC, trained by `train_v2.py`): a small neural network. Each turn it sees 15 numbers (time left, cookies left, price move, how jumpy the market is, the spread...) and gives back **3 numbers**: how much to sell, market or limit, and how patient to be on price.
   - It starts out random, plays thousands of rounds, and gets a score. **Score = how much better it did than TWAP on the exact same prices.** Beat TWAP = points. Lose to TWAP = minus points.
4. **Scoreboard** (`eval_v2.py`, with `baselines_v2.py`): plays 300+ rounds on prices the robot has *never seen*, running the robot, TWAP, VWAP and Almgren-Chriss on identical prices, and prints one table. This is the honest test.

The **dashboard** (`frontend/` + `src/api/server.py`) is just a live TV for part 3: it shows what the robot decided, which inputs pushed that decision, and how it compares to TWAP and "dump everything".

---

## 4. What happened (the story so far)

- **First version:** the robot sold almost everything immediately with market orders. It looked like a genius.
  **Why it was cheating:** the simulator paid it the *ask* price (what buyers pay) instead of the *bid* (what sellers get), and had no price impact. In that broken world, dumping was free money. The robot just learned the loophole.
- **Fix (v2):** sell at the bid, add price impact, let limit orders wait, and score the robot against TWAP.
- **Result on generated data:** the robot no longer dumps, but was about equal to TWAP.
- **Result on real Binance data (current):** after training on real trades, the robot sells for **1.7 bps less than TWAP** and is cheaper on 71% of fresh games. When fills for resting limit orders are made much harder, the edge shrinks to about 0.4 bps, so it is real but modest. See `RESULTS.md`.
- **The original TD3 "win" was fake too.** The very first simulator had a units bug: the price volatility was meant per YEAR but applied per SECOND, so the simulated price collapsed about 99% in every game. In a market that always crashes, "sell as fast as you can" beats everything, so even a dumb rule beat TWAP and Almgren-Chriss 99% of the time. That bug (and a scoring bug that never charged for unsold shares) is now fixed in `execution_env.py`; old TD3 models were trained in the broken world and need retraining before you can trust them.
- **The scoreboard now includes all the boring strategies:** TWAP, VWAP and Almgren-Chriss, all replayed on the exact same prices with the same costs as the robot.
- **Caveat:** it is still a simulator (3 coins, one period), so this is not proof of live profit.

---

## 5. Where everything lives (after the cleanup)

```
trade book/
├── README.md                  short intro
├── sac_real.zip               the trained robot (trained on real Binance data)
├── sac_real_best/ checkpoints/   best-so-far and periodic saves from training
├── sac_v2.zip                 older robot trained on generated data
├── data/                      the price ticks (BTC, ETH, BNB), real Binance trades
├── data_generated/ raw/       generated test data and downloaded raw files
├── docs/
│   ├── PROJECT_MAP.md         <- this file
│   ├── RESULTS.md             the scoreboard tables and stress test
│   ├── MODEL_GUIDE.md         technical findings: what the first robot did and why
│   └── (older notes)
├── frontend/                  the live dashboard (React)
├── src/
│   ├── execution/             the boring strategies: TWAP, VWAP, Almgren-Chriss + cost maths
│   ├── market/                reading price data, downloading Binance, generating test data
│   ├── rl/
│   │   ├── execution_env_v2.py   THE GAME (rules + scoring)      <- most important file
│   │   ├── train_v2.py           teaches the robot
│   │   ├── eval_v2.py            the honest scoreboard
│   │   ├── baselines_v2.py       TWAP / VWAP / Almgren-Chriss run inside the v2 game
│   │   └── execution_env.py      the original simple environment (tests use it)
│   ├── api/
│   │   ├── server.py          runs the robot live and streams it to the dashboard
│   │   └── explain.py         works out "why did it do that?"
│   └── evaluation/            older comparison helpers
├── tests/                     automatic checks for the maths
└── archive/                   everything old, moved here (safe to delete later)
    ├── old_models/            13 TD3 models, older SAC models, checkpoints
    ├── old_code/              earlier experiments (phases 8-10), paper trading, risk module
    ├── old_data/              the first fake data, RELIANCE sample
    └── old_logs/              old training logs
```

**If you only learn three files:** `execution_env_v2.py` (the rules), `train_v2.py` (the learning), `eval_v2.py` (the truth).

---

## 6. How to run it

```powershell
cd "C:\Users\VARAD\OneDrive\Desktop\trade book"
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn websockets stable-baselines3 gymnasium pandas

python -m src.rl.eval_v2 --model sac_real.zip --episodes 300   # the scoreboard
python -m uvicorn src.api.server:app --port 8000               # live robot (terminal 1)
cd frontend; npm install; npm run dev                          # dashboard (terminal 2)
```
Retrain: `python -m src.rl.train_v2 --timesteps 400000 --save sac_real` (about 3.7 h at 30 fps)
Stress test: add `--limit-edge-bps 1` (and/or `--limit-vol-scale 0.5 --limit-ttl 2`) to the eval command.

---

## 7. What's left

1. Done: real Binance data, retrain, scoreboard, stress test (see `RESULTS.md`).
2. Optional: retrain with harsher limit fills switched on during training, to see if the edge grows back.
3. Optional: try more assets or a different time period.
4. Stale tests: 7 old test files (importing modules that no longer exist, or failing on old price-process code) live in `archive/old_tests/`. Move them back into `tests/` if you ever restore those modules.
