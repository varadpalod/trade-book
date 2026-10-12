# Optimal Execution Engine

Sell a big order at the best total price: compare simple strategies (TWAP, VWAP, Almgren-Chriss) against a reinforcement-learning agent (SAC) in a simulator with realistic costs (sell at the bid, market impact, resting limit orders).

**Result:** on held-out real Binance data the SAC agent sells for **1.7 bps less than TWAP** (71% of episodes), shrinking to about 0.4 bps under much harsher limit-order fill assumptions. Almgren-Chriss and VWAP do not beat TWAP here. Full tables: [docs/RESULTS.md](docs/RESULTS.md).

- Start here: [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md) (plain-language tour)
- Results: [docs/RESULTS.md](docs/RESULTS.md)
- Technical findings: [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)

## Quick start
```
pip install -r requirements.txt
# data: real Binance trades (see data/README.md) or generated test data
python -m src.market.generate_microstructure_data
python -m src.rl.train_v2 --timesteps 400000 --save sac_real
python -m src.rl.eval_v2 --model sac_real.zip --episodes 300
# optional stress test: python -m src.rl.eval_v2 --model sac_real.zip --limit-edge-bps 1
python -m uvicorn src.api.server:app --port 8000      # optional live dashboard backend
cd frontend && npm install && npm run dev
```

## Stack
Python, PyTorch, Stable-Baselines3 (SAC), Gymnasium, NumPy, Pandas, FastAPI, React + Vite + Recharts.
