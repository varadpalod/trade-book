import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config, Server
import numpy as np

from stable_baselines3 import SAC
from src.market.multi_asset_replay import MultiAssetReplay
from src.rl.execution_env_v2 import ExecutionEnvV2
from src.api.explain import FeatureInfluence, describe_decision, realistic_cost_bps

app = FastAPI(title="Trade Book Execution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE = "v2"       # realistic-cost env + retrained model (the only engine now)
TICK_DELAY = 0.5    # seconds between ticks, for a watchable UI
is_killed = False
clients = []
rl_model = None
multi_replay = None
influence = None

@app.get("/")
def read_root():
    return {"status": "Trade Book Engine is Running"}

@app.post("/api/kill")
def kill_switch():
    global is_killed
    is_killed = True
    return {"status": "EMERGENCY HALT INITIATED. ALL TRADING STOPPED."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)

async def broadcast(message: dict):
    for client in list(clients):
        try:
            await client.send_json(message)
        except:
            pass

def _make_env(seed=None):
    return ExecutionEnvV2(
        multi_replay=multi_replay,
        random_seed=np.random.randint(0, 100000) if seed is None else seed,
        split="eval",  # held-out period the model never trained on
    )

async def execution_engine_loop():
    """Background task streaming the actual trained SAC model execution."""
    global is_killed, rl_model, multi_replay, influence

    # 1. Load the MultiAsset Data Replay
    print("Loading Multi-Asset Replay Data...")
    multi_replay = MultiAssetReplay(data_dir="data/", symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"])

    # 2. Load the trained SAC Model
    print("Loading Trained SAC Model weights...")
    for path in ("sac_real.zip", "sac_v2.zip"):
        try:
            rl_model = SAC.load(path)
            print(f"Loaded {path}")
            break
        except Exception as e:
            print(f"Could not load {path}: {e}")
    if rl_model is None:
        return

    # 3. Calibrate the "why did it do that" explainer on the model's own rollouts
    print("Calibrating decision explainer...")
    influence = FeatureInfluence(rl_model, make_env=lambda: _make_env())

    while True:
        if is_killed:
            await broadcast({"type": "HALT", "message": "KILL SWITCH ACTIVATED. System halted."})
            await asyncio.sleep(5)
            continue

        # 4. Initialize Env (randomly selects an asset for this episode)
        env = _make_env()
        obs, _ = env.reset()

        baselines = {"twap": env.twap_bps_curve, "dump": [env.dump_bps] * env.intervals}

        await broadcast({
            "type": "INIT",
            "data": {
                "total_shares": env.total_shares,
                "arrival_price": env.arrival_price,
                "symbol": env.current_symbol,
                "intervals": env.intervals,
                "baselines": baselines,
                "engine": ENGINE,
            }
        })
        await asyncio.sleep(1.0) # Pause before starting episode

        done = False
        fills = []          # [(qty, price)] realistic fills (sell hits the bid)
        prev_filled = 0
        n_market = 0
        n_limit = 0
        n_hold = 0
        while not done:
            if is_killed:
                break

            snap = env._current_snapshot

            # --- REAL AI EXECUTION ---
            # Query the neural network for the optimal action
            action, _ = rl_model.predict(obs, deterministic=True)

            # Explain it BEFORE stepping (uses the exact observation the model saw)
            why = influence.explain(obs)
            free_inventory = env.remaining_shares - env._fill_sim.get_pending_qty()
            attempted = int(np.clip(action[0], 0.0, 1.0) * free_inventory)
            decision = describe_decision(action, attempted, free_inventory)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Work out what actually executed this tick (market + limit + final cleanup) ---
            prev_filled = env.filled_shares
            limit_filled = int(info.get("filled_qty", 0))
            tick_fills = list(info["fills"])           # exact fills, including impact
            market_filled = int(info["market_filled"])
            cleanup = int(info["cleanup"])
            fills.extend(tick_fills)

            if decision["kind"] == "market":
                n_market += 1
            elif decision["kind"] == "limit":
                n_limit += 1
            else:
                n_hold += 1

            tick_qty = sum(q for q, _ in tick_fills)
            exec_price = (sum(q * p for q, p in tick_fills) / tick_qty) if tick_qty else None
            agent_bps = realistic_cost_bps(env.arrival_price, fills)

            state = {
                "type": "TICK",
                "data": {
                    "step": env.current_step,
                    "symbol": env.current_symbol,
                    "mid_price": info.get("mid_price", snap.mid_price),
                    "bid": snap.bid_price,
                    "ask": snap.ask_price,
                    "spread": info.get("spread", snap.spread),
                    "remaining_shares": info.get("remaining_shares", 0),
                    "filled_shares": info.get("filled_shares", env.filled_shares),
                    "shortfall": info.get("shortfall", 0),
                    "fill_rate": info.get("fill_rate", 0),
                    "pending_qty": env._fill_sim.get_pending_qty(),
                    "order_event": {
                        "type": info.get("order_type") or "hold",
                        "price": info.get("limit_price") if info.get("order_type") == "limit" else "MARKET",
                        # includes market fills (previously always 0 for market orders)
                        "filled_this_tick": tick_qty,
                        "limit_filled": limit_filled,
                        "market_filled": market_filled,
                        "end_cleanup": cleanup,
                    },
                    "decision": decision,
                    "why": why,
                    "exec_price": exec_price,
                    "agent_cost_bps": agent_bps,
                    "env_reported_bps": info.get("shortfall", 0) / max(env.total_shares, 1) * 1e4,
                }
            }
            await broadcast(state)
            await asyncio.sleep(TICK_DELAY) # slower tick for visual UI

        if not is_killed:
            n_ticks = max(n_market + n_limit + n_hold, 1)
            summary = {
                "symbol": env.current_symbol,
                "agent_bps": realistic_cost_bps(env.arrival_price, fills),
                "twap_bps": baselines["twap"][-1],
                "dump_bps": baselines["dump"][-1],
                "market_ticks": n_market,
                "limit_ticks": n_limit,
                "hold_ticks": n_hold,
                "market_share": n_market / n_ticks,
            }
            await broadcast({
                "type": "DONE",
                "message": f"Episode Complete on {env.current_symbol}. Starting new episode...",
                "summary": summary,
            })
            await asyncio.sleep(3)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(execution_engine_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
