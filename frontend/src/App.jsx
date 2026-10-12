import { useState, useEffect, useRef, useMemo } from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend
} from 'recharts';
import './index.css';

const WS_URL = 'ws://localhost:8000/ws';
const API_URL = 'http://localhost:8000';

const fmtBps = (v) => (v === null || v === undefined || Number.isNaN(v) ? '---' : `${v >= 0 ? '+' : ''}${v.toFixed(1)} bps`);
const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);

/* ---------- small presentational pieces ---------- */

function Meter({ label, hint, value, min, max, threshold, valueText }) {
  const pct = ((value - min) / (max - min)) * 100;
  const thr = threshold !== undefined ? ((threshold - min) / (max - min)) * 100 : null;
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <strong>{valueText}</strong>
      </div>
      <div className="meter-track">
        {thr !== null && <div className="meter-threshold" style={{ left: `${thr}%` }} />}
        <div className="meter-fill" style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
      </div>
      <div className="meter-hint">{hint}</div>
    </div>
  );
}

function StackBar({ groups }) {
  const colors = { clock: 'var(--accent-red)', market: 'var(--accent-green)', asset: 'var(--accent-blue)' };
  return (
    <>
      <div className="stack-bar">
        {groups.map((g) => (
          <div key={g.key} style={{ width: `${g.influence * 100}%`, background: colors[g.key] }} title={`${g.label}: ${(g.influence * 100).toFixed(0)}%`} />
        ))}
      </div>
      <div className="stack-legend">
        {groups.map((g) => (
          <span key={g.key}>
            <i style={{ background: colors[g.key] }} />
            {g.label} <strong>{(g.influence * 100).toFixed(0)}%</strong>
          </span>
        ))}
      </div>
    </>
  );
}

function CompareRow({ label, value, best, tone }) {
  return (
    <div className={`compare-row ${best ? 'best' : ''}`}>
      <span><i className={`dot ${tone}`} />{label}</span>
      <strong>{fmtBps(value)}</strong>
    </div>
  );
}

const ChartTooltip = ({ active, payload, label, unit }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tooltip">
      <p style={{ color: 'var(--text-secondary)' }}>Tick {label}</p>
      {payload.filter((p) => p.value !== null && p.value !== undefined).map((p, i) => (
        <p key={i} style={{ color: p.color || p.stroke }}>
          {p.name}: {typeof p.value === 'number' ? (unit === '$' ? `$${p.value.toFixed(2)}` : unit === 'bps' ? fmtBps(p.value) : Math.round(p.value)) : p.value}
        </p>
      ))}
    </div>
  );
};

/* ---------- main app ---------- */

function App() {
  const [connected, setConnected] = useState(false);
  const [halted, setHalted] = useState(false);
  const [initData, setInitData] = useState(null);
  const [latest, setLatest] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [episodes, setEpisodes] = useState([]);

  const ws = useRef(null);
  const epClock = useRef([]); // per-episode "clock/inventory" influence samples
  const sawInit = useRef(false); // only score episodes we watched from tick 1

  const addLog = (text, type) => {
    setLogs((prev) => [{ text, type, time: new Date().toLocaleTimeString() }, ...prev].slice(0, 60));
  };

  useEffect(() => {
    let closed = false;
    let retry = null;

    const connect = () => {
      const sock = new WebSocket(WS_URL);
      ws.current = sock;

      sock.onopen = () => setConnected(true);
      sock.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 2000); // auto-reconnect
      };

      sock.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'INIT') {
          setInitData(msg.data);
          setLatest(null);
          setChartData([]);
          setHalted(false);
          epClock.current = [];
          sawInit.current = true;
          addLog(`New episode on ${msg.data.symbol}: sell ${msg.data.total_shares.toLocaleString()} units over ${msg.data.intervals} ticks.`, 'neutral');
        } else if (msg.type === 'TICK') {
          const d = msg.data;
          setLatest(d);
          const clock = d.why.groups.find((g) => g.key === 'clock');
          if (clock) epClock.current.push(clock.influence);

          setChartData((prev) => [...prev, {
            step: d.step,
            midPrice: d.mid_price,
            bid: d.bid,
            ask: d.ask,
            remaining: d.remaining_shares,
            execPrice: d.exec_price,
            execQty: d.order_event.filled_this_tick,
            agentBps: d.agent_cost_bps,
          }]);

          const ev = d.order_event;
          const dec = d.decision;
          let text;
          if (dec.kind === 'market') {
            text = `MARKET SELL: crossed the spread, sold ${ev.market_filled.toLocaleString()} (${(dec.qty_frac * 100).toFixed(0)}% of free inventory).`;
          } else if (dec.kind === 'limit') {
            text = `LIMIT SELL: posted ${dec.shares_attempted.toLocaleString()} at $${Number(ev.price).toFixed(2)}, filled ${ev.limit_filled.toLocaleString()}.`;
          } else {
            text = ev.limit_filled > 0 ? `HOLD: resting order filled ${ev.limit_filled.toLocaleString()}.` : 'HOLD: no order this tick.';
          }
          if (ev.end_cleanup > 0) text += ` Forced end-of-horizon liquidation: ${ev.end_cleanup.toLocaleString()} at the bid.`;
          addLog(`Tick ${d.step}: ${text}`, dec.kind === 'hold' ? 'neutral' : dec.kind);
        } else if (msg.type === 'HALT') {
          setHalted(true);
          addLog(msg.message, 'market');
        } else if (msg.type === 'DONE') {
          const clockAvg = avg(epClock.current);
          if (sawInit.current) setEpisodes((prev) => [...prev, { ...msg.summary, clock_share: clockAvg }].slice(-50));
          sawInit.current = false;
          const s = msg.summary;
          addLog(`Episode done: model ${fmtBps(s.agent_bps)} | TWAP ${fmtBps(s.twap_bps)} | Dump ${fmtBps(s.dump_bps)}.`, 'limit');
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      if (ws.current) ws.current.close();
    };
  }, []);

  const handleKillSwitch = async () => {
    try {
      const res = await fetch(`${API_URL}/api/kill`, { method: 'POST' });
      await res.json();
      setHalted(true);
      addLog('MANUAL OVERRIDE: Kill Switch Activated', 'market');
    } catch (e) {
      console.error(e);
    }
  };

  const totalShares = initData?.total_shares || 1000;
  const intervals = initData?.intervals || 10;
  const filled = latest?.filled_shares ?? 0;
  const fillRate = (filled / totalShares) * 100;
  const decision = latest?.decision;
  const symbol = latest?.symbol || initData?.symbol || 'Awaiting symbol...';

  // Inventory chart: agent vs a plain TWAP schedule
  const fullData = useMemo(
    () => Array.from({ length: intervals }, (_, i) => ({
      ...(chartData[i] ?? { step: i + 1 }),
      twapRemaining: totalShares * (1 - (i + 1) / intervals),
    })),
    [chartData, totalShares, intervals]
  );
  const inventoryData = fullData;
  const xTicks = Array.from({ length: intervals }, (_, i) => i + 1);

  // Cost chart: agent running cost vs the two dumb baselines, same ticks
  const costData = useMemo(() => {
    if (!initData?.baselines) return [];
    return Array.from({ length: intervals }, (_, i) => ({
      step: i + 1,
      agent: chartData[i]?.agentBps ?? null,
      twap: initData.baselines.twap[i],
      dump: initData.baselines.dump[i],
    }));
  }, [chartData, initData, intervals]);

  // Session scoreboard + auto verdict
  const board = useMemo(() => {
    const n = episodes.length;
    return {
      n,
      agent: avg(episodes.map((e) => e.agent_bps).filter((v) => v !== null)),
      twap: avg(episodes.map((e) => e.twap_bps)),
      dump: avg(episodes.map((e) => e.dump_bps)),
      market: avg(episodes.map((e) => e.market_share)),
      clock: avg(episodes.map((e) => e.clock_share).filter((v) => v !== null && v !== undefined)),
    };
  }, [episodes]);

  let verdict = { tone: 'neutral', text: `Collecting evidence... ${board.n}/10 episodes finished (one episode is too noisy to judge).` };
  if (board.n >= 10) {
    const gapDump = board.dump - board.agent; // >0 means the model is cheaper than dumping
    const gapTwap = board.twap - board.agent;
    if (Math.abs(gapDump) < 2 && board.market > 0.5) {
      verdict = { tone: 'negative', text: `Behaves like a sell-off: costs within ${Math.abs(gapDump).toFixed(1)} bps of "dump everything", and ${(board.market * 100).toFixed(0)}% of its ticks are market orders.` };
    } else if (gapDump > 2 && gapTwap > 2) {
      verdict = { tone: 'positive', text: `Beating both dumb rules by ${Math.min(gapDump, gapTwap).toFixed(1)}+ bps on average.` };
    } else {
      verdict = { tone: 'neutral', text: `No clear edge: ${fmtBps(-gapDump)} vs dump, ${fmtBps(-gapTwap)} vs TWAP (negative = model cheaper).` };
    }
  }

  const live = {
    agent: latest?.agent_cost_bps ?? null,
    twap: latest ? initData?.baselines?.twap[Math.min(latest.step, intervals) - 1] ?? null : null,
    dump: latest ? initData?.baselines?.dump[0] ?? null : null,
  };
  const liveVals = [live.agent, live.twap, live.dump].filter((v) => v !== null);
  const bestLive = liveVals.length ? Math.min(...liveVals) : null;
  const costSign = live.agent === null ? 'neutral' : live.agent > 0 ? 'negative' : 'positive';

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1>Mission Control</h1>
          <p className="subtitle">
            SAC execution agent selling {totalShares.toLocaleString()} units over {intervals} ticks on <strong>{symbol}</strong>
          </p>
        </div>

        <div className="header-right">
          <div className="status">
            <div
              className="status-dot"
              style={{
                background: connected ? 'var(--accent-green)' : 'var(--accent-red)',
                boxShadow: `0 0 10px ${connected ? 'var(--accent-green-glow)' : 'var(--accent-red-glow)'}`,
              }}
            />
            <span>{connected ? (halted ? 'System Halted' : 'Engine Online') : 'Connecting...'}</span>
          </div>
          <button className="kill-switch" onClick={handleKillSwitch} disabled={halted || !connected}>
            Kill Switch
          </button>
        </div>
      </header>

      {/* ---------- MODEL BRAIN ---------- */}
      <section className="card brain">
        <div className="brain-col">
          <div className="card-title">1 · What the model decided</div>
          <div className={`decision-headline ${decision?.kind || 'idle'}`}>
            {decision ? decision.headline : 'Waiting for first tick...'}
          </div>
          <p className="muted small">The network outputs 3 numbers every tick:</p>
          <Meter
            label="Size"
            value={decision?.qty_frac ?? 0}
            min={0}
            max={1}
            valueText={decision ? `${(decision.qty_frac * 100).toFixed(0)}% of free inventory` : '---'}
            hint="How much of what's left to try to sell now."
          />
          <Meter
            label="Order type bias"
            value={decision?.type_bias ?? 0}
            min={0}
            max={1}
            threshold={0.5}
            valueText={decision ? (decision.type_bias > 0.5 ? 'Market (> 0.5)' : 'Limit (< 0.5)') : '---'}
            hint="Above the white line = cross the spread immediately."
          />
          <Meter
            label="Limit price aggression"
            value={decision?.price_agg ?? 0}
            min={-1}
            max={1}
            threshold={0}
            valueText={decision ? decision.price_agg.toFixed(2) : '---'}
            hint="Only used for limit orders: -1 = least patient price, +1 = most patient (highest) price."
          />
        </div>

        <div className="brain-col">
          <div className="card-title">2 · Why: what moved that decision</div>
          {latest ? (
            <>
              <StackBar groups={latest.why.groups} />
              <p className="muted small" style={{ margin: '0.75rem 0 0.5rem' }}>
                Top inputs (share of the decision, measured by nudging each input by one typical swing):
              </p>
              {latest.why.features.slice(0, 6).map((f) => (
                <div className="bar-row" key={f.key}>
                  <span className="bar-label">{f.label}</span>
                  <div className="bar-track">
                    <div className={`bar-fill ${f.group}`} style={{ width: `${Math.min(100, f.influence * 100 * 3)}%` }} />
                  </div>
                  <span className="bar-val">{(f.influence * 100).toFixed(0)}%</span>
                </div>
              ))}
              <p className="muted small" style={{ marginTop: '0.75rem' }}>
                Price move, volatility and spread are inputs too; if they sit near 0%, the model is ignoring the market.
              </p>
            </>
          ) : (
            <p className="muted">Waiting for first tick...</p>
          )}
        </div>

        <div className="brain-col">
          <div className="card-title">3 · Is it better than a dumb rule?</div>
          <p className="muted small">Cost vs arrival price, same ticks, sells hit the bid. Lower is better.</p>
          <div className="compare">
            <CompareRow label="SAC model" value={live.agent} best={live.agent !== null && live.agent === bestLive} tone="blue" />
            <CompareRow label="TWAP (equal slices)" value={live.twap} best={live.twap !== null && live.twap === bestLive} tone="green" />
            <CompareRow label="Dump (all at tick 1)" value={live.dump} best={live.dump !== null && live.dump === bestLive} tone="red" />
          </div>

          <div className="card-title" style={{ marginTop: '1.25rem' }}>Session average ({board.n} episode{board.n === 1 ? '' : 's'})</div>
          <div className="compare">
            <CompareRow label="SAC model" value={board.agent} tone="blue" />
            <CompareRow label="TWAP" value={board.twap} tone="green" />
            <CompareRow label="Dump" value={board.dump} tone="red" />
          </div>
          <div className={`verdict ${verdict.tone}`}>{verdict.text}</div>
          {board.n > 0 && (
            <p className="muted small" style={{ marginTop: '0.5rem' }}>
              Market orders: {(board.market * 100).toFixed(0)}% of ticks · clock-driven: {board.clock !== null ? `${(board.clock * 100).toFixed(0)}%` : '---'} of decisions
            </p>
          )}
        </div>
      </section>

      <div className="grid-layout">
        {/* Left Column */}
        <div className="col">
          <div className="metric-grid">
            <div className="card">
              <div className="card-title">Inventory Sold</div>
              <div className="metric-value neutral">{fillRate.toFixed(1)}%</div>
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${fillRate}%` }} />
              </div>
              <p className="muted small" style={{ marginTop: '0.5rem' }}>
                {filled.toLocaleString()} / {totalShares.toLocaleString()} units executed
              </p>
            </div>

            <div className="card">
              <div className="card-title">Execution Cost vs Arrival</div>
              <div className={`metric-value ${costSign}`}>{fmtBps(live.agent)}</div>
              <p className="muted small" style={{ marginTop: '0.5rem' }}>
                Positive = sold below the arrival mid (a cost). Sells hit the bid.
              </p>
              {initData?.engine !== 'v2' ? (
                <p className="muted tiny">
                  Simulator's own number: {latest ? fmtBps(latest.env_reported_bps) : '---'} (it books market sells at the ask, so it flatters dumping)
                </p>
              ) : (
                <p className="muted tiny">Engine v2: realistic costs (sells at bid, market impact), held-out data.</p>
              )}
            </div>
          </div>

          <div className="card chart-card" style={{ height: '320px' }}>
            <div className="card-title">Price & Executions</div>
            <ResponsiveContainer width="100%" height="85%">
              <ComposedChart data={fullData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorMid" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent-blue)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="step" ticks={xTicks} padding={{ left: 12, right: 12 }} stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <YAxis domain={['auto', 'auto']} stroke="var(--text-secondary)" tick={{ fontSize: 12 }} tickFormatter={(v) => `$${Number(v).toFixed(v > 1000 ? 0 : 2)}`} width={70} />
                <Tooltip content={<ChartTooltip unit="$" />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area type="monotone" dataKey="midPrice" name="Mid" stroke="var(--accent-blue)" fillOpacity={1} fill="url(#colorMid)" isAnimationActive={false} />
                <Line type="monotone" dataKey="ask" name="Ask" stroke="var(--accent-red)" strokeWidth={1} dot={false} opacity={0.6} isAnimationActive={false} />
                <Line type="monotone" dataKey="bid" name="Bid" stroke="var(--accent-green)" strokeWidth={1} dot={false} opacity={0.6} isAnimationActive={false} />
                <Line
                  type="linear"
                  dataKey="execPrice"
                  name="Fill price"
                  stroke="none"
                  connectNulls={false}
                  isAnimationActive={false}
                  dot={(p) => {
                    if (p.payload?.execPrice === null || p.payload?.execPrice === undefined) return <g key={`d-${p.index}`} />;
                    const r = 4 + 10 * Math.sqrt((p.payload.execQty || 0) / totalShares);
                    return <circle key={`d-${p.index}`} cx={p.cx} cy={p.cy} r={r} fill="#F6E05E" fillOpacity={0.85} stroke="#000" strokeWidth={1} />;
                  }}
                  activeDot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
            <p className="muted tiny" style={{ textAlign: 'right' }}>Yellow dot size = units filled that tick.</p>
          </div>

          <div className="card chart-card" style={{ height: '260px' }}>
            <div className="card-title">Inventory: Model vs Equal-Slice (TWAP) Schedule</div>
            <ResponsiveContainer width="100%" height="85%">
              <ComposedChart data={inventoryData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="step" ticks={xTicks} padding={{ left: 12, right: 12 }} stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, totalShares]} stroke="var(--text-secondary)" tick={{ fontSize: 12 }} width={50} />
                <Tooltip content={<ChartTooltip unit="n" />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area type="stepAfter" dataKey="remaining" name="Model: remaining" stroke="var(--accent-green)" strokeWidth={2} fill="var(--accent-green)" fillOpacity={0.15} isAnimationActive={false} />
                <Line type="linear" dataKey="twapRemaining" name="TWAP: remaining" stroke="#A0AEC0" strokeDasharray="5 4" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="card chart-card" style={{ height: '260px' }}>
            <div className="card-title">Running Cost (bps): Model vs Dumb Rules</div>
            <ResponsiveContainer width="100%" height="85%">
              <ComposedChart data={costData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="step" ticks={xTicks} padding={{ left: 12, right: 12 }} stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 12 }} width={50} />
                <Tooltip content={<ChartTooltip unit="bps" />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="agent" name="SAC model" stroke="var(--accent-blue)" strokeWidth={2.5} dot={{ r: 3 }} connectNulls isAnimationActive={false} />
                <Line type="monotone" dataKey="twap" name="TWAP" stroke="var(--accent-green)" strokeWidth={1.5} dot={false} strokeDasharray="5 4" isAnimationActive={false} />
                <Line type="monotone" dataKey="dump" name="Dump" stroke="var(--accent-red)" strokeWidth={1.5} dot={false} strokeDasharray="2 3" isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Column */}
        <div className="col">
          <div className="card">
            <div className="card-title">Live Order Book</div>
            <div className="book-row">
              <div>
                <p className="muted tiny">Best Bid</p>
                <p className="metric-value positive" style={{ fontSize: '1.5rem' }}>${latest?.bid?.toFixed(2) || '---'}</p>
              </div>
              <div style={{ opacity: 0.6 }}>
                <p className="muted tiny">Mid</p>
                <p className="metric-value neutral" style={{ fontSize: '1.2rem' }}>${latest?.mid_price?.toFixed(2) || '---'}</p>
              </div>
              <div>
                <p className="muted tiny">Best Ask</p>
                <p className="metric-value negative" style={{ fontSize: '1.5rem' }}>${latest?.ask?.toFixed(2) || '---'}</p>
              </div>
            </div>
            <div className="book-foot">
              <span>Agent resting queue:</span>
              <strong style={{ color: 'var(--accent-blue)' }}>{latest?.pending_qty || 0} units</strong>
            </div>
          </div>

          <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="card-title">Live Execution Log</div>
            <div className="log-feed" style={{ flex: 1 }}>
              {logs.map((log, i) => (
                <div key={i} className={`log-entry ${log.type}`}>
                  <span className="log-time">[{log.time}]</span>
                  {log.text}
                </div>
              ))}
              {logs.length === 0 && (
                <p className="muted" style={{ textAlign: 'center', marginTop: '2rem' }}>Waiting for execution engine...</p>
              )}
            </div>
          </div>

          <details className="card explainer">
            <summary>How to read this dashboard</summary>
            <ul>
              <li><strong>The model</strong> is a Soft Actor-Critic neural network. Every tick it is given 15 numbers (clock, inventory, price, volatility, spread, order-flow, which asset) and returns 3 numbers (size, order type, limit price).</li>
              <li><strong>One tick</strong> is one trade in the replayed data, so the whole {intervals}-tick "day" is a few seconds of market time.</li>
              <li><strong>Why panel</strong> shows which inputs the 3 outputs are sensitive to right now. Mostly red = it is following the clock, not the market.</li>
              <li><strong>Cost</strong> is measured vs the arrival mid. TWAP and Dump are replayed on the exact same ticks so the comparison is fair.</li>
              <li><strong>Caveat:</strong> {initData?.engine === 'v2' ? 'engine v2 charges market impact and sells at the bid, but the data is still simulated; confirm on real Binance data.' : 'engine v1 has no market impact on large orders, so "dump" is unrealistically cheap.'}</li>
            </ul>
          </details>
        </div>
      </div>
    </div>
  );
}

export default App;
