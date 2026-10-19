// Training Monitor — event-driven panel that answers "stop or keep going?" while PPO trains.
// Reads run/monitor.json (one row per rollout, written by snapshot.make_rollout_callback)
// and run/index.json (evals). Every verdict states the number it is based on and its threshold,
// so nothing here is a black box. No render loop: redrawn only when a new row arrives.
let C = { cyan: "#19e6ff", mag: "#ff2e88", green: "#00ffa3", gold: "#ffc53d", dim: "#5d6b8a", fg: "#d9e7ff", orange: "#ff8a1f", purple: "#b48cff" };
function themePalette() {        // chart colours follow the page theme (--c-* on body)
  const cs = getComputedStyle(document.body), g = (k, d) => (cs.getPropertyValue(k) || "").trim() || d;
  C = { cyan: g("--c-cyan", C.cyan), mag: g("--c-mag", C.mag), green: g("--c-green", C.green), gold: g("--c-gold", C.gold), dim: g("--c-dim", C.dim), fg: g("--c-fg", C.fg), orange: g("--c-orange", C.orange), purple: g("--c-purple", C.purple) };
}
const CSS = `
  #mon { position: fixed; inset: 0; overflow: auto; padding: 22px 24px 72px; font-family: var(--hud); color: var(--fg); }
  #mon .mtitle { max-width: 1700px; margin: 0 auto 12px; display: flex; align-items: baseline; gap: 16px; }
  #mon .mtitle h1 { margin: 0; font: 700 20px/1 var(--hud); letter-spacing: .32em; color: var(--cyan); text-shadow: 0 0 8px var(--cyan); }
  #mon .mtitle span { font-size: 12px; letter-spacing: .14em; color: var(--dim); }
  #mon .mgrid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; max-width: 1700px; margin: 0 auto; }
  #mon .mcard { background: rgba(6, 10, 24, .78); border: 1px solid rgba(25, 230, 255, .16); border-radius: 12px; padding: 12px 14px; min-width: 0; }
  #mon .mcard h3 { margin: 0 0 8px; font: 600 10px/1 var(--hud); letter-spacing: .26em; color: var(--dim); }
  #mon .mcard h3 b { color: var(--fg); font-weight: 600; letter-spacing: .08em; }
  #mon .s12 { grid-column: span 12; } #mon .s6 { grid-column: span 6; } #mon .s4 { grid-column: span 4; } #mon .s3 { grid-column: span 3; }
  #mon .chips { display: flex; flex-wrap: wrap; gap: 10px; }
  #mon .chip { flex: 1 1 260px; border-radius: 10px; padding: 10px 12px; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.03); }
  #mon .chip .t { font: 700 12px/1.2 var(--hud); letter-spacing: .14em; }
  #mon .chip .d { margin-top: 5px; font-size: 12px; color: var(--dim); line-height: 1.4; }
  #mon .chip.red { border-color: rgba(255,46,136,.6); background: rgba(255,46,136,.10); } #mon .chip.red .t { color: var(--mag); }
  #mon .chip.amber { border-color: rgba(255,197,61,.55); background: rgba(255,197,61,.09); } #mon .chip.amber .t { color: var(--gold); }
  #mon .chip.green { border-color: rgba(0,255,163,.5); background: rgba(0,255,163,.08); } #mon .chip.green .t { color: var(--green); }
  #mon .chip.grey .t { color: var(--dim); }
  #mon .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  #mon .kpi .k { font: 600 10px/1 var(--hud); letter-spacing: .22em; color: var(--dim); } #mon .kpi .v { font: 600 22px/1.25 var(--hud); font-variant-numeric: tabular-nums; }
  #mon .kpi .u { font-size: 11px; color: var(--dim); }
  #mon table { width: 100%; border-collapse: collapse; font-size: 13px; } #mon td, #mon th { padding: 5px 8px; border-bottom: 1px solid rgba(255,255,255,.06); text-align: right; }
  #mon th { color: var(--dim); font-weight: 600; letter-spacing: .08em; font-size: 10px; } #mon td:first-child, #mon th:first-child { text-align: left; }
  #mon svg { display: block; width: 100%; } #mon .lab { font-size: 11px; color: var(--dim); margin-top: 6px; }
  #mon .empty { color: var(--dim); font-size: 13px; padding: 30px 0; text-align: center; letter-spacing: .1em; }
  #mon .tl { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
  #mon .tl .p { background: rgba(255,255,255,.025); border-radius: 8px; padding: 6px 8px; } #mon .tl .p .k { font: 600 10px/1 var(--hud); letter-spacing: .2em; color: var(--dim); margin-bottom: 4px; }
  #mon .leg { font-size: 11px; color: var(--dim); margin-top: 6px; } #mon .leg i, #mon .lab i { display: inline-block; width: 14px; height: 3px; margin: 0 5px 2px 10px; }
  #mon canvas { display: block; width: 100%; }
  #mon .stsel { display: flex; align-items: center; gap: 10px; margin: 10px 0 6px; } #mon .stsel input { flex: 1; accent-color: var(--cyan); margin: 0; }
  #mon .stsel b { font-size: 12px; color: var(--fg); white-space: nowrap; font-weight: 600; }
  #mon .lab .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px -1px 10px; }
  /* Clear Liquid */
  body.clear #mon .mcard, body.clear #mon .tl .p { background: linear-gradient(140deg, rgba(255,255,255,.74), rgba(255,255,255,.44)); border: 1px solid rgba(255,255,255,.8);
    backdrop-filter: blur(22px) saturate(180%); box-shadow: inset 0 1px 0 rgba(255,255,255,.95), inset 0 -1px 0 rgba(15,26,43,.05), 0 14px 40px rgba(15,26,43,.12), 0 1px 2px rgba(15,26,43,.08); border-radius: 20px; }
  body.clear #mon .mcard::before { content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none; background: linear-gradient(115deg, rgba(255,255,255,.5) 0%, transparent 35%, transparent 70%, rgba(255,255,255,.3) 100%); }
  body.clear #mon .mcard { position: relative; overflow: hidden; } body.clear #mon .mcard > * { position: relative; }
  body.clear #mon .mcard h3 { color: var(--dim); } body.clear #mon .mcard h3 b { color: var(--fg); }
  body.clear #mon .mtitle h1 { color: #0f1a2b; text-shadow: none; letter-spacing: .22em; } body.clear #mon .mtitle span { color: var(--dim); }
  body.clear #mon .chip { background: rgba(255,255,255,.55); border: 1px solid rgba(255,255,255,.9); box-shadow: inset 0 1px 0 #fff, 0 4px 14px rgba(15,26,43,.08); border-radius: 14px; }
  body.clear #mon .chip .d { color: #3b4a62; }
  body.clear #mon .chip.green { background: linear-gradient(135deg, rgba(53,214,145,.35), rgba(255,255,255,.5)); } body.clear #mon .chip.green .t { color: #0f7a4a; }
  body.clear #mon .chip.red { background: linear-gradient(135deg, rgba(255,92,120,.38), rgba(255,255,255,.5)); } body.clear #mon .chip.red .t { color: #b3123c; }
  body.clear #mon .chip.amber { background: linear-gradient(135deg, rgba(255,196,70,.45), rgba(255,255,255,.5)); } body.clear #mon .chip.amber .t { color: #8a5a00; }
  body.clear #mon .chip.grey .t { color: var(--dim); }
  body.clear #mon .kpi .v { color: var(--fg); } body.clear #mon .kpi .k, body.clear #mon .kpi .u { color: var(--dim); }
  body.clear #mon td, body.clear #mon th { border-bottom-color: rgba(15,26,43,.08); color: var(--fg); } body.clear #mon th { color: var(--dim); }
  body.clear #mon .lab, body.clear #mon .empty, body.clear #mon .leg { color: var(--dim); }
  body.clear #mon .tl .p { border-radius: 12px; }
`;

const fmt = (v, d = 2, suf = "") => (v == null || !Number.isFinite(v)) ? "–" : (v > 0 && suf ? "+" : "") + v.toFixed(d) + suf;
const sum = a => a.reduce((x, y) => x + y, 0);
const mean = a => a.length ? sum(a) / a.length : null;

function pooled(rows) {
  // trade-weighted pools over a window of rollouts
  const gw = sum(rows.map(r => r.gross_win || 0)), gl = sum(rows.map(r => r.gross_loss || 0));
  const nTr = sum(rows.map(r => r.trades_closed || 0)), nOp = sum(rows.map(r => r.trades_opened || 0)), bars = sum(rows.map(r => r.bars || 0));
  const pnl = nTr ? sum(rows.map(r => (r.pnl_mean_bp || 0) * (r.trades_closed || 0))) / nTr : null;
  const hold = nTr ? sum(rows.map(r => (r.hold_mean || 0) * (r.trades_closed || 0))) / nTr : null;
  const exits = {}; rows.forEach(r => Object.entries(r.exits || {}).forEach(([k, v]) => exits[k] = (exits[k] || 0) + v));
  const g = rows.filter(r => r.gate);
  const sig = sum(g.map(r => r.signal_bars || 0)), acc = sum(g.map(r => r.accepted || 0));
  const w = (key, nkey) => { const n = sum(g.map(r => r[nkey] || 0)); return n ? sum(g.map(r => (r[key] || 0) * (r[nkey] || 0))) / n : null; };
  return { pf: gl > 0 ? gw / gl : (gw > 0 ? Infinity : null), nTr, nOp, bars, pnl, hold, exits,
           per100: bars ? 100 * nOp / bars : null, sig, acc, acceptance: sig ? acc / sig : null,
           cfAcc: w("cf_accepted_bp", "cf_accepted_n"), cfSkip: w("cf_skipped_bp", "cf_skipped_n"),
           nAcc: sum(g.map(r => r.cf_accepted_n || 0)), nSkip: sum(g.map(r => r.cf_skipped_n || 0)),
           blocked: sum(g.map(r => r.blocked_attempts || 0)), entropy: mean(rows.map(r => r.action_entropy).filter(Number.isFinite)) };
}

export function verdicts(rows, meta) {
  const out = [];
  if (!rows.length) return out;
  const last = rows[rows.length - 1], W = rows.slice(-20), P = pooled(W), nA = (meta.actions || []).length || 4;
  const progress = last.total_steps ? last.step / last.total_steps : 0;
  const chip = (cls, t, d) => out.push({ cls, t, d });
  // 1. does it use the signal / gate?
  if (P.sig > 0) {
    if (P.acceptance < 0.05) chip("amber", "TAKES ALMOST NOTHING", `accepted ${P.acc} of ${P.sig} signal bars (${(100 * P.acceptance).toFixed(0)}%) in the last ${W.length} rollouts · threshold < 5%`);
    else if (P.acceptance > 0.9) chip("amber", "NO FILTERING", `accepted ${(100 * P.acceptance).toFixed(0)}% of signal bars · the gate alone decides · threshold > 90%`);
    else chip("green", "FILTERS THE SIGNAL", `accepted ${P.acc} of ${P.sig} signal bars (${(100 * P.acceptance).toFixed(0)}%) · ${P.blocked} off-gate buy/sell attempts ignored by the gate`);
  } else if (meta.entry_gate) {
    chip("grey", "NO SIGNAL BARS YET", "the gate column had no non-zero bars in the last rollouts");
  }
  // 2. over-trading
  if (P.per100 != null && P.nTr >= 10) {
    if (P.per100 > 6 && P.hold != null && P.hold < 4) chip("red", "OVER-TRADING", `${P.per100.toFixed(1)} entries per 100 bars, mean hold ${P.hold.toFixed(1)} bars · threshold > 6 and < 4`);
    else if (P.per100 > 6) chip("amber", "TRADES OFTEN", `${P.per100.toFixed(1)} entries per 100 bars · threshold > 6`);
    else chip("green", "TRADE FREQUENCY OK", `${P.per100.toFixed(1)} entries per 100 bars, mean hold ${fmt(P.hold, 1)} bars`);
  }
  // 3. exits
  const nx = sum(Object.values(P.exits)); const mh = (P.exits.max_hold || 0) / Math.max(1, nx), sl = ((P.exits.sl || 0) + (P.exits.SL || 0) + (P.exits.managed_sl || 0)) / Math.max(1, nx);
  if (nx >= 10) {
    if (mh > 0.5) chip("amber", "EXITS BY TIMEOUT", `${(100 * mh).toFixed(0)}% of closes are max_hold (${meta.max_hold || "?"} bars) · the agent does not decide its exits · threshold > 50%`);
    else if (sl > 0.6) chip("amber", "EXITS BY STOP", `${(100 * sl).toFixed(0)}% of closes hit the stop · threshold > 60%`);
    else chip("green", "AGENT-TIMED EXITS", `${(100 * (1 - mh - sl)).toFixed(0)}% closed by the agent's own close action`);
  }
  // 4. selection skill (counterfactual, same fixed rule on accepted vs skipped signal bars)
  if (P.cfAcc != null && P.cfSkip != null && P.nAcc >= 30 && P.nSkip >= 30) {
    const d = P.cfAcc - P.cfSkip;
    if (d > 2) chip("green", "SELECTION HELPS", `fixed ${meta.cf_exit_bars || 4}-bar rule: accepted ${fmt(P.cfAcc, 1)} bp vs skipped ${fmt(P.cfSkip, 1)} bp (n ${P.nAcc}/${P.nSkip}) · threshold > +2 bp`);
    else if (d < -2) chip("red", "SELECTION HURTS", `fixed rule on accepted ${fmt(P.cfAcc, 1)} bp vs skipped ${fmt(P.cfSkip, 1)} bp (n ${P.nAcc}/${P.nSkip}) · the agent picks the worse signals`);
    else chip("grey", "SELECTION ≈ RANDOM", `accepted ${fmt(P.cfAcc, 1)} vs skipped ${fmt(P.cfSkip, 1)} bp · within ±2 bp`);
  }
  else if (P.sig > 0) chip("grey", "SELECTION: COLLECTING", `${P.nAcc} accepted / ${P.nSkip} skipped signal bars so far · verdict needs 30 of each`);
  // 5. exit timing: agent's realised pnl vs the fixed rule on the same accepted bars
  if (P.pnl != null && P.cfAcc != null && P.nTr >= 30) {
    const d = P.pnl - P.cfAcc;
    if (d > 2) chip("green", "EXIT TIMING ADDS", `agent ${fmt(P.pnl, 1)} bp per trade vs fixed-rule ${fmt(P.cfAcc, 1)} bp on the same entries · threshold > +2 bp`);
    else if (d < -2) chip("red", "EXIT TIMING LOSES", `agent ${fmt(P.pnl, 1)} bp per trade vs fixed-rule ${fmt(P.cfAcc, 1)} bp on the same entries`);
    else chip("grey", "EXIT TIMING ≈ RULE", `agent ${fmt(P.pnl, 1)} vs rule ${fmt(P.cfAcc, 1)} bp`);
  }
  // 6. exploration
  if (P.entropy != null) {
    const maxH = Math.log(nA);
    if (P.entropy < 0.25 && progress < 0.3) chip("amber", "EXPLORATION COLLAPSED EARLY", `sampled-action entropy ${P.entropy.toFixed(2)} nats at ${(100 * progress).toFixed(0)}% of training · threshold < 0.25 before 30%`);
    else if (P.entropy > 0.85 * maxH && progress > 0.7) chip("amber", "STILL NEAR-RANDOM", `entropy ${P.entropy.toFixed(2)} of max ${maxH.toFixed(2)} nats at ${(100 * progress).toFixed(0)}% of training`);
  }
  // 7. training-slice edge (exploring policy, so a lower bound)
  if (P.nTr >= 30 && P.pf != null) {
    if (P.pf > 1.2) chip("green", "TRAIN PF " + (Number.isFinite(P.pf) ? P.pf.toFixed(2) : "∞"), `${P.nTr} closed trades in the last ${W.length} rollouts, mean ${fmt(P.pnl, 1)} bp · exploring policy, in-sample`);
    else if (P.pf < 0.9) chip("red", "TRAIN PF " + P.pf.toFixed(2), `${P.nTr} closed trades · losing even in-sample`);
    else chip("grey", "TRAIN PF " + P.pf.toFixed(2), `${P.nTr} closed trades · flat`);
  }
  return out;
}

export function createMonitor(root) {
  if (!document.getElementById("mon-css")) { const st = document.createElement("style"); st.id = "mon-css"; st.textContent = CSS; document.head.appendChild(st); }
  root.innerHTML = `<div class="mtitle"><h1>TRAINING MONITOR</h1><span id="mSub">stop or keep going? — every verdict shows the number and threshold it rests on</span></div><div class="mgrid">
    <div class="mcard s12"><h3>VERDICTS · <b id="mHdr">waiting for the first rollout</b></h3><div class="chips" id="mChips"><div class="empty">no rollout yet</div></div></div>
    <div class="mcard s12"><h3>TRAIN vs VALIDATION · <b id="mGapHdr">greedy policy on two fixed slices of equal length, after every evaluation</b></h3>
      <div class="chips" id="mGapChip"><div class="empty">waiting for the first evaluation</div></div><div class="tl" id="mGap" style="margin-top:10px"></div><div class="lab" id="mGapLab"></div></div>
    <div class="mcard s12"><h3>DECISION STABILITY · <b id="mStHdr">the same validation bars at every evaluation</b></h3>
      <div class="kpis" id="mStKpi"></div>
      <div class="lab" id="mStHeatLab" style="margin:10px 0 4px">each row = one evaluation (oldest at the top), each column = one bar · colour = greedy action, brightness = its probability</div>
      <canvas id="mStHeat" height="120"></canvas>
      <div class="stsel"><span class="lab" style="margin:0">shade the price by the probabilities of</span><input type="range" id="mStSel" min="0" max="0" value="0"><b id="mStSelLab"></b></div>
      <canvas id="mStChart" height="190"></canvas><div class="lab" id="mStChartLab"></div></div>
    <div class="mcard s12"><h3>TRAINING DATA COVERAGE · <b id="mCovHdr">which Train bars the agent has played</b></h3>
      <div class="kpis" id="mCovKpi"></div><canvas id="mCov" height="96" style="margin-top:10px"></canvas><div class="lab" id="mCovLab"></div></div>
    <div class="mcard s12"><h3>LAST 20 ROLLOUTS · <b>pooled</b></h3><div class="kpis" id="mKpi"></div></div>
    <div class="mcard s6"><h3>COUNTERFACTUAL · <b>same bars, fixed rule vs agent</b></h3><div id="mCf"></div><div class="lab" id="mCfLab"></div></div>
    <div class="mcard s6"><h3>EXITS & ACTIONS · <b>last 20 rollouts</b></h3><div id="mExit"></div></div>
    <div class="mcard s4"><h3>ENTRIES PER 100 BARS</h3><svg id="mS1" height="90"></svg></div>
    <div class="mcard s4"><h3>MEAN HOLD (BARS)</h3><svg id="mS2" height="90"></svg></div>
    <div class="mcard s4"><h3>ACCEPTANCE %</h3><svg id="mS3" height="90"></svg></div>
    <div class="mcard s4"><h3>AGENT PNL PER TRADE (BP)</h3><svg id="mS4" height="90"></svg></div>
    <div class="mcard s4"><h3>SELECTION EDGE (ACCEPTED − SKIPPED, BP)</h3><svg id="mS5" height="90"></svg></div>
    <div class="mcard s4"><h3>ACTION ENTROPY (NATS)</h3><svg id="mS6" height="90"></svg></div>
    <div class="mcard s6"><h3>WHAT DRIVES THE DECISIONS · <b>share of |gradient × input| per feature, lags summed</b></h3><div id="mDrv"></div><div class="lab" id="mDrvLab"></div></div>
    <div class="mcard s6"><h3>WHICH BARS OF THE WINDOW MATTER · <b>share per lag (right = current bar)</b></h3><svg id="mLag" height="120"></svg><div class="lab">flat = the window length is not used; a spike at lag 0 = decisions rest on the latest bar</div></div>
    <div class="mcard s12"><h3>BEHAVIOUR OVER TRAINING · <b id="mTlHdr">per evaluation (validation, greedy) · this run bold, sibling seeds thin</b></h3><div class="tl" id="mTlEval"></div></div>
    <div class="mcard s12"><h3>TRAINING ROLLOUTS · <b id="mTrHdr">per rollout, 5-rollout mean · this run bold, sibling seeds thin</b></h3><div class="tl" id="mTlTrain"></div></div>
    <div class="mcard s12"><h3>EVALUATIONS · <b id="mEvHdr">validation slice, greedy policy</b></h3><div id="mEv"></div></div>
  </div>`;
  const $ = id => root.querySelector("#" + id);
  const width = el => Math.max(160, el.clientWidth || el.parentElement.clientWidth - 28);
  let last = null;

  function spark(el, xs, { color = C.cyan, zero = false, fmtv = v => fmt(v, 1) } = {}) {
    const w = width(el), h = 90, pad = 6, v = xs.filter(Number.isFinite);
    if (v.length < 2) { el.innerHTML = `<text x="8" y="50" fill="${C.dim}" font-size="12">not enough rollouts</text>`; return; }
    let lo = Math.min(...v), hi = Math.max(...v); if (zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); } if (hi === lo) hi = lo + 1;
    const X = i => pad + i * (w - 2 * pad) / (xs.length - 1), Y = y => h - pad - (y - lo) / (hi - lo) * (h - 2 * pad - 14);
    const pts = xs.map((y, i) => Number.isFinite(y) ? `${X(i).toFixed(1)},${Y(y).toFixed(1)}` : null).filter(Boolean).join(" ");
    const z = zero ? `<line x1="${pad}" x2="${w - pad}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.dim}" stroke-dasharray="3 3"/>` : "";
    const lastV = v[v.length - 1];
    el.setAttribute("viewBox", `0 0 ${w} ${h}`);
    el.innerHTML = z + `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.8"/>` +
      `<text x="${w - pad}" y="12" text-anchor="end" fill="${color}" font-size="13" font-weight="600">${fmtv(lastV)}</text>` +
      `<text x="${pad}" y="12" fill="${C.dim}" font-size="10">${xs.length} rollouts</text>`;
  }

  const PEER_COLORS = [C.cyan, C.gold, C.green, C.mag, C.purple, C.orange, "#7dff6b", "#9fb3d9"];
  function roll(xs, k = 5) { return xs.map((_, i) => { const w = xs.slice(Math.max(0, i - k + 1), i + 1).filter(Number.isFinite); return w.length ? sum(w) / w.length : NaN; }); }
  // small multiples: one panel per metric, one polyline per run (x = training step)
  function timelines(el, runs, metrics, { zero = {} } = {}) {
    const have = runs.filter(r => r.pts && r.pts.length);
    if (!have.length) { el.innerHTML = `<div class="empty">nothing yet</div>`; return; }
    const w = Math.max(220, Math.floor((el.clientWidth || 900) / Math.min(3, metrics.length)) - 24), h = 84, pad = 6;
    el.innerHTML = metrics.map(m => `<div class="p"><div class="k">${m.label}</div><svg id="tl_${m.key}" height="${h}"></svg></div>`).join("") +
      `<div class="leg" style="grid-column:1/-1">${have.map((r, i) => `<i style="background:${PEER_COLORS[i % 8]}"></i>${r.name}${i === 0 ? " (this run)" : ""}`).join("")}</div>`;
    for (const m of metrics) {
      const svg = el.querySelector("#tl_" + m.key);
      const all = have.map(r => r.pts.map(p => [p.step, m.get(p)]).filter(q => Number.isFinite(q[1])));
      const ys = all.flat().map(q => q[1]), xs = all.flat().map(q => q[0]);
      if (!ys.length) { svg.innerHTML = `<text x="8" y="48" fill="${C.dim}" font-size="12">–</text>`; continue; }
      let lo = Math.min(...ys), hi = Math.max(...ys); if (zero[m.key]) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); } if (hi === lo) hi = lo + 1;
      const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
      const X = x => pad + (x1 > x0 ? (x - x0) / (x1 - x0) : 0.5) * (w - 2 * pad), Y = y => h - pad - (y - lo) / (hi - lo) * (h - 2 * pad - 12);
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      let inner = zero[m.key] ? `<line x1="${pad}" x2="${w - pad}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.dim}" stroke-dasharray="3 3"/>` : "";
      all.forEach((pts, i) => {
        if (!pts.length) return;
        const d = pts.map(q => `${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(" ");
        inner += `<polyline points="${d}" fill="none" stroke="${PEER_COLORS[i % 8]}" stroke-width="${i === 0 ? 2.2 : 1}" opacity="${i === 0 ? 1 : 0.55}"/>`;
      });
      const lastThis = all[0].length ? all[0][all[0].length - 1][1] : null;
      inner += `<text x="${w - pad}" y="11" text-anchor="end" fill="${PEER_COLORS[0]}" font-size="12" font-weight="600">${lastThis == null ? "" : m.fmt(lastThis)}</text>`;
      svg.innerHTML = inner;
    }
  }

  const tile = (k, v, u = "", col = C.fg) => `<div class="kpi"><div class="k">${k}</div><div class="v" style="color:${col}">${v}</div><div class="u">${u}</div></div>`;

  // ---- train vs validation (greedy walks on two fixed slices, one pair per evaluation)
  const pfOf = s => s && s.gross_loss_bp > 0 ? s.gross_win_bp / s.gross_loss_bp : (s && s.gross_win_bp > 0 ? Infinity : NaN);
  const pfFmt = v => v === Infinity ? "∞" : Number.isFinite(v) ? v.toFixed(2) : "–";
  function pairLines(el, pts, metrics) {      // small multiples with 1–2 series per panel
    if (!pts.length) { el.innerHTML = ""; return; }
    const w = Math.max(220, Math.floor((el.clientWidth || 900) / Math.min(3, metrics.length)) - 24), h = 84, pad = 6;
    el.innerHTML = metrics.map(m => `<div class="p"><div class="k">${m.label}</div><svg id="gp_${m.key}" height="${h}"></svg></div>`).join("");
    for (const m of metrics) {
      const svg = el.querySelector("#gp_" + m.key);
      const series = m.series.map(s => pts.map(p => [p.step, s.get(p)]).filter(q => Number.isFinite(q[1])));
      const ys = series.flat().map(q => q[1]), xs = pts.map(p => p.step);
      if (!ys.length) { svg.innerHTML = `<text x="8" y="48" fill="${C.dim}" font-size="12">–</text>`; continue; }
      let lo = Math.min(...ys), hi = Math.max(...ys); if (m.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); } if (hi === lo) hi = lo + 1;
      const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
      const X = x => pad + (x1 > x0 ? (x - x0) / (x1 - x0) : 0.5) * (w - 2 * pad), Y = y => h - pad - (y - lo) / (hi - lo) * (h - 2 * pad - 12);
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      let inner = m.zero ? `<line x1="${pad}" x2="${w - pad}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.dim}" stroke-dasharray="3 3"/>` : "";
      series.forEach((s, k) => { if (s.length) inner += `<polyline points="${s.map(q => `${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${m.series[k].color}" stroke-width="${k === m.series.length - 1 ? 2.2 : 1.6}"/>`; });
      const lastTxt = m.series.map((s, k) => { const v = series[k].length ? series[k][series[k].length - 1][1] : null; return v == null ? "" : `<tspan fill="${s.color}">${m.fmt(v)}</tspan>`; }).filter(Boolean).join(`<tspan fill="${C.dim}"> · </tspan>`);
      inner += `<text x="${w - pad}" y="11" text-anchor="end" font-size="12" font-weight="600">${lastTxt}</text>`;
      svg.innerHTML = inner;
    }
  }
  function gapPanel(index) {
    const pts = index.filter(e => e.gap && e.gap.train && e.gap.val).slice().sort((a, b) => a.step - b.step);
    if (!pts.length) {
      $("mGapChip").innerHTML = `<div class="empty">${index.length ? "no train/validation walks in this run (trained before the gap panel existed, or no validation slice)" : "waiting for the first evaluation (10,000 steps)"}</div>`;
      $("mGap").innerHTML = ""; $("mGapLab").innerHTML = ""; return;
    }
    const last = pts[pts.length - 1], g = last.gap, v = g.verdict || { cls: "grey", t: "–", d: "" };
    $("mGapChip").innerHTML = `<div class="chip ${v.cls}"><div class="t">${v.t}</div><div class="d">${v.d}</div></div>` +
      `<div class="chip grey"><div class="t">EVAL ${last.i + 1} · STEP ${Number(last.step).toLocaleString()}</div><div class="d">` +
      `<span style="color:${C.gold}">train</span> ${fmt(g.train.return_pct, 2, "%")} · PF ${pfFmt(pfOf(g.train))} · ${g.train.trades} trades · win ${g.train.win_rate == null ? "–" : (100 * g.train.win_rate).toFixed(0) + "%"}<br>` +
      `<span style="color:${C.cyan}">validation</span> ${fmt(g.val.return_pct, 2, "%")} · PF ${pfFmt(pfOf(g.val))} · ${g.val.trades} trades · win ${g.val.win_rate == null ? "–" : (100 * g.val.win_rate).toFixed(0) + "%"}</div></div>`;
    $("mGapHdr").textContent = `greedy policy on two fixed ${g.frames}-bar slices · ${pts.length} evaluations`;
    const S = (get, color) => ({ get, color }), cap5 = x => Number.isFinite(x) ? Math.min(x, 5) : NaN;
    pairLines($("mGap"), pts, [
      { key: "ret", label: "SLICE RETURN %", zero: true, fmt: v => fmt(v, 2, "%"), series: [S(p => p.gap.train.return_pct, C.gold), S(p => p.gap.val.return_pct, C.cyan)] },
      { key: "gap", label: "GAP = TRAIN − VALIDATION (PP)", zero: true, fmt: v => fmt(v, 2), series: [S(p => p.gap.train.return_pct - p.gap.val.return_pct, C.orange)] },
      { key: "pf", label: "PROFIT FACTOR (CAPPED AT 5)", fmt: pfFmt, series: [S(p => cap5(pfOf(p.gap.train)), C.gold), S(p => cap5(pfOf(p.gap.val)), C.cyan)] },
      { key: "wr", label: "WIN RATE %", fmt: v => fmt(v, 0, "%"), series: [S(p => p.gap.train.win_rate != null ? 100 * p.gap.train.win_rate : NaN, C.gold), S(p => p.gap.val.win_rate != null ? 100 * p.gap.val.win_rate : NaN, C.cyan)] },
      { key: "tr", label: "TRADES ON THE SLICE", fmt: v => fmt(v, 0), series: [S(p => p.gap.train.trades, C.gold), S(p => p.gap.val.trades, C.cyan)] },
      { key: "rpb", label: "REWARD PER BAR", zero: true, fmt: v => fmt(v, 4), series: [S(p => p.gap.train.reward_per_bar, C.gold), S(p => p.gap.val.reward_per_bar, C.cyan)] },
    ]);
    $("mGapLab").innerHTML = `train slice = the last ${g.frames} bars of Train (in-sample) · validation = unseen · both walked with the argmax action, costs included <i style="background:${C.gold}"></i>train <i style="background:${C.cyan}"></i>validation · the verdict looks at the last 4 evaluations`;
  }

  // ---- decision stability (the same validation bars at every evaluation)
  let stSel = -1;
  const ACT_COL = name => ({ hold: C.dim, buy: C.green, sell: C.mag, close: C.gold })[name] || C.purple;
  const decodeP = e => { const u = Uint8Array.from(atob(e.p), c => c.charCodeAt(0)); return { u, na: e.n_actions || Math.max(1, Math.round(u.length / Math.max(1, e.bars))) }; };
  function stabilityPanel(st) {
    const evals = (st && st.evals) || [], bars = st && st.bars, acts = (st && st.actions) || [];
    const show = evals.length && bars && bars.c && bars.c.length;
    $("mStHeat").style.display = $("mStChart").style.display = show ? "" : "none"; $("mStSel").parentElement.style.display = show ? "" : "none";
    if (!show) { $("mStKpi").innerHTML = `<div class="empty">no stability data yet (stability.json is written after every evaluation; runs trained before this panel existed have none)</div>`; $("mStHeatLab").textContent = ""; $("mStChartLab").textContent = ""; return; }
    const last = evals[evals.length - 1], n = bars.c.length, th = st.settled_thresh || 0.1, settled = st.settled_since;
    $("mStKpi").innerHTML = [
      tile("CHANGED MIND", last.changed == null ? "–" : (100 * last.changed).toFixed(0) + "%", `of ${last.bars} greedy decisions vs the previous eval`,
           last.changed == null ? C.fg : last.changed > 0.3 ? C.mag : last.changed > th ? C.gold : C.green),
      tile("SETTLED SINCE", settled == null ? "not yet" : "eval " + (settled + 1), `every later eval changed ≤ ${Math.round(100 * th)}% of the decisions (needs 2 later evals)`, settled == null ? C.fg : C.green),
      tile("CONFIDENCE", last.confidence == null ? "–" : (100 * last.confidence).toFixed(0) + "%", "mean probability of the chosen action"),
      tile("ENTROPY", fmt(last.entropy, 2), `nats · ${Math.log(acts.length || 4).toFixed(2)} = uniform over ${acts.length || 4} actions`),
      tile("IN A POSITION", (100 * (last.in_position || 0)).toFixed(0) + "%", `of the bars · ${last.trades} trades · ${fmt(last.return_pct, 2, "%")}`),
    ].join("");
    const vs = st.val_slice || {};
    $("mStHdr").textContent = `${n} validation bars${vs.start ? " · " + String(vs.start).slice(0, 10) + " → " + String(vs.end).slice(0, 10) : ""} · ${evals.length} evaluations · greedy`;
    // heatmap: rows = evaluations, columns = bars
    const heat = $("mStHeat"), W = Math.max(300, heat.clientWidth || 900), rowH = Math.max(4, Math.min(14, Math.floor(240 / evals.length))), H = rowH * evals.length + 4;
    heat.width = W; heat.height = H; heat.style.height = H + "px";
    const g = heat.getContext("2d"); g.clearRect(0, 0, W, H);
    const x0 = 44, colW = (W - x0 - 6) / n;
    evals.forEach((e, r) => {
      const { u, na } = decodeP(e);
      g.globalAlpha = 1; g.fillStyle = C.dim; g.font = "10px sans-serif"; g.textAlign = "right"; g.fillText(String(e.i + 1), x0 - 6, r * rowH + rowH * 0.85);
      for (let b = 0; b < e.bars && b < n; b++) {
        const a = e.greedy[b]; g.fillStyle = ACT_COL(acts[a]); g.globalAlpha = 0.2 + 0.8 * (u[b * na + a] / 255);
        g.fillRect(x0 + b * colW, r * rowH, Math.max(1, colW), rowH - 1);
      }
    });
    g.globalAlpha = 1;
    $("mStHeatLab").innerHTML = `each row = one evaluation (oldest at the top), each column = one bar · colour = greedy action, brightness = its probability` +
      acts.map(a => `<span class="sw" style="background:${ACT_COL(a)}"></span>${a}`).join("");
    const sel = $("mStSel"); sel.max = String(evals.length - 1);
    sel.value = String(stSel >= 0 && stSel < evals.length ? stSel : evals.length - 1);
    sel.oninput = () => { stSel = +sel.value; drawShaded(evals[stSel], bars, acts); };
    drawShaded(evals[+sel.value], bars, acts);
    $("mStChartLab").innerHTML = `background <span class="sw" style="background:${C.green}"></span>P(buy) <span class="sw" style="background:${C.mag}"></span>P(sell) · line = close · bottom strip = position held (long / short) · drag the slider to see how the same chart was read at earlier evaluations`;
  }
  function drawShaded(e, bars, acts) {
    $("mStSelLab").textContent = `eval ${e.i + 1} · step ${Number(e.step).toLocaleString()} · changed ${e.changed == null ? "–" : (100 * e.changed).toFixed(0) + "%"}`;
    const cv = $("mStChart"), W = Math.max(300, cv.clientWidth || 900), H = 190; cv.width = W; cv.height = H; cv.style.height = H + "px";
    const g = cv.getContext("2d"); g.clearRect(0, 0, W, H);
    const n = bars.c.length, x0 = 44, cw = (W - x0 - 6) / n, top = 8, bot = H - 30, stripY = H - 22;
    const { u, na } = decodeP(e), iBuy = acts.indexOf("buy"), iSell = acts.indexOf("sell");
    for (let b = 0; b < n && b < e.bars; b++) {
      const pb = iBuy >= 0 ? u[b * na + iBuy] / 255 : 0, ps = iSell >= 0 ? u[b * na + iSell] / 255 : 0, x = x0 + b * cw, ww = Math.max(1, cw);
      if (pb > 0.02) { g.fillStyle = C.green; g.globalAlpha = 0.65 * pb; g.fillRect(x, top, ww, bot - top); }
      if (ps > 0.02) { g.fillStyle = C.mag; g.globalAlpha = 0.65 * ps; g.fillRect(x, top, ww, bot - top); }
      const p = e.pos[b]; if (p) { g.fillStyle = p > 0 ? C.green : C.mag; g.globalAlpha = 0.9; g.fillRect(x, stripY, ww, 6); }
    }
    g.globalAlpha = 1;
    const lo = Math.min(...bars.l), hi = Math.max(...bars.h), Y = v => bot - (v - lo) / (hi - lo || 1) * (bot - top);
    g.strokeStyle = C.fg; g.lineWidth = 1.4; g.beginPath();
    bars.c.forEach((c, b) => { const x = x0 + b * cw + cw / 2, y = Y(c); b ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    g.fillStyle = C.dim; g.font = "10px sans-serif"; g.textAlign = "right"; g.fillText(hi.toFixed(2), x0 - 4, top + 9); g.fillText(lo.toFixed(2), x0 - 4, bot);
    g.textAlign = "left"; g.fillText(String(bars.ts[0]).slice(0, 10), x0, H - 6); g.textAlign = "right"; g.fillText(String(bars.ts[n - 1]).slice(0, 10), W - 6, H - 6);
  }

  // ---- training data coverage (episodes.json: every training episode's start bar and length)
  function coveragePanel(ep) {
    const cv = $("mCov"), eps = (ep && ep.episodes) || [], N = ep ? ep.train_rows : 0;
    cv.style.display = eps.length && N ? "" : "none";
    if (!eps.length || !N) {
      $("mCovKpi").innerHTML = `<div class="empty">no episodes.json (written by rl_train after every rollout; runs trained before 2026-10-19 have none)</div>`;
      $("mCovLab").textContent = ""; return;
    }
    const diff = new Int32Array(N + 1);
    eps.forEach(e => { if (e.len > 0) { diff[Math.max(0, e.start)]++; diff[Math.min(N, e.start + e.len)]--; } });
    const cnt = new Int32Array(N); let run = 0, mx = 1;
    for (let i = 0; i < N; i++) { run += diff[i]; cnt[i] = run; if (run > mx) mx = run; }
    const s = ep.summary || {}, lo = ep.start_min || 0, pct = v => v == null ? "–" : (100 * v).toFixed(1) + "%";
    const open = eps[eps.length - 1].end === "open" ? eps[eps.length - 1] : null;
    $("mCovKpi").innerHTML = [
      tile("COVERED", pct(s.covered), `of ${(N - lo).toLocaleString()} playable Train bars, played at least once`, s.covered >= 0.95 ? C.green : s.covered >= 0.8 ? C.gold : C.mag),
      tile("NEVER PLAYED", (s.never ?? 0).toLocaleString(), "bars", s.never ? C.mag : C.green),
      tile("PLAYS PER BAR", s.passes == null ? "–" : s.passes.toFixed(2), `mean · max ${s.max_plays ?? "–"}`),
      tile("FIRST 10%", pct(s.first10), "oldest Train bars covered", s.first10 < 0.6 ? C.gold : C.fg),
      tile("LAST 10%", pct(s.last10), "newest Train bars covered (next to Validation)", s.last10 < 0.6 ? C.gold : C.fg),
      tile("EPISODES", String(s.episodes ?? eps.length), `${s.blowups || 0} ended by blow-up · ${ep.ep_len} bars max`, s.blowups ? C.gold : C.fg),
    ].join("");
    $("mCovHdr").textContent = `${eps.length} episodes · step ${Number(ep.step).toLocaleString()}${ep.total_steps ? " / " + Number(ep.total_steps).toLocaleString() : ""}${ep.final ? " · final" : " · live"}`;
    const W = Math.max(300, cv.clientWidth || 900), H = 96; cv.width = W; cv.height = H; cv.style.height = H + "px";
    const g = cv.getContext("2d"); g.clearRect(0, 0, W, H);
    const x0 = 8, x1 = W - 8, stripTop = 16, stripH = 46, X = i => x0 + i / N * (x1 - x0);
    g.fillStyle = C.dim; g.globalAlpha = 0.12; g.fillRect(x0, stripTop, x1 - x0, stripH); g.globalAlpha = 1;
    for (let px = 0; px < x1 - x0; px++) {                 // one pixel column = a bucket of bars: mean plays
      const a = Math.floor(px / (x1 - x0) * N), b = Math.max(a + 1, Math.floor((px + 1) / (x1 - x0) * N));
      if (b <= lo) continue;
      let sum = 0, zero = 0; for (let i = a; i < b; i++) { sum += cnt[i]; if (!cnt[i]) zero++; }
      const m = sum / (b - a);
      if (zero === b - a) { g.fillStyle = C.mag; g.globalAlpha = 0.55; }
      else { g.fillStyle = C.cyan; g.globalAlpha = 0.18 + 0.82 * Math.sqrt(m / mx); }
      g.fillRect(x0 + px, stripTop, 1, stripH);
    }
    g.globalAlpha = 1;
    eps.forEach(e => {                                     // a tick per episode start above the strip
      g.fillStyle = e.end === "blowup" ? C.mag : e.end === "open" ? C.gold : C.dim;
      g.fillRect(X(e.start), stripTop - (e.end === "open" ? 10 : 7), 1.5, e.end === "open" ? 9 : 6);
    });
    if (open) { g.strokeStyle = C.gold; g.lineWidth = 2; g.strokeRect(X(open.start), stripTop - 1, Math.max(2, X(open.start + open.len) - X(open.start)), stripH + 2); }
    g.fillStyle = C.dim; g.font = "10px sans-serif";
    (ep.ticks || []).forEach(([i, t], k, arr) => {
      g.textAlign = k === 0 ? "left" : k === arr.length - 1 ? "right" : "center";
      g.fillText(String(t).slice(0, 10), X(i), stripTop + stripH + 14);
    });
    $("mCovLab").innerHTML = `x = Train bars, oldest → newest · colour = how often the agent played that bar <span class="sw" style="background:${C.cyan}"></span>played (brighter = more) <span class="sw" style="background:${C.mag}"></span>never played` +
      ` · ticks = episode starts${s.blowups ? " (red = ended by blow-up)" : ""}${open ? ` · <span style="color:${C.gold}">gold box</span> = ${ep.final ? "the last episode, cut short when training ended" : "the episode in progress"}` : ""}` +
      ` · each episode starts at a random bar and plays up to ${ep.ep_len} bars, so the first and last bars of Train are played least · decision bars only (each observation also looks back ${ep.window - 1} bars)`;
  }

  function render(state) {
    last = state;
    themePalette();
    const { rows = [], meta = {}, index = [], peers = [], stability = null, episodes = null } = state;
    coveragePanel(episodes);
    gapPanel(index);
    stabilityPanel(stability);
    // --- timelines (drawn even before the first rollout: evals may exist)
    const runsEval = (peers.length ? peers : [{ name: meta.model || "this run", index, rows }]).map(r => ({ name: r.name, pts: (r.index || []).slice().sort((a, b) => a.step - b.step) }));
    timelines($("mTlEval"), runsEval, [
      { key: "rew", label: "EVAL REWARD", get: p => p.eval_reward, fmt: v => fmt(v, 2) },
      { key: "eqe", label: "EQUITY END %", get: p => p.equity_end != null ? (p.equity_end - 1) * 100 : NaN, fmt: v => fmt(v, 2, "%") },
      { key: "tr", label: "TRADES ON THE SLICE", get: p => p.trades, fmt: v => fmt(v, 0) },
      { key: "wr", label: "WIN RATE %", get: p => p.win_rate != null ? 100 * p.win_rate : NaN, fmt: v => fmt(v, 0, "%") },
      { key: "hold", label: "MEAN HOLD (BARS)", get: p => p.hold_mean, fmt: v => fmt(v, 1) },
      { key: "acc", label: "ACCEPTANCE %", get: p => p.acceptance != null ? 100 * p.acceptance : NaN, fmt: v => fmt(v, 0, "%") },
    ], { zero: { rew: true, eqe: true } });
    const runsTrain = (peers.length ? peers : [{ name: meta.model || "this run", index, rows }]).map(r => {
      const rr = (r.rows || []).slice().sort((a, b) => a.step - b.step);
      const sm = (key, f = x => x) => roll(rr.map(x => f(x[key])));
      const per100 = sm("trades_per_100"), hold = sm("hold_mean"), pnl = sm("pnl_mean_bp"), acc = sm("acceptance", v => v == null ? NaN : 100 * v), ent = sm("action_entropy");
      const sel = roll(rr.map(x => (x.cf_accepted_bp != null && x.cf_skipped_bp != null) ? x.cf_accepted_bp - x.cf_skipped_bp : NaN));
      return { name: r.name, pts: rr.map((x, i) => ({ step: x.step, per100: per100[i], hold: hold[i], pnl: pnl[i], acc: acc[i], ent: ent[i], sel: sel[i] })) };
    });
    timelines($("mTlTrain"), runsTrain, [
      { key: "per100", label: "ENTRIES / 100 BARS", get: p => p.per100, fmt: v => fmt(v, 1) },
      { key: "hold2", label: "MEAN HOLD (BARS)", get: p => p.hold, fmt: v => fmt(v, 1) },
      { key: "pnl", label: "PNL / TRADE (BP)", get: p => p.pnl, fmt: v => fmt(v, 1) },
      { key: "acc2", label: "ACCEPTANCE %", get: p => p.acc, fmt: v => fmt(v, 0, "%") },
      { key: "sel", label: "SELECTION EDGE (BP)", get: p => p.sel, fmt: v => fmt(v, 1) },
      { key: "ent", label: "ACTION ENTROPY", get: p => p.ent, fmt: v => fmt(v, 2) },
    ], { zero: { pnl: true, sel: true } });
    const lastRow = rows[rows.length - 1];
    if (!lastRow) {
      if (index.length) { $("mHdr").textContent = `${index.length} evaluations · no rollout rows (run trained before the Monitor existed)`; $("mChips").innerHTML = `<div class="empty">verdicts need monitor.json rows — retrain with the live visual on</div>`; renderEvals(index); }
      return;
    }
    const W = rows.slice(-20), P = pooled(W);
    $("mSub").textContent = `${meta.model || ""}${meta.symbol ? " · " + meta.symbol : ""}${meta.entry_gate ? " · gate " + meta.entry_gate : ""} · step ${Number(lastRow.step).toLocaleString()}${lastRow.total_steps ? " / " + Number(lastRow.total_steps).toLocaleString() : ""}`;
    $("mHdr").textContent = `rollout ${lastRow.seq + 1} · step ${Number(lastRow.step).toLocaleString()}${lastRow.total_steps ? " / " + Number(lastRow.total_steps).toLocaleString() : ""}${meta.entry_gate ? " · gate " + meta.entry_gate : " · no gate"}`;
    const vs = verdicts(rows, meta);
    $("mChips").innerHTML = vs.length ? vs.map(v => `<div class="chip ${v.cls}"><div class="t">${v.t}</div><div class="d">${v.d}</div></div>`).join("") : `<div class="empty">collecting…</div>`;
    $("mKpi").innerHTML = [
      tile("TRADES", P.nTr, `${P.nOp} opened · ${P.bars.toLocaleString()} bars`),
      tile("ENTRIES / 100 BARS", fmt(P.per100, 1), "", P.per100 > 6 ? C.mag : C.fg),
      tile("MEAN HOLD", fmt(P.hold, 1), "bars"),
      tile("PNL / TRADE", fmt(P.pnl, 1, " bp"), "after spread + commission", P.pnl > 0 ? C.green : C.mag),
      tile("PF (TRAIN)", P.pf == null ? "–" : Number.isFinite(P.pf) ? P.pf.toFixed(2) : "∞", "exploring policy", P.pf > 1 ? C.green : C.mag),
      meta.entry_gate ? tile("ACCEPTANCE", P.acceptance == null ? "–" : (100 * P.acceptance).toFixed(0) + "%", `${P.acc} of ${P.sig} signal bars`) : tile("ENTROPY", fmt(P.entropy, 2), "sampled actions"),
      tile("REWARD Σ", fmt(lastRow.reward_sum, 2), "last rollout"),
    ].join("");
    if (meta.entry_gate && P.sig) {
      $("mCf").innerHTML = `<table><tr><th>on the signal bars of the last ${W.length} rollouts</th><th>n</th><th>bp / trade</th></tr>
        <tr><td>fixed rule, EVERY signal (enter on the bar, exit after ${meta.cf_exit_bars || 4} bars, no stop)</td><td>${P.nAcc + P.nSkip}</td><td>${fmt(P.nAcc + P.nSkip ? ((P.cfAcc || 0) * P.nAcc + (P.cfSkip || 0) * P.nSkip) / (P.nAcc + P.nSkip) : null, 1)}</td></tr>
        <tr><td>fixed rule on the signals the agent ACCEPTED</td><td>${P.nAcc}</td><td>${fmt(P.cfAcc, 1)}</td></tr>
        <tr><td>fixed rule on the signals the agent SKIPPED</td><td>${P.nSkip}</td><td>${fmt(P.cfSkip, 1)}</td></tr>
        <tr><td><b>agent's own trades (its exits)</b></td><td>${P.nTr}</td><td><b>${fmt(P.pnl, 1)}</b></td></tr></table>`;
      $("mCfLab").textContent = "selection skill = accepted − skipped · exit-timing value = agent − accepted · bp = 0.01% of price, costs included";
    } else {
      $("mCf").innerHTML = `<div class="empty">no entry gate on this run — the counterfactual needs a signal column (rl_train --entry_gate)</div>`;
      $("mCfLab").textContent = "";
    }
    const nx = sum(Object.values(P.exits)) || 1;
    const bar = (label, parts) => `<div style="margin:6px 0"><div class="lab" style="margin:0 0 3px">${label}</div><div style="display:flex;height:14px;border-radius:4px;overflow:hidden;background:rgba(255,255,255,.05)">${parts.map(p => `<div title="${p.name} ${p.n}" style="width:${(100 * p.n / Math.max(1, p.tot)).toFixed(1)}%;background:${p.col}"></div>`).join("")}</div><div class="lab">${parts.map(p => `<span style="color:${p.col}">■</span> ${p.name} ${p.n}`).join(" · ")}</div></div>`;
    const exitCols = { signal: C.cyan, max_hold: C.gold, sl: C.mag, SL: C.mag, managed_sl: C.mag, episode_end: C.dim };
    const exits = Object.entries(P.exits).map(([k, n]) => ({ name: k, n, tot: nx, col: exitCols[k] || C.purple }));
    const acts = (meta.actions || []).map((a, i) => ({ name: a, n: sum(W.map(r => (r.action_mix || [])[i] || 0)), col: [C.dim, C.green, C.mag, C.gold, C.purple, C.cyan][i % 6] }));
    const tot = sum(acts.map(a => a.n)) || 1; acts.forEach(a => a.tot = tot);
    $("mExit").innerHTML = bar("how trades closed", exits) + bar("sampled actions (exploring)", acts);
    spark($("mS1"), rows.map(r => r.trades_per_100));
    spark($("mS2"), rows.map(r => r.hold_mean), { color: C.gold });
    spark($("mS3"), rows.map(r => r.gate ? 100 * (r.acceptance || 0) : NaN), { color: C.green, fmtv: v => fmt(v, 0, "%") });
    spark($("mS4"), rows.map(r => r.pnl_mean_bp), { color: C.green, zero: true });
    spark($("mS5"), rows.map(r => (r.cf_accepted_bp != null && r.cf_skipped_bp != null) ? r.cf_accepted_bp - r.cf_skipped_bp : NaN), { color: C.orange, zero: true });
    spark($("mS6"), rows.map(r => r.action_entropy), { color: C.purple, fmtv: v => fmt(v, 2) });
    renderEvals(index);
    drivers(W, meta);
  }
  function drivers(W, meta) {
    const el = $("mDrv"), lag = $("mLag");
    const withAttr = W.filter(r => r.attr_top && r.attr_top.length);
    if (!withAttr.length) { el.innerHTML = `<div class="empty">no attribution in these rows (run trained before 2026-10-18)</div>`; lag.innerHTML = ""; return; }
    const acc = new Map();
    withAttr.forEach(r => r.attr_top.forEach(([n, v]) => acc.set(n, (acc.get(n) || 0) + v / withAttr.length)));
    const top = [...acc.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12), mx = top[0][1] || 1;
    const lastTop = new Map(withAttr[withAttr.length - 1].attr_top);
    el.innerHTML = top.map(([n, v]) => `<div style="display:grid;grid-template-columns:1fr 1fr 52px;gap:8px;align-items:center;font-size:12px;margin:3px 0">
      <span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${n}">${n}</span>
      <span style="height:7px;border-radius:4px;background:linear-gradient(90deg,${C.orange},${C.gold});width:${(100 * v / mx).toFixed(0)}%"></span>
      <span style="text-align:right;color:${lastTop.has(n) ? C.fg : C.dim}">${(100 * v).toFixed(1)}%</span></div>`).join("");
    $("mDrvLab").textContent = `mean over the last ${withAttr.length} rollouts (top-10 of each) · grey % = not in the latest rollout's top-10 · the three state inputs (position, unrealized_pnl, bars_in_position) compete with the features`;
    const lags = withAttr[withAttr.length - 1].attr_lag || [];
    if (lags.length) {
      const w = width(lag), h = 120, pad = 8, n = lags.length, bw = (w - 2 * pad) / n, m2 = Math.max(...lags) || 1;
      lag.setAttribute("viewBox", `0 0 ${w} ${h}`);
      lag.innerHTML = lags.map((v, i) => `<rect x="${(pad + i * bw).toFixed(1)}" y="${(h - 18 - v / m2 * (h - 30)).toFixed(1)}" width="${(bw * 0.8).toFixed(1)}" height="${(v / m2 * (h - 30)).toFixed(1)}" fill="${i === n - 1 ? C.gold : C.cyan}" opacity=".85"/>` +
        `<text x="${(pad + i * bw + bw * 0.4).toFixed(1)}" y="${h - 5}" text-anchor="middle" fill="${C.dim}" font-size="10">-${n - 1 - i}</text>`).join("");
    }
  }
  function renderEvals(index) {
    if (index.length) {
      const tr = index.slice(-8).reverse().map(e => `<tr><td>eval ${e.i + 1}</td><td>${Number(e.step).toLocaleString()}</td><td>${fmt(e.eval_reward, 2)}</td><td>${e.trades ?? "–"}</td><td>${e.eq_curve ? fmt((e.eq_curve[e.eq_curve.length - 1] - 1) * 100, 2, "%") : "–"}</td><td>${e.max_dd != null ? fmt(e.max_dd * 100, 1, "%") : "–"}</td></tr>`).join("");
      $("mEv").innerHTML = `<table><tr><th>eval</th><th>step</th><th>reward</th><th>trades</th><th>equity</th><th>max dd</th></tr>${tr}</table>`;
      $("mEvHdr").textContent = `validation slice, greedy policy · ${index.length} evaluations`;
    } else $("mEv").innerHTML = `<div class="empty">first evaluation at 10,000 steps</div>`;
  }
  return { render, rerender: () => last && render(last) };
}
