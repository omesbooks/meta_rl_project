// Training Monitor — event-driven panel that answers "stop or keep going?" while PPO trains.
// Reads run/monitor.json (one row per rollout, written by snapshot.make_rollout_callback)
// and run/index.json (evals). Every verdict states the number it is based on and its threshold,
// so nothing here is a black box. No render loop: redrawn only when a new row arrives.
const C = { cyan: "#19e6ff", mag: "#ff2e88", green: "#00ffa3", gold: "#ffc53d", dim: "#5d6b8a", fg: "#d9e7ff", orange: "#ff8a1f", purple: "#b48cff" };
const CSS = `
  #mon { position: fixed; inset: 0; overflow: auto; padding: 96px 24px 72px; font-family: var(--hud); color: var(--fg); }
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
  root.innerHTML = `<div class="mgrid">
    <div class="mcard s12"><h3>VERDICTS · <b id="mHdr">waiting for the first rollout</b></h3><div class="chips" id="mChips"><div class="empty">no rollout yet</div></div></div>
    <div class="mcard s12"><h3>LAST 20 ROLLOUTS · <b>pooled</b></h3><div class="kpis" id="mKpi"></div></div>
    <div class="mcard s6"><h3>COUNTERFACTUAL · <b>same bars, fixed rule vs agent</b></h3><div id="mCf"></div><div class="lab" id="mCfLab"></div></div>
    <div class="mcard s6"><h3>EXITS & ACTIONS · <b>last 20 rollouts</b></h3><div id="mExit"></div></div>
    <div class="mcard s4"><h3>ENTRIES PER 100 BARS</h3><svg id="mS1" height="90"></svg></div>
    <div class="mcard s4"><h3>MEAN HOLD (BARS)</h3><svg id="mS2" height="90"></svg></div>
    <div class="mcard s4"><h3>ACCEPTANCE %</h3><svg id="mS3" height="90"></svg></div>
    <div class="mcard s4"><h3>AGENT PNL PER TRADE (BP)</h3><svg id="mS4" height="90"></svg></div>
    <div class="mcard s4"><h3>SELECTION EDGE (ACCEPTED − SKIPPED, BP)</h3><svg id="mS5" height="90"></svg></div>
    <div class="mcard s4"><h3>ACTION ENTROPY (NATS)</h3><svg id="mS6" height="90"></svg></div>
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

  function render(state) {
    last = state;
    const { rows = [], meta = {}, index = [] } = state;
    const lastRow = rows[rows.length - 1];
    if (!lastRow) return;
    const W = rows.slice(-20), P = pooled(W);
    $("mHdr").textContent = `rollout ${lastRow.seq + 1} · step ${Number(lastRow.step).toLocaleString()}${lastRow.total_steps ? " / " + Number(lastRow.total_steps).toLocaleString() : ""}${meta.entry_gate ? " · gate " + meta.entry_gate : " · no gate"}`;
    const vs = verdicts(rows, meta);
    $("mChips").innerHTML = vs.length ? vs.map(v => `<div class="chip ${v.cls}"><div class="t">${v.t}</div><div class="d">${v.d}</div></div>`).join("") : `<div class="empty">collecting…</div>`;
    const tile = (k, v, u = "", col = C.fg) => `<div class="kpi"><div class="k">${k}</div><div class="v" style="color:${col}">${v}</div><div class="u">${u}</div></div>`;
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
    if (index.length) {
      const tr = index.slice(-8).reverse().map(e => `<tr><td>eval ${e.i + 1}</td><td>${Number(e.step).toLocaleString()}</td><td>${fmt(e.eval_reward, 2)}</td><td>${e.trades ?? "–"}</td><td>${e.eq_curve ? fmt((e.eq_curve[e.eq_curve.length - 1] - 1) * 100, 2, "%") : "–"}</td><td>${e.max_dd != null ? fmt(e.max_dd * 100, 1, "%") : "–"}</td></tr>`).join("");
      $("mEv").innerHTML = `<table><tr><th>eval</th><th>step</th><th>reward</th><th>trades</th><th>equity</th><th>max dd</th></tr>${tr}</table>`;
      $("mEvHdr").textContent = `validation slice, greedy policy · ${index.length} evaluations`;
    } else $("mEv").innerHTML = `<div class="empty">first evaluation at 10,000 steps</div>`;
  }
  return { render, rerender: () => last && render(last) };
}
