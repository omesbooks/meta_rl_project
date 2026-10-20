// Backtest page (phase 3). Command, M1 auto-select, Skip-% preview and the report text
// all come from Python (webgui/backtest.py, backtest_report.py); this file only paints.
"use strict";
const BT = { cat: null, m1Auto: null, loaded: false, shownModel: "", runModel: "", running: false, shown: null, live: null };
const BT_FIELDS = ["conf", "start", "risk", "max_pos", "sl", "tp", "stop_slip", "swap_long", "swap_short", "random_baseline", "mc", "window"];

function btForm() {
  const f = { model: $("bt_model").value, source: $("bt_source").value, csv: $("bt_csv").value, m1_csv: $("bt_m1").value,
    mode: $("bt_mode").value, intrabar: $("bt_intrabar").value };
  for (const k of BT_FIELDS) f[k] = $("bt_" + k).value;
  return f;
}
function btMode() {
  const pure = $("bt_mode").value === "pure_agent";
  document.querySelectorAll("#p-backtest .sltp").forEach(e => e.style.display = pure ? "none" : "");
  $("bt_realism_hint").textContent = pure
    ? "Pure Agent: ไม่มี initial SL/TP → ATR SL/TP กับ Intrabar Fill ไม่ถูกใช้ · Stop Slippage ยังมีผลกับ model แบบ Manage 6 (agent ตั้ง SL เองผ่าน BE/Trailing)"
    : "ถ้า SL+TP อยู่ใน bar เดียว OHLC บอกไม่ได้ว่าโดนอะไรก่อน → รันทั้ง 2 โหมดเพื่อดูช่วงผลจริง · slippage คิดเฉพาะ SL (stop order)";
}
function ensureOption(sel, value, label) { if (![...sel.options].some(o => o.value === value)) sel.insertAdjacentHTML("beforeend", `<option value="${esc(value)}">${esc(label || value)}</option>`); }
async function btAutoM1() {
  const r = await api().backtest_m1($("bt_csv").value, $("bt_m1").value, BT.m1Auto);
  BT.m1Auto = r.auto;
  if (r.value) { ensureOption($("bt_m1"), r.value); $("bt_m1").value = r.value; }
  btM1Hint();
}
function btM1Hint() {
  $("bt_m1_hint").textContent = BT.m1Auto && $("bt_m1").value === BT.m1Auto ? `M1 replay: ${BT.m1Auto} (auto-detected companion file)` :
    $("bt_m1").value !== "(none)" ? "M1 replay: your choice is kept" : "no M1 file — bars where SL+TP both fit fall back to the intrabar assumption";
  $("bt_m1_hint").className = "hint " + ($("bt_m1").value !== "(none)" ? "ok" : "");
}
async function btRange() {
  const r = await api().backtest_range($("bt_csv").value, $("bt_start").value);
  $("bt_range").textContent = r.text; $("bt_range").className = "hint " + (r.level || "");
}
const btRangeSoon = debounce(btRange, 250);

function card(label, [text, tone]) { return `<div class="kpi ${tone === "dim" ? "" : tone}"><div class="k">${label}</div><div class="v">${esc(text)}</div></div>`; }
async function btReport(model) {
  BT.shownModel = model;
  $("bt_rep_name").textContent = model || "select a model";
  if (!model) return;
  const r = await api().backtest_report(model);
  if (BT.shownModel !== model) return;
  if (!r.ok) {
    $("bt_cards").innerHTML = ""; $("bt_equity").innerHTML = `<div class="empty">${esc(r.error)}</div>`; $("bt_mcimg").innerHTML = "";
    $("bt_report").innerHTML = ""; $("bt_report_txt").textContent = ""; $("bt_rep_src").textContent = ""; return;
  }
  BT.shown = r; btRepSrc();
  const c = r.cards;
  $("bt_cards").innerHTML = card("WIN RATE", c.win_rate) + card("PROFIT FACTOR", c.profit_factor) + card("RETURN", c.return) + card("MAX DD", c.max_dd);
  eqImgSlot(model).innerHTML = r.zero_trades ? `<div class="empty">0 trades — no equity chart</div>` : r.equity ? `<img src="${r.equity}" alt="equity curve">` : `<div class="empty">no equity image stored for this run</div>`;
  $("bt_mcimg").innerHTML = r.mc ? `<img src="${r.mc}" alt="Monte Carlo fan chart">` : "";
  $("bt_report").innerHTML = "<table>" + r.rows.map(([k, v]) => k === "" && v === "" ? `<tr class="sep"><td colspan="2"></td></tr>` :
    `<tr class="${v.startsWith("->") ? "verdict" : ""}"><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("") + "</table>";
  $("bt_report_txt").textContent = r.lines.join("\n");
}

// the RESULTS header names the stored run (date + data) so it is not mistaken for live output, and says so
// while a new backtest for the same model is running (the panel updates only when that run finishes)
function btRepSrc() {
  const r = BT.shown, el = $("bt_rep_src");
  if (!r) { el.textContent = ""; el.className = "r meta"; return; }
  const running = BT.running && BT.runModel === r.model;
  el.textContent = `stored result · ${r.created_at || "date unknown"}${r.data_csv ? " · " + r.data_csv : ""}` +
    (r.status && r.status !== "complete" ? ` · status: ${r.status}` : "") + (running ? " · previous run — a new backtest is running, the panel updates when it finishes" : "");
  el.className = "r meta" + (running ? " warn" : "");
}
// the equity area holds an optional live block (candles + markers + live equity, kept after the run for the
// model it was made for) and the stored PNG slot below it
function eqImgSlot(model) {
  const eq = $("bt_equity");
  let wrap = $("bt_live_wrap");
  if (wrap && wrap.dataset.model !== model) { wrap.remove(); wrap = null; }
  let slot = $("bt_equity_img");
  if (!slot) { eq.innerHTML = ""; if (wrap) eq.appendChild(wrap); slot = document.createElement("div"); slot.id = "bt_equity_img"; slot.className = "eq"; slot.style.width = "100%"; eq.appendChild(slot); }
  return slot;
}
// ---- live equity while backtest_live.py runs (BT_LIVE lines, ~150 per run; nothing is stored until the run ends)
const fmtPct = v => (v >= 0 ? "+" : "") + (100 * v).toFixed(2) + "%";
function liveStart(s) {
  BT.live = { initial: s.initial, bars: s.bars, pts: [], last: null, t0: "", t1: "", raf: 0,
              ohlc: null, idx: new Map(), opens: [], trades: [], cursor: -1, follow: true, view: 140 };
  $("bt_cards").innerHTML = card("WIN RATE", ["–", "dim"]) + card("PROFIT FACTOR", ["–", "dim"]) + card("RETURN", ["–", "dim"]) + card("MAX DD", ["–", "dim"]);
  $("bt_equity").innerHTML = `<div id="bt_live_wrap" data-model="${esc(BT.runModel)}" style="width:100%">
    <canvas id="bt_candles" width="900" height="260" style="width:100%;max-width:100%;border-radius:12px;display:block"></canvas>
    <div class="ctl" style="display:flex;align-items:center;gap:10px;margin:6px 0 8px;font-size:12px;color:var(--dim)">
      <span id="bt_cd_lab">loading bars…</span><input type="range" id="bt_cd_pos" min="0" max="0" value="0" style="flex:1;accent-color:var(--accent)">
      <button type="button" class="d" id="bt_cd_follow" style="padding:4px 10px;font-size:12px">follow</button></div>
    <canvas id="bt_live" width="900" height="220" style="width:100%;max-width:100%;border-radius:12px;display:block"></canvas></div>`;
  $("bt_cd_pos").oninput = e => { const L = BT.live; if (!L) return; L.follow = +e.target.value >= L.cursor; candlesDraw(+e.target.value); };
  const cdc = $("bt_candles");
  cdc.onmousemove = e => { const L = BT.live; if (!L || !L.ohlc) return; const r = cdc.getBoundingClientRect(), dpr = devicePixelRatio || 1, W = cdc.width;
    const x0 = 10 * dpr, x1 = W - 62 * dpr, end = Math.min(L.ohlc.ts.length - 1, Math.max(0, L.viewEnd ?? 0)), a = Math.max(0, end - L.view + 1);
    const i = a + Math.floor(((e.clientX - r.left) * dpr - x0) / ((x1 - x0) / (end - a + 1))); L.hoverC = i >= a && i <= end ? i : null; candlesDraw(); };
  cdc.onmouseleave = () => { const L = BT.live; if (L) { L.hoverC = null; candlesDraw(); } };
  const eqc = $("bt_live");
  eqc.onmousemove = e => { const L = BT.live; if (!L || L.pts.length < 2) return; const r = eqc.getBoundingClientRect(), dpr = devicePixelRatio || 1, W = eqc.width;
    const x0 = 64 * dpr, x1 = W - 16 * dpr, b = ((e.clientX - r.left) * dpr - x0) / (x1 - x0) * L.bars;
    let best = 0; for (let i = 1; i < L.pts.length; i++) if (Math.abs(L.pts[i][0] - b) < Math.abs(L.pts[best][0] - b)) best = i; L.hoverE = best; liveDraw(); };
  eqc.onmouseleave = () => { const L = BT.live; if (L) { L.hoverE = null; liveDraw(); } };
  $("bt_cd_follow").onclick = () => { const L = BT.live; if (!L) return; L.follow = true; $("bt_cd_pos").value = String(L.cursor); candlesDraw(); };
  $("bt_mcimg").innerHTML = ""; $("bt_report").innerHTML = `<div class="empty">live run · the full report (realism, baseline, Monte Carlo) appears when it finishes</div>`;
  $("bt_report_txt").textContent = "the text report is written when the run finishes";
  const L = BT.live;
  api().backtest_bars($("bt_csv").value, $("bt_start").value).then(b => {
    if (BT.live !== L) return;
    if (!b.ok) { $("bt_cd_lab").textContent = "no bars: " + b.error; return; }
    L.ohlc = b; b.ts.forEach((t, i) => L.idx.set(t, i));
    $("bt_cd_pos").max = String(b.ts.length - 1); candlesDraw();
  });
  const el = $("bt_rep_src"); el.textContent = `live · ${BT.runModel} · bar 0 / ${s.bars.toLocaleString()}`; el.className = "r meta ok";
}
function liveTrade(t) {
  const L = BT.live; if (!L) return;
  const i0 = L.idx.get(t.open_time), i1 = L.idx.get(t.exit_time);
  L.trades.push({ i0, i1, ...t }); L.opens = L.opens.filter(o => !(o.time === t.open_time && o.side === t.side));
  if (L.follow && !L.raf) L.raf = requestAnimationFrame(() => { L.raf = 0; liveDraw(); candlesDraw(); });
}
function liveOpen(o) { const L = BT.live; if (!L) return; L.opens.push({ i: L.idx.get(o.time), ...o }); }
// candles of the backtest range with the trades on them (MT visual-tester style): ▲ long entry, ▼ short entry,
// dotted line to the exit square (green = won, red = lost); an open position runs to the live cursor.
// time ticks for a span of timestamps: ~1 label per 110 px, format by the span (hours within a few days,
// day within a few months, month otherwise); returns [[index, label], ...] in the given index range
function timeTicks(tsAt, i0, i1, widthPx, dpr) {
  const n = Math.min(10, Math.floor(widthPx / (110 * dpr))), out = [];
  if (i1 <= i0 || n < 1) return i1 > i0 ? [[i0, tsAt(i0).slice(0, 16)]] : out;   // too narrow for two labels: just the start
  const t0 = Date.parse(tsAt(i0).replace(" ", "T")), t1 = Date.parse(tsAt(i1).replace(" ", "T")), days = (t1 - t0) / 864e5;
  const fmt = t => days <= 4 ? t.slice(5, 16) : days <= 400 ? t.slice(0, 10) : t.slice(0, 7);
  for (let k = 0; k <= n; k++) { const i = Math.round(i0 + (i1 - i0) * k / n); out.push([i, fmt(tsAt(i))]); }
  return out;
}
function labelBox(g, x, y, lines, dpr, fg, bg, W) {
  g.font = `${11 * dpr}px sans-serif`; g.textAlign = "left";
  const w = Math.max(...lines.map(l => g.measureText(l).width)) + 12 * dpr, h = lines.length * 14 * dpr + 8 * dpr;
  const bx = Math.min(Math.max(4 * dpr, x + 10 * dpr), W - w - 4 * dpr), by = Math.max(4 * dpr, y - h - 6 * dpr);
  g.fillStyle = bg; g.globalAlpha = 0.92; g.fillRect(bx, by, w, h); g.globalAlpha = 1;
  g.fillStyle = fg; lines.forEach((l, i) => g.fillText(l, bx + 6 * dpr, by + 15 * dpr + i * 14 * dpr));
}
function candlesDraw(endAt) {
  const L = BT.live, cv = $("bt_candles"); if (!L || !cv || !L.ohlc) return;
  const B = L.ohlc, n = B.ts.length;
  if (endAt != null) L.viewEnd = endAt; else if (L.follow || L.viewEnd == null) L.viewEnd = L.cursor >= 0 ? L.cursor : L.view - 1;
  const end = Math.min(n - 1, Math.max(0, L.viewEnd));
  const a = Math.max(0, end - L.view + 1), m = end - a + 1;
  const dpr = devicePixelRatio || 1, W = cv.width = Math.max(300, cv.clientWidth * dpr), H = cv.height = 260 * dpr;
  const g = cv.getContext("2d"), cs = getComputedStyle(document.body), col = k => cs.getPropertyValue(k).trim();
  const fg = col("--fg") || "#111", dim = col("--dim") || "#888", line = col("--line") || "#ccc", green = col("--green") || "#1c8a55", red = col("--red") || "#c24135", accent = col("--acc") || "#245b8f";
  g.clearRect(0, 0, W, H);
  const x0 = 10 * dpr, x1 = W - 62 * dpr, y0 = 16 * dpr, y1 = H - 24 * dpr, bw = (x1 - x0) / m;
  let lo = Infinity, hi = -Infinity;
  for (let i = a; i <= end; i++) { lo = Math.min(lo, B.l[i]); hi = Math.max(hi, B.h[i]); }
  const pad = (hi - lo || 1) * 0.06; lo -= pad; hi += pad;
  const X = i => x0 + (i - a + 0.5) * bw, Y = v => y1 - (v - lo) / (hi - lo) * (y1 - y0);
  g.strokeStyle = line; g.lineWidth = 1; [0.25, 0.5, 0.75].forEach(f => { const y = y0 + (y1 - y0) * f; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); });
  for (let i = a; i <= end; i++) {
    const up = B.c[i] >= B.o[i], c = up ? green : red; g.strokeStyle = g.fillStyle = c; g.lineWidth = Math.max(1, dpr);
    g.beginPath(); g.moveTo(X(i), Y(B.h[i])); g.lineTo(X(i), Y(B.l[i])); g.stroke();
    const t = Y(Math.max(B.o[i], B.c[i])), b = Y(Math.min(B.o[i], B.c[i])); g.fillRect(X(i) - bw * 0.3, t, Math.max(1, bw * 0.6), Math.max(1, b - t));
  }
  const tri = (x, y, upward, c) => { g.fillStyle = c; g.beginPath(); const s = 5 * dpr; if (upward) { g.moveTo(x, y); g.lineTo(x - s, y + s * 1.6); g.lineTo(x + s, y + s * 1.6); } else { g.moveTo(x, y); g.lineTo(x - s, y - s * 1.6); g.lineTo(x + s, y - s * 1.6); } g.closePath(); g.fill(); };
  const vis = i => i != null && i >= a && i <= end;
  for (const t of L.trades) {
    if (t.i0 == null || t.i1 == null || (t.i1 < a) || (t.i0 > end)) continue;
    const c = t.pnl >= 0 ? green : red;
    g.strokeStyle = c; g.setLineDash([3 * dpr, 3 * dpr]); g.lineWidth = Math.max(1, dpr); g.beginPath(); g.moveTo(X(Math.max(t.i0, a)), Y(t.entry)); g.lineTo(X(Math.min(t.i1, end)), Y(t.exit)); g.stroke(); g.setLineDash([]);
    if (vis(t.i0)) tri(X(t.i0), t.side === "long" ? Y(B.l[t.i0]) + 3 * dpr : Y(B.h[t.i0]) - 3 * dpr, t.side === "long", t.side === "long" ? green : red);
    if (vis(t.i1)) { g.fillStyle = c; g.fillRect(X(t.i1) - 3.5 * dpr, Y(t.exit) - 3.5 * dpr, 7 * dpr, 7 * dpr); }
  }
  for (const o of L.opens) {
    if (o.i == null || o.i > end) continue;
    if (vis(o.i)) tri(X(o.i), o.side === "long" ? Y(B.l[o.i]) + 3 * dpr : Y(B.h[o.i]) - 3 * dpr, o.side === "long", o.side === "long" ? green : red);
    g.strokeStyle = accent; g.setLineDash([2 * dpr, 4 * dpr]); g.beginPath(); g.moveTo(X(Math.max(o.i, a)), Y(o.price)); g.lineTo(X(end), Y(o.price)); g.stroke(); g.setLineDash([]);
  }
  if (L.cursor >= a && L.cursor <= end) { g.strokeStyle = accent; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(X(L.cursor), y0); g.lineTo(X(L.cursor), y1); g.stroke(); g.globalAlpha = 1; }
  g.fillStyle = dim; g.font = `${11 * dpr}px sans-serif`; g.textAlign = "left";
  g.fillText(hi.toFixed(hi > 100 ? 2 : 5), x1 + 4 * dpr, y0 + 8 * dpr); g.fillText(lo.toFixed(lo > 100 ? 2 : 5), x1 + 4 * dpr, y1);
  const ticks = timeTicks(i => B.ts[i], a, end, x1 - x0, dpr);
  ticks.forEach(([i, lab], k) => {
    g.strokeStyle = line; g.globalAlpha = 0.7; g.beginPath(); g.moveTo(X(i), y0); g.lineTo(X(i), y1 + 3 * dpr); g.stroke(); g.globalAlpha = 1;
    g.fillStyle = fg; g.textAlign = k === 0 ? "left" : k === ticks.length - 1 ? "right" : "center"; g.fillText(lab, X(i), H - 8 * dpr);
  });
  if (L.hoverC != null && L.hoverC >= a && L.hoverC <= end) {          // crosshair + the bar under the mouse
    const i = L.hoverC; g.strokeStyle = fg; g.globalAlpha = 0.35; g.setLineDash([3 * dpr, 3 * dpr]); g.beginPath(); g.moveTo(X(i), y0); g.lineTo(X(i), y1); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    const here = L.trades.filter(t => t.i0 === i || t.i1 === i).map(t => (t.i0 === i ? "entry " : "exit ") + t.side + (t.i1 === i ? ` ${t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(2)} (${t.reason})` : ""));
    labelBox(g, X(i), Y(B.h[i]), [B.ts[i].slice(0, 16), `O ${B.o[i]}  H ${B.h[i]}  L ${B.l[i]}  C ${B.c[i]}`, ...here], dpr, fg, (document.body.classList.contains("deep") ? "#0b1220" : "#ffffff"), W);
  }
  const nOpen = L.opens.length, won = L.trades.filter(t => t.pnl >= 0).length;
  $("bt_cd_lab").textContent = `${m} bars · ${L.trades.length} closed (${won} won) · ${nOpen} open${L.follow ? " · following" : ""}`;
  $("bt_cd_follow").style.display = L.follow ? "none" : "";
}
function livePoint(p) {
  const L = BT.live; if (!L) return;
  L.pts.push([p.bar, p.equity, p.time, p.trades]); L.last = p; if (!L.t0) L.t0 = p.time; L.t1 = p.time;
  const ci = L.idx.get(p.time); if (ci != null) { L.cursor = ci; if (L.follow) $("bt_cd_pos").value = String(ci); }
  const ret = p.balance / L.initial - 1, pf = p.gross_loss > 0 ? p.gross_win / p.gross_loss : (p.gross_win > 0 ? Infinity : NaN), wr = p.trades ? p.wins / p.trades : NaN;
  $("bt_cards").innerHTML =
    card("WIN RATE", [isNaN(wr) ? "–" : (100 * wr).toFixed(1) + "%", wr >= 0.5 ? "ok" : "dim"]) +
    card("PROFIT FACTOR", [isNaN(pf) ? "–" : pf === Infinity ? "∞" : pf.toFixed(2), pf >= 1 ? "ok" : "bad"]) +
    card("RETURN", [fmtPct(ret), ret >= 0 ? "ok" : "bad"]) + card("MAX DD", [fmtPct(p.max_dd), p.max_dd > -0.1 ? "ok" : "bad"]);
  $("bt_rep_src").textContent = `live · ${BT.runModel} · bar ${p.bar.toLocaleString()} / ${p.bars.toLocaleString()} · ${p.trades} trades · ${p.time}`;
  $("bt_status").textContent = `Backtesting ${BT.runModel}… ${Math.round(100 * p.bar / p.bars)}%`;
  if (!L.raf) L.raf = requestAnimationFrame(() => { L.raf = 0; liveDraw(); if (L.follow) candlesDraw(); });
}
function livePhase(ph) {
  const txt = ph.name === "baseline" ? `random-agent baseline ${ph.k} / ${ph.n}` : ph.name === "mc" ? `Monte Carlo · ${ph.n.toLocaleString()} shuffles` : ph.name;
  $("bt_status").textContent = `Backtesting ${BT.runModel}… ${txt}`;
  if (BT.live) $("bt_rep_src").textContent = `live · ${BT.runModel} · equity done · ${txt}`;
}
function liveDraw() {
  const L = BT.live, cv = $("bt_live"); if (!L || !cv) return;
  const W = cv.width = Math.max(300, cv.clientWidth * (devicePixelRatio || 1)), H = cv.height = 220 * (devicePixelRatio || 1);
  const g = cv.getContext("2d"), cs = getComputedStyle(document.body), col = k => cs.getPropertyValue(k).trim();
  const fg = col("--fg") || "#111", dim = col("--dim") || "#888", line = col("--line") || "#ccc", green = col("--green") || "#1c8a55", red = col("--red") || "#c24135", accent = col("--acc") || "#245b8f";
  g.clearRect(0, 0, W, H);
  const pts = L.pts; if (pts.length < 2) return;
  const x0 = 64 * (devicePixelRatio || 1), x1 = W - 16 * (devicePixelRatio || 1), y0 = 14 * (devicePixelRatio || 1), y1 = H - 26 * (devicePixelRatio || 1);
  let lo = Math.min(L.initial, ...pts.map(p => p[1])), hi = Math.max(L.initial, ...pts.map(p => p[1])); const pad = (hi - lo || 1) * 0.08; lo -= pad; hi += pad;
  const X = b => x0 + b / L.bars * (x1 - x0), Y = v => y1 - (v - lo) / (hi - lo) * (y1 - y0), yI = Y(L.initial);
  g.strokeStyle = line; g.lineWidth = 1; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(x0, yI); g.lineTo(x1, yI); g.stroke(); g.setLineDash([]);
  for (const [c, above] of [[green, true], [red, false]]) {   // shaded area above / below the starting balance
    g.beginPath(); g.moveTo(X(pts[0][0]), yI);
    for (const [b, v] of pts) g.lineTo(X(b), above ? Math.min(Y(v), yI) : Math.max(Y(v), yI));
    g.lineTo(X(pts[pts.length - 1][0]), yI); g.closePath(); g.fillStyle = c; g.globalAlpha = 0.18; g.fill(); g.globalAlpha = 1;
  }
  g.strokeStyle = accent; g.lineWidth = 1.8 * (devicePixelRatio || 1); g.beginPath();
  pts.forEach(([b, v], i) => i ? g.lineTo(X(b), Y(v)) : g.moveTo(X(b), Y(v))); g.stroke();
  const [lb, lv] = pts[pts.length - 1]; g.fillStyle = accent; g.beginPath(); g.arc(X(lb), Y(lv), 3.5 * (devicePixelRatio || 1), 0, Math.PI * 2); g.fill();
  g.fillStyle = dim; g.font = `${11 * (devicePixelRatio || 1)}px sans-serif`; g.textAlign = "right";
  g.fillText(hi.toFixed(0), x0 - 6, y0 + 10); g.fillText(lo.toFixed(0), x0 - 6, y1);
  if (Math.min(Math.abs(yI - y0 - 6), Math.abs(y1 - yI - 4)) > 14 * (devicePixelRatio || 1)) g.fillText(L.initial.toFixed(0), x0 - 6, yI + 4);   // skip when it would overlap hi/lo
  const dprE = devicePixelRatio || 1, ticksE = timeTicks(i => pts[i][2], 0, pts.length - 1, X(pts[pts.length - 1][0]) - X(pts[0][0]), dprE);
  ticksE.forEach(([i, lab], k) => {
    const x = X(pts[i][0]); g.strokeStyle = line; g.globalAlpha = 0.7; g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1 + 3 * dprE); g.stroke(); g.globalAlpha = 1;
    g.fillStyle = fg; g.textAlign = k === 0 ? "left" : k === ticksE.length - 1 ? "right" : "center"; g.fillText(lab, x, H - 8);
  });
  if (L.hoverE != null && L.hoverE < pts.length) {                    // crosshair + the point under the mouse
    const [b, v, t, ntr] = pts[L.hoverE], x = X(b); g.strokeStyle = fg; g.globalAlpha = 0.35; g.setLineDash([3 * dprE, 3 * dprE]); g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    g.fillStyle = accent; g.beginPath(); g.arc(x, Y(v), 3 * dprE, 0, Math.PI * 2); g.fill();
    labelBox(g, x, Y(v), [String(t).slice(0, 16), `equity ${v.toFixed(2)} (${fmtPct(v / L.initial - 1)}) · bar ${b.toLocaleString()} · ${ntr} trades`], dprE, fg, (document.body.classList.contains("deep") ? "#0b1220" : "#ffffff"), W);
  }
  g.textAlign = "center"; g.fillStyle = dim; g.fillText(`equity · live · ${lb.toLocaleString()} / ${L.bars.toLocaleString()} bars`, (x0 + x1) / 2, y0 + 10);
}
function onBacktestEvent(ev) {
  if (ev.type === "start") {
    setBusy(true, ev.kind === "chart" ? "generating chart" : "backtesting");
    BT.runModel = ev.model; BT.running = ev.kind === "backtest"; BT.live = null; btRepSrc(); logLine("$ " + ev.command, "cmd", "bt_log");
    $("bt_status").textContent = ev.kind === "chart" ? "Generating chart…" : `Backtesting ${ev.model}…`;
  } else if (ev.type === "line") {
    if (ev.bt) { if (ev.bt.start) liveStart(ev.bt.start); else if (ev.bt.point) livePoint(ev.bt.point); else if (ev.bt.phase) livePhase(ev.bt.phase);
                 else if (ev.bt.open) liveOpen(ev.bt.open); else if (ev.bt.trade) liveTrade(ev.bt.trade); return; }
    logLine(ev.line, ev.cls, "bt_log");
  } else if (ev.type === "done") {
    setBusy(false); BT.running = false; BT.done = BT.live; BT.live = null; btRepSrc();
    if (BT.done) { BT.live = BT.done; BT.live.follow = false; const c = $("bt_live"); if (c) c.remove(); candlesDraw(BT.live.cursor); }   // chart stays scrollable
    const ok = ev.rc === 0;
    logLine(ev.cancelled ? "[stop] cancelled" : ok ? "✓ Done" : `✗ Failed (exit ${ev.rc})`, ok ? "success" : "error", "bt_log");
    $("bt_status").textContent = ev.cancelled ? "Stopped" : ok ? (ev.kind === "chart" ? "✓ Chart ready — Open Chart" : "✓ Backtest complete") : `✗ Failed (exit ${ev.rc})`;
    // render the model the run was FOR, even if the dropdown changed meanwhile (CTk rule)
    if (ok && ev.kind === "backtest" && ev.model) { if ($("bt_model").value !== ev.model) toast(`report shown for ${ev.model} (the model of this run)`); btReport(ev.model); }
  }
}

window.rlbacktest = {
  async init() {
    window.rlweb.handlers.backtest = onBacktestEvent;
    $("bt_mode").onchange = btMode;
    addEventListener("resize", () => { if (BT.live) { liveDraw(); candlesDraw(+($("bt_cd_pos") || {}).value || undefined); } });
    $("bt_model").onchange = () => btReport($("bt_model").value);
    $("bt_csv").onchange = () => { btRange(); btAutoM1(); };
    $("bt_m1").onchange = () => { BT.m1Auto = null; btM1Hint(); };   // explicit choice: kept until the dataset changes
    $("bt_start").oninput = btRangeSoon;
    $("bt_m1_browse").onclick = async () => { const p = await api().browse_m1(); if (p) { ensureOption($("bt_m1"), p, p.split(/[\\/]/).pop() + " (browsed)"); $("bt_m1").value = p; BT.m1Auto = null; btM1Hint(); } };
    const run = async (fn) => { clearBad(); const r = await api()[fn](btForm()); if (!r.ok) { toast(r.error, "err"); markBad(r.field && r.field.startsWith("bt_") ? r.field : r.field && "bt_" + r.field); } };
    $("bt_run").onclick = () => run("start_backtest");
    $("bt_chart").onclick = () => run("generate_chart");
    $("bt_open_chart").onclick = async () => { const r = await api().open_chart($("bt_model").value); if (!r.ok) toast(r.error, "err"); };
    $("bt_stop").onclick = async () => { logLine("Stopping…", "warn", "bt_log"); await api().stop_run(); };
  },
  async show() {
    if (BT.loaded) return;
    const c = BT.cat = await api().backtest_catalog(), d = c.defaults;
    setOptions($("bt_model"), c.models.length ? c.models.map(m => [esc(m), esc(m)]) : [["", "(none)"]]);
    setOptions($("bt_source"), Object.entries(c.sources), d.source);
    setOptions($("bt_csv"), c.datasets.length ? c.datasets.map(x => [esc(x), esc(x)]) : [["", "(none)"]]);
    setOptions($("bt_m1"), c.m1_options.map(x => [esc(x), esc(x)]), "(none)");
    setOptions($("bt_mode"), Object.entries(c.modes), d.mode);
    setOptions($("bt_intrabar"), Object.entries(c.intrabar), d.intrabar);
    for (const k of BT_FIELDS) $("bt_" + k).value = d[k];
    BT.loaded = true; btMode();
    await Promise.all([btRange(), btAutoM1(), btReport($("bt_model").value)]);
  },
};
