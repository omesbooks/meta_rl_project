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
  $("bt_equity").innerHTML = r.zero_trades ? `<div class="empty">0 trades — no equity chart</div>` : r.equity ? `<img src="${r.equity}" alt="equity curve">` : `<div class="empty">no equity image stored for this run</div>`;
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
// ---- live equity while backtest_live.py runs (BT_LIVE lines, ~150 per run; nothing is stored until the run ends)
const fmtPct = v => (v >= 0 ? "+" : "") + (100 * v).toFixed(2) + "%";
function liveStart(s) {
  BT.live = { initial: s.initial, bars: s.bars, pts: [], last: null, t0: "", t1: "", raf: 0 };
  $("bt_cards").innerHTML = card("WIN RATE", ["–", "dim"]) + card("PROFIT FACTOR", ["–", "dim"]) + card("RETURN", ["–", "dim"]) + card("MAX DD", ["–", "dim"]);
  $("bt_equity").innerHTML = `<canvas id="bt_live" width="900" height="220" style="width:100%;max-width:100%;border-radius:12px"></canvas>`;
  $("bt_mcimg").innerHTML = ""; $("bt_report").innerHTML = `<div class="empty">live run · the full report (realism, baseline, Monte Carlo) appears when it finishes</div>`; $("bt_report_txt").textContent = "";
  const el = $("bt_rep_src"); el.textContent = `live · ${BT.runModel} · bar 0 / ${s.bars.toLocaleString()}`; el.className = "r meta ok";
}
function livePoint(p) {
  const L = BT.live; if (!L) return;
  L.pts.push([p.bar, p.equity]); L.last = p; if (!L.t0) L.t0 = p.time; L.t1 = p.time;
  const ret = p.balance / L.initial - 1, pf = p.gross_loss > 0 ? p.gross_win / p.gross_loss : (p.gross_win > 0 ? Infinity : NaN), wr = p.trades ? p.wins / p.trades : NaN;
  $("bt_cards").innerHTML =
    card("WIN RATE", [isNaN(wr) ? "–" : (100 * wr).toFixed(1) + "%", wr >= 0.5 ? "ok" : "dim"]) +
    card("PROFIT FACTOR", [isNaN(pf) ? "–" : pf === Infinity ? "∞" : pf.toFixed(2), pf >= 1 ? "ok" : "bad"]) +
    card("RETURN", [fmtPct(ret), ret >= 0 ? "ok" : "bad"]) + card("MAX DD", [fmtPct(p.max_dd), p.max_dd > -0.1 ? "ok" : "bad"]);
  $("bt_rep_src").textContent = `live · ${BT.runModel} · bar ${p.bar.toLocaleString()} / ${p.bars.toLocaleString()} · ${p.trades} trades · ${p.time}`;
  $("bt_status").textContent = `Backtesting ${BT.runModel}… ${Math.round(100 * p.bar / p.bars)}%`;
  if (!L.raf) L.raf = requestAnimationFrame(() => { L.raf = 0; liveDraw(); });
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
  const fg = col("--fg") || "#111", dim = col("--dim") || "#888", line = col("--line") || "#ccc", green = col("--green") || "#1c8a55", red = col("--red") || "#c24135", accent = col("--accent") || "#245b8f";
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
  g.fillText(hi.toFixed(0), x0 - 6, y0 + 10); g.fillText(lo.toFixed(0), x0 - 6, y1); g.fillText(L.initial.toFixed(0), x0 - 6, yI + 4);
  g.textAlign = "left"; g.fillStyle = fg; g.fillText(String(L.t0).slice(0, 16), x0, H - 8); g.textAlign = "right"; g.fillText(String(L.t1).slice(0, 16), x1, H - 8);
  g.textAlign = "center"; g.fillStyle = dim; g.fillText(`equity · live · ${lb.toLocaleString()} / ${L.bars.toLocaleString()} bars`, (x0 + x1) / 2, y0 + 10);
}
function onBacktestEvent(ev) {
  if (ev.type === "start") {
    setBusy(true, ev.kind === "chart" ? "generating chart" : "backtesting");
    BT.runModel = ev.model; BT.running = ev.kind === "backtest"; BT.live = null; btRepSrc(); logLine("$ " + ev.command, "cmd", "bt_log");
    $("bt_status").textContent = ev.kind === "chart" ? "Generating chart…" : `Backtesting ${ev.model}…`;
  } else if (ev.type === "line") {
    if (ev.bt) { if (ev.bt.start) liveStart(ev.bt.start); else if (ev.bt.point) livePoint(ev.bt.point); else if (ev.bt.phase) livePhase(ev.bt.phase); return; }
    logLine(ev.line, ev.cls, "bt_log");
  } else if (ev.type === "done") {
    setBusy(false); BT.running = false; BT.live = null; btRepSrc();
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
    addEventListener("resize", () => BT.live && liveDraw());
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
