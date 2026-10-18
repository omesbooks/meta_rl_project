// Backtest page (phase 3). Command, M1 auto-select, Skip-% preview and the report text
// all come from Python (webgui/backtest.py, backtest_report.py); this file only paints.
"use strict";
const BT = { cat: null, m1Auto: null, loaded: false, shownModel: "", runModel: "" };
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
  $("bt_rep_src").textContent = r.meta_path.split(/[\\/]/).slice(-3).join("/") + (r.status && r.status !== "complete" ? ` · status: ${r.status}` : "");
  const c = r.cards;
  $("bt_cards").innerHTML = card("WIN RATE", c.win_rate) + card("PROFIT FACTOR", c.profit_factor) + card("RETURN", c.return) + card("MAX DD", c.max_dd);
  $("bt_equity").innerHTML = r.zero_trades ? `<div class="empty">0 trades — no equity chart</div>` : r.equity ? `<img src="${r.equity}" alt="equity curve">` : `<div class="empty">no equity image stored for this run</div>`;
  $("bt_mcimg").innerHTML = r.mc ? `<img src="${r.mc}" alt="Monte Carlo fan chart">` : "";
  $("bt_report").innerHTML = "<table>" + r.rows.map(([k, v]) => k === "" && v === "" ? `<tr class="sep"><td colspan="2"></td></tr>` :
    `<tr class="${v.startsWith("->") ? "verdict" : ""}"><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("") + "</table>";
  $("bt_report_txt").textContent = r.lines.join("\n");
}

function onBacktestEvent(ev) {
  if (ev.type === "start") {
    setBusy(true, ev.kind === "chart" ? "generating chart" : "backtesting");
    BT.runModel = ev.model; logLine("$ " + ev.command, "cmd", "bt_log");
    $("bt_status").textContent = ev.kind === "chart" ? "Generating chart…" : `Backtesting ${ev.model}…`;
  } else if (ev.type === "line") logLine(ev.line, ev.cls, "bt_log");
  else if (ev.type === "done") {
    setBusy(false);
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
