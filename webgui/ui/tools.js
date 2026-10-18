// Data Tools, Analyze and Regime Check pages (phase 5). The work and every rule live in
// Python (data_tools.py, regime_tools.py, webgui/api.py); this file only paints.
"use strict";
const TL = { loaded: false, outAuto: null, exportAuto: null, regimeCsv: "", resultsCsv: null, regimeSel: null };

function tlBad(field) { if (field && $(field)) markBad(field); }
function selectOptions(sel, items, keep) {
  const cur = sel.value;
  setOptions(sel, items.length ? items.map(x => [esc(x), esc(x)]) : [["", "(none)"]]);
  if (keep && items.includes(cur)) sel.value = cur;
}

// ------------------------------------------------------------------ data tools
function onToolsEvent(ev) {
  if (ev.type === "start") setBusy(true, "data tools: " + ev.kind);
  else if (ev.type === "line") logLine(ev.line, ev.cls, "tool_log");
  else if (ev.type === "notice") {
    const cls = { success: "ok", warn: "warn", error: "err" }[ev.level] || "";
    $("tool_notice").textContent = `${ev.title}: ${ev.message}` + (ev.paths.length ? "\n" + ev.paths.join(" | ") : "");
    $("tool_notice").className = "meta " + cls; $("tool_notice").style.whiteSpace = "pre-line";
    logLine(`[notice] ${ev.title}: ${ev.message}` + (ev.paths.length ? " | " + ev.paths.join(" | ") : ""), ev.level, "tool_log");
    if (ev.popup) toast(`${ev.title} — ${ev.message}`, ev.level === "error" ? "err" : ev.level === "warn" ? "" : "ok");
  } else if (ev.type === "done") {
    setBusy(false);
    if (ev.kind === "correlation" && ev.png)
      $("tool_corr_img").innerHTML = `<img src="${ev.png}" alt="correlation heatmap" title="click to open full size" style="cursor:zoom-in">`,
      $("tool_corr_img").querySelector("img").onclick = () => api().tools_open(ev.png_path);
    if (["import", "split", "clean"].includes(ev.kind) && ev.ok) toolsRefresh(true);   // new CSVs appear in every dropdown
  }
}
async function toolsRefresh(keep) {
  const c = await api().tools_catalog();
  for (const id of ["tool_split_csv", "tool_feat_csv"]) selectOptions($(id), c.csvs, keep);
  selectOptions($("tool_export_model"), c.models, keep);
  const src = $("tool_src").value;
  selectOptions($("tool_src"), c.collector, true);
  if (!keep || src !== $("tool_src").value) toolSrcChange();
  $("tool_common_dir").textContent = `${c.collector.length} CSV(s) in ${c.common_dir}`;
  // other pages list CSVs too: drop their cache so they reload on the next visit
  if (typeof FL !== "undefined") FL.loaded = false;
  if (typeof BT !== "undefined") BT.loaded = false;
  return c;
}
function toolSrcChange() {   // CTk _on_collector_src_change: follow the source until the user types a name
  const v = $("tool_src").value; if (!v) return;
  const def = "training_data_" + v.replace(/\.csv$/i, "") + ".csv", cur = $("tool_out").value.trim();
  if (!cur || cur === TL.outAuto) $("tool_out").value = def;
  TL.outAuto = def;
}
async function toolRun(fn, form, retry) {
  clearBad();
  let r = await api()[fn](form, ...(retry ? [true] : []));
  if (!r.ok && r.confirm) { if (!confirm(r.confirm)) return; r = await api()[fn](form, true); }
  if (!r.ok) { toast(r.error, "err"); tlBad(r.field); return; }
  if (fn === "tools_export" && r.deploy && r.deploy !== $("tool_export_name").value) $("tool_export_name").value = r.deploy;
}

// ------------------------------------------------------------------ analyze
function onAnalyzeEvent(ev) {
  if (ev.type === "start") { setBusy(true, "analyzing"); logLine("$ " + ev.command, "cmd", "an_log"); $("an_status").textContent = `Analyzing ${ev.model}…`; }
  else if (ev.type === "line") logLine(ev.line, ev.cls, "an_log");
  else if (ev.type === "done") {
    setBusy(false); const ok = ev.rc === 0;
    $("an_status").textContent = ev.cancelled ? "Stopped" : ok ? "✓ Done" : `✗ Failed (exit ${ev.rc})`;
    logLine(ev.cancelled ? "[stop] cancelled" : ok ? "✓ Done" : `✗ Failed (exit ${ev.rc})`, ok ? "success" : "error", "an_log");
  }
}

// ------------------------------------------------------------------ regime check
function regimeMethod() { const r = document.querySelector("input[name=regime_method]:checked"); return r ? r.value : "hmm"; }
function regimeParams() {
  document.querySelectorAll("#p-regime .rp").forEach(e => e.style.display = e.classList.contains("rp-" + regimeMethod()) ? "" : "none");
}
function regimeForm() {
  const f = { csv: TL.regimeCsv, method: regimeMethod() };
  for (const k of ["hmm_states", "km_k", "km_win", "pelt_pen", "ev_symbol", "ev_topk"]) f[k] = $("regime_" + k).value;
  return f;
}
async function regimePick(r) {
  if (!r.ok) { if (!r.cancelled) toast(r.error, "err"); return; }
  TL.regimeCsv = r.path; $("regime_meta").textContent = r.meta;
  ensureOption($("regime_csv"), r.path, r.name); $("regime_csv").value = r.path;
  if (TL.resultsCsv && TL.resultsCsv !== r.path) {      // CTk: results from another CSV are stale
    $("regime_table").querySelector("tbody").innerHTML = ""; TL.regimeSel = null;
    $("regime_summary").textContent = "Breakpoints are from a previous CSV — re-run detection"; $("regime_summary").className = "warnc";
  }
}
async function regimeResults() {
  const r = await api().regime_results();
  $("regime_ev_status").textContent = r.status;
  if (!r.ok) { if (!r.missing) logLine(r.error, "error", "regime_log"); else logLine(r.error, "warn", "regime_log"); return; }
  TL.resultsCsv = r.csv; TL.regimeSel = null;
  $("regime_summary").textContent = r.summary; $("regime_summary").className = r.good ? "goodc" : "warnc";
  $("regime_table").querySelector("tbody").innerHTML = r.rows.map(([d, m, x], i) => `<tr data-i="${i}" data-d="${esc(d)}" data-m="${esc(m)}"><td>${esc(d)}</td><td>${esc(m)}</td><td>${esc(x)}</td></tr>`).join("")
    || `<tr><td colspan="3" class="empty">no breakpoints found</td></tr>`;
  $("regime_table").querySelectorAll("tbody tr[data-d]").forEach(tr => tr.onclick = () => {
    TL.regimeSel = { date: tr.dataset.d, match: tr.dataset.m };
    $("regime_table").querySelectorAll("tbody tr").forEach(x => x.classList.toggle("on", x === tr));
  });
}
function onRegimeEvent(ev) {
  if (ev.type === "start") { setBusy(true, ev.kind === "gemini" ? "labeling with Gemini" : "detecting regime"); logLine("$ " + ev.command, "cmd", "regime_log"); }
  else if (ev.type === "line") logLine(ev.line, ev.cls, "regime_log");
  else if (ev.type === "done") {
    setBusy(false); const ok = ev.rc === 0;
    logLine(ev.cancelled ? "[stop] cancelled" : ok ? "✓ Done" : `✗ Failed (exit ${ev.rc})`, ok ? "success" : "error", "regime_log");
    if (ok && ev.kind === "detect") regimeResults();
    else api().regime_results().then(r => $("regime_ev_status").textContent = r.status);   // Gemini path refreshes the events line
  }
}
async function regimeCutoff() {
  if (!TL.regimeSel) return toast("Click a breakpoint row in the table first.", "err");
  if (!TL.regimeCsv) return toast("No CSV loaded.", "err");
  const { date, match } = TL.regimeSel, pv = await api().regime_cutoff_preview(TL.regimeCsv, date);
  const msg = `Filter CSV to rows where timestamp >= ${date}\n${match !== "—" ? "(" + match + ")" : ""}\n\nSource:  ${pv.source}\nOutput:  ${pv.output}\n\n` +
    "After saving, you'll be moved to the Train page with the new file pre-selected.";
  if (!confirm(msg)) return;
  const r = await api().regime_cutoff(TL.regimeCsv, date, TL.resultsCsv);
  if (!r.ok) { toast((r.title ? r.title + ": " : "") + r.error, "err"); return; }
  logLine(r.log, "success", "regime_log");
  // jump to Train with the new CSV pre-selected (CTk _set_train_csv)
  const csvs = await api().list_csvs(), name = r.path.split(/[\\/]/).pop(), sel = $("csv");
  setOptions(sel, csvs.map(c => [c, c]));
  const value = csvs.includes(name) && r.path.replace(/\\/g, "/").startsWith(((await api().hello()).root).replace(/\\/g, "/")) ? name : r.path;
  ensureOption(sel, value, value === name ? name : name + " (regime cutoff)"); sel.value = value;
  document.querySelector("#side a[data-p=train]").click(); refreshCsv();
  toast(`✓ ${name}: ${r.after.toLocaleString()} / ${r.before.toLocaleString()} rows kept — selected on the Train page`, "ok");
}

// ------------------------------------------------------------------ wiring
window.rltools = {
  async init() {
    Object.assign(window.rlweb.handlers, { tools: onToolsEvent, analyze: onAnalyzeEvent, regime: onRegimeEvent });
    Object.assign(window.rlpages.live, { tools: "import · split · feature cleanup · ONNX export — no command line",
      analyze: "rl_analyze.py · confidence → accuracy", regime: "regime_compare.py · breakpoints → train cutoff" });
    // data tools
    $("tool_src_refresh").onclick = async () => { const c = await toolsRefresh(true); logLine(`[collector] ${c.collector.length} CSV(s) in Common\\Files`, "info", "tool_log"); };
    $("tool_src").onchange = toolSrcChange;
    $("tool_split_preset").onchange = () => { const v = $("tool_split_preset").value; if (v) $("tool_split_date").value = v.split(" ")[0]; };
    $("tool_export_model").onchange = () => {   // CTk: deploy name follows the model until the user types one
      const m = $("tool_export_model").value, cur = $("tool_export_name").value.trim();
      if (m && (!cur || cur === "rl_v10" || cur === TL.exportAuto)) { $("tool_export_name").value = m; TL.exportAuto = m; }
    };
    $("tool_import").onclick = () => toolRun("tools_import", { src: $("tool_src").value, out: $("tool_out").value, m1: $("tool_m1").checked });
    $("tool_split").onclick = () => toolRun("tools_split", { csv: $("tool_split_csv").value, split_date: $("tool_split_date").value, train_suffix: $("tool_train_suffix").value,
      test_suffix: $("tool_test_suffix").value, clean_after: $("tool_clean_after").checked, clean_thr: $("tool_clean_thr").value });
    const feat = () => ({ csv: $("tool_feat_csv").value, threshold: $("tool_feat_thr").value, train_pct: $("tool_feat_train_pct").value });
    $("tool_corr").onclick = () => toolRun("tools_correlation", feat());
    $("tool_clean").onclick = () => toolRun("tools_clean", feat());
    $("tool_export").onclick = () => toolRun("tools_export", { model: $("tool_export_model").value, deploy: $("tool_export_name").value,
      source: $("tool_export_source").value, outdir: $("tool_export_outdir").value });
    // analyze
    $("an_run").onclick = async () => { clearBad(); const r = await api().start_analyze({ model: $("an_model").value, csv: $("an_csv").value }); if (!r.ok) { toast(r.error, "err"); tlBad(r.field); } };
    $("an_stop").onclick = async () => { logLine("Stopping…", "warn", "an_log"); await api().stop_run(); };
    // regime
    $("regime_csv").onchange = async () => regimePick(await api().regime_csv($("regime_csv").value));
    $("regime_browse").onclick = async () => regimePick(await api().regime_browse());
    $("regime_run").onclick = async () => { clearBad(); const r = await api().regime_detect(regimeForm()); if (!r.ok) { toast(r.error, "err"); tlBad(r.field); } };
    $("regime_stop").onclick = async () => { logLine("Stopping…", "warn", "regime_log"); await api().stop_run(); };
    $("regime_chart").onclick = async () => { const r = await api().open_regime_chart(); if (!r.ok) toast(r.error, "err"); };
    $("regime_gemini").onclick = async () => {
      clearBad(); let r = await api().regime_gemini(regimeForm(), false);
      if (!r.ok && r.need_key) {
        if (!confirm("No Gemini API key found in Settings.\n\nContinue anyway? (You'll get shock dates but no event names — events will be labeled 'Shock @ YYYY-MM'.)\n\nCancel = go to Settings and add a key first."))
          return document.querySelector("#side a[data-p=settings]").click();
        r = await api().regime_gemini(regimeForm(), true);
      }
      if (!r.ok) { toast(r.error, "err"); tlBad(r.field); }
    };
    $("regime_cutoff").onclick = regimeCutoff;
  },
  async show(page) {
    if (!TL.loaded) {
      const c = await toolsRefresh(false), d = c.defaults;
      $("tool_m1").checked = d.import_m1;
      for (const [id, k] of [["tool_split_date", "split_date"], ["tool_train_suffix", "train_suffix"], ["tool_test_suffix", "test_suffix"],
        ["tool_clean_thr", "clean_thr"], ["tool_feat_thr", "feat_thr"], ["tool_feat_train_pct", "feat_train_pct"]]) $(id).value = d[k];
      setOptions($("tool_split_preset"), [["", "Quick presets"], ...c.split_presets.map(p => [esc(p), esc(p)])], "");
      setOptions($("tool_export_source"), c.export_sources.map(s => [esc(s), esc(s)]));
      $("tool_export_model").onchange();
      selectOptions($("an_model"), c.models); selectOptions($("an_csv"), c.csvs);
      const rc = await api().regime_catalog(), rd = rc.defaults;
      $("regime_methods").innerHTML = rc.methods.map(([v, label, desc]) =>
        `<label class="radio"><input type="radio" name="regime_method" value="${v}" ${v === rd.method ? "checked" : ""}><b>${esc(label)}</b><span>${esc(desc)}</span></label>`).join("");
      document.querySelectorAll("input[name=regime_method]").forEach(r => r.onchange = regimeParams);
      for (const k of ["hmm_states", "km_k", "km_win", "pelt_pen", "ev_symbol", "ev_topk"]) $("regime_" + k).value = rd[k];
      setOptions($("regime_csv"), [["", "— pick a CSV —"], ...rc.csvs.map(x => [esc(x), esc(x)])], "");
      $("regime_ev_status").textContent = rc.status;
      $("regime_key_hint").textContent = rc.has_key ? "Gemini key: ✓ saved in Settings" : "no Gemini key — set one in Settings (or continue without names)";
      regimeParams(); TL.loaded = true;
      regimeResults();
    }
  },
};
