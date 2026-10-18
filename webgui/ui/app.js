// RL Trading Studio — web GUI front end. Renders state and calls window.pywebview.api;
// every rule (validation, command building, recipe hash) lives in Python (webgui/).
"use strict";
const $ = id => document.getElementById(id);
// every bridge call returns ASCII-escaped JSON text (webgui/api.py JsBridge) -> decode here
const _raw = () => window.pywebview && window.pywebview.api;
const _proxy = new Proxy({}, { get: (_, name) => async (...args) => { const r = await _raw()[name](...args); return typeof r === "string" ? JSON.parse(r) : r; } });
const api = () => (_raw() ? _proxy : null);
let CAT = null, busy = false, rewardJson = null, actionJson = null, rewardValues = {}, actionValues = {};
const MAX_LOG = 2000;
const PPO_FIELDS = [
  ["lr", "Learning Rate ⭐", "lower it if KL/clip run high (3e-4 → 1e-4)"], ["clip", "Clip Range", "PPO core: policy may not move > 20% per update"],
  ["ent", "Entropy Coef", "raise it if exploration dies (0.01 → 0.05)"], ["nsteps", "n_steps", "rollout buffer per env (more = stabler)"],
  ["nepochs", "n_epochs", "SGD passes per rollout (10 = normal)"], ["batch", "batch_size", "minibatch (must be ≤ n_steps)"],
  ["gamma", "gamma (γ)", "discount factor (how far ahead)"], ["gae", "gae_lambda (λ)", "advantage smoothing (0.9–0.99)"], ["vf", "vf_coef", "critic loss weight (0.25–1.0)"],
];
const PAGES = { tools: "🛠️ Data Tools", pipeline: "▶ Full Pipeline", train: "🎯 Train New Model", backtest: "📊 Backtest Model", walkfwd: "🔬 Walk-Forward Validation",
  finetune: "🔄 Fine-tune Existing Model", analyze: "🔍 Confidence Analysis", regime: "🌐 Regime Detection", models: "🗂️ Model Library", settings: "⚙️ Settings" };

// ------------------------------------------------------------------ helpers
function toast(msg, kind = "") { const t = $("toast"); t.textContent = msg; t.className = "glass on " + kind; clearTimeout(toast.h); toast.h = setTimeout(() => t.className = "glass", kind === "err" ? 7000 : 3500); }
function fmtNum(v, d) { return Number.isFinite(+v) ? (+v).toFixed(d) : String(v); }
function setOptions(sel, items, value) { sel.innerHTML = items.map(([v, l]) => `<option value="${v}">${l}</option>`).join(""); if (value != null) sel.value = value; }
function clearBad() { document.querySelectorAll(".bad").forEach(e => e.classList.remove("bad")); }
function markBad(field) {
  if (!field) return;
  const base = String(field).split(".")[0], el = $(field.replace("reward_values.", "rw_").replace("action_values.", "ac_")) || $(base);
  if (el) { el.classList.add("bad"); el.scrollIntoView({ block: "center", behavior: "smooth" }); el.focus && el.focus(); }
  if (base === "advanced") $("advPPO").open = true;
  if (base === "reward_values" || base === "reward_formula") $("advReward").open = true;
  if (base === "action_values") $("advAction").open = true;
}
function debounce(fn, ms) { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; }

// ------------------------------------------------------------------ sliders
function specScale(s) { return Number(s.display_scale || 1); }
function specDec(s) { return Number(s.display_decimals ?? s.decimals ?? 3); }
function sliderCard(prefix, s, raw, onChange) {
  const sc = specScale(s), dec = specDec(s), lo = s.min * sc, hi = s.max * sc, step = (hi - lo) / Math.max(1, s.steps || 100);
  const box = document.createElement("div"); box.className = "sl"; box.id = prefix + "card_" + s.key;
  box.innerHTML = `<div class="t"><span>${s.label}</span><input id="${prefix}${s.key}" value="${(raw * sc).toFixed(dec)}"></div>
    <input type="range" min="${lo}" max="${hi}" step="${step}" value="${raw * sc}">
    <div class="d">${(s.description || s.unit || "")} (${+lo.toFixed(dec)}..${+hi.toFixed(dec)}${s.display_suffix || ""})</div>`;
  const txt = box.querySelector("input:not([type=range])"), rng = box.querySelector("input[type=range]");
  rng.oninput = () => { txt.value = (+rng.value).toFixed(dec); onChange(s.key, +rng.value / sc); };
  txt.onchange = () => { const v = parseFloat(txt.value); if (Number.isFinite(v)) { rng.value = v; onChange(s.key, v / sc); } else onChange(s.key, txt.value); };
  return box;
}
function renderRewardSliders() {
  const wrap = $("rewardSliders"); wrap.innerHTML = "";
  const prof = CAT.reward_profiles.find(p => p.key === $("reward_profile").value);
  const base = Object.assign({}, prof.params, (rewardJson && rewardJson.overrides) || {});
  for (const s of CAT.reward_specs) {
    if (!(s.key in prof.params)) continue;
    const card = sliderCard("rw_", s, rewardValues[s.key] ?? prof.params[s.key], (k, v) => { rewardValues[k] = v; card.classList.toggle("changed", Math.abs(v - base[k]) > Math.pow(10, -(Number(s.decimals ?? 3) + 1))); });
    card.classList.toggle("changed", Math.abs((rewardValues[s.key] ?? base[s.key]) - base[s.key]) > Math.pow(10, -(Number(s.decimals ?? 3) + 1)));
    wrap.appendChild(card);
  }
  const mtm = $("reward_mode").value === "mtm";
  wrap.querySelectorAll("input").forEach(i => i.disabled = mtm); $("formula_on").disabled = mtm; $("formula").disabled = mtm;
  $("rmHint").textContent = mtm ? "MTM mode: reward profile, sliders and formula are IGNORED (legacy mark-to-market reward)" : "realized: reward on closed trades (recommended)";
  $("rmHint").className = "hint " + (mtm ? "warn" : "");
}
function renderActionSliders() {
  const wrap = $("actionSliders"); wrap.innerHTML = "";
  const prof = CAT.action_profiles.find(p => p.key === $("action_profile").value);
  for (const s of CAT.action_specs) {
    if (!(s.key in prof.params)) continue;
    wrap.appendChild(sliderCard("ac_", s, actionValues[s.key] ?? prof.params[s.key], (k, v) => { actionValues[k] = v; }));
  }
}
function resetReward(clearJson = true) { if (clearJson) { rewardJson = null; $("rwJsonHint").textContent = "no reward JSON"; } const prof = CAT.reward_profiles.find(p => p.key === $("reward_profile").value); rewardValues = Object.assign({}, prof.params); renderRewardSliders(); }
function resetAction(clearJson = true) { if (clearJson) { actionJson = null; $("acJsonHint").textContent = "no action JSON"; } const prof = CAT.action_profiles.find(p => p.key === $("action_profile").value); actionValues = Object.assign({}, prof.params); renderActionSliders(); }

// ------------------------------------------------------------------ form
function form() {
  const f = { csv: $("csv").value, name: $("name").value, algo: $("algo").value, reward_mode: $("reward_mode").value,
    reward_profile: $("reward_profile").value, action_profile: $("action_profile").value,
    reward_values: rewardValues, action_values: actionValues, reward_formula_enabled: $("formula_on").checked, reward_formula: $("formula").value,
    reward_json: rewardJson, action_json: actionJson, viz: $("viz").checked, viz_mode: $("viz_mode").value };
  for (const k of ["steps", "window", "max_hold", "train_pct", "mc_eval", "mc_skip", "corr_threshold", "entry_gate", "spread_pct", "commission", "seed"]) f[k] = $(k).value;
  for (const [k] of PPO_FIELDS) f[k] = $("ppo_" + k).value;
  return f;
}
async function refreshCsv() {
  const csv = $("csv").value;
  if (!csv) { $("dsMeta").textContent = "no CSV in the project root"; return; }
  const info = await api().csv_info(csv, $("train_pct").value, $("window").value);
  $("dsName").textContent = csv.split(/[\/]/).pop();
  if (!info.ok) { $("dsMeta").textContent = info.error; $("dsMeta").className = "hint err"; $("dsSplit").innerHTML = ""; $("useSuggest").disabled = true; return; }
  $("dsMeta").textContent = `${info.rows.toLocaleString()} rows · ${info.cols} columns · ` + (info.params ? "✓ .params.json (parity guaranteed)" : "⚠ no .params.json (EA will use defaults)");
  $("dsMeta").className = "hint " + (info.params ? "ok" : "warn");
  $("dsSplit").innerHTML = info.split.map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("") || `<span class="meta">no timestamp column</span>`;
  $("gateList").innerHTML = info.gate_columns.map(c => `<option value="${c}">`).join("");
  if (info.suggest) { $("suggestHint").textContent = `Suggested: ${info.suggest.steps.toLocaleString()} steps (~${info.suggest.passes} passes over ${info.suggest.train_bars.toLocaleString()} train bars)`; $("suggestHint").className = "hint ok"; $("useSuggest").disabled = false; $("useSuggest").onclick = () => $("steps").value = info.suggest.steps; }
  else { $("suggestHint").textContent = "Suggested: –"; $("suggestHint").className = "hint"; $("useSuggest").disabled = true; }
}
const refreshCsvSoon = debounce(refreshCsv, 300);

// ------------------------------------------------------------------ health
function renderHealthShell() {
  const names = Object.keys(CAT.metrics);
  $("health").innerHTML = names.map(n => `<div class="kpi" id="k_${n}"><div class="k">${CAT.metrics[n].label.toUpperCase()}</div><div class="v">—</div><div class="u">good ${CAT.metrics[n].good.map(v => v == null ? "∞" : v).join(" … ")}</div></div>`).join("");
  $("pills").innerHTML = names.map(n => `<span class="pill" id="p_${n}">${n}<b>—</b></span>`).join("");
}
const rewards = [];
function onMetric(m) {
  const k = $("k_" + m.name), p = $("p_" + m.name), d = Math.abs(m.value) < 0.1 ? 4 : 3;
  if (k) { k.className = "kpi " + (m.status || ""); k.querySelector(".v").textContent = fmtNum(m.value, d); }
  if (p) { p.className = "pill " + (m.status || ""); p.querySelector("b").textContent = fmtNum(m.value, d); }
  if (m.name === "ep_rew_mean") { rewards.push(m.value); drawTrend(); }
}
function drawTrend() {
  const w = 600, h = 140, pad = 10, n = rewards.length;
  if (n < 2) { $("trend").innerHTML = `<text x="10" y="70" fill="#5b6b85" font-size="13">waiting for ep_rew_mean …</text>`; return; }
  const lo = Math.min(...rewards), hi = Math.max(...rewards), span = hi - lo || 1;
  const X = i => pad + i * (w - 2 * pad) / (n - 1), Y = v => h - pad - (v - lo) / span * (h - 2 * pad - 14);
  const zero = lo < 0 && hi > 0 ? `<line x1="${pad}" x2="${w - pad}" y1="${Y(0)}" y2="${Y(0)}" stroke="#5b6b85" stroke-dasharray="3 3"/>` : "";
  $("trend").innerHTML = zero + `<polyline fill="none" stroke="#0a66ff" stroke-width="2.4" points="${rewards.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")}"/>` +
    `<text x="${w - pad}" y="14" text-anchor="end" fill="#0a66ff" font-size="13" font-weight="600">${rewards[n - 1].toFixed(3)}</text><text x="${pad}" y="14" fill="#5b6b85" font-size="11">${n} updates</text>`;
}

// ------------------------------------------------------------------ run state + events
function setBusy(b, text) {
  busy = b; if (b) $("pills").classList.remove("off");
  // one subprocess at a time across pages (same as the CTk ProcessRunner): every run button waits
  document.querySelectorAll("[data-run]").forEach(x => x.disabled = b); document.querySelectorAll("[data-stop]").forEach(x => x.disabled = !b);
  document.querySelector("#status .dot").className = "dot" + (b ? " run" : ""); $("statusText").textContent = text || (b ? "training" : "idle");
}
function logLine(text, cls = "", id = "log") {
  const el = $(id), atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  const span = document.createElement("span"); span.className = cls; span.textContent = text + "\n"; el.appendChild(span);
  while (el.childNodes.length > MAX_LOG) el.removeChild(el.firstChild);
  if (atEnd) el.scrollTop = el.scrollHeight;
}
function onTrainEvent(ev) {
  if (ev.type === "start") { setBusy(true); rewards.length = 0; drawTrend(); logLine("$ " + ev.command, "cmd"); $("runStatus").textContent = "Starting…"; $("bar").style.width = "0"; }
  else if (ev.type === "line") {
    logLine(ev.line, ev.cls);
    if (ev.progress) { const p = Math.min(1, ev.progress.step / ev.progress.total); $("bar").style.width = (100 * p).toFixed(1) + "%"; $("runStatus").textContent = `Step ${ev.progress.step.toLocaleString()} / ${ev.progress.total.toLocaleString()}  (${(100 * p).toFixed(1)}%)`; }
    if (ev.metric) onMetric(ev.metric);
    if (ev.viz) $("statusText").textContent = `training · visual: ${ev.viz.kind === "rollout" ? "training rollout" : "eval"} ${ev.viz.index + 1} @ ${ev.viz.step.toLocaleString()}`;
  } else if (ev.type === "done") {
    setBusy(false);
    const ok = ev.rc === 0;
    $("runStatus").textContent = ev.cancelled ? "Stopped" : ok ? "✓ Training complete" : `✗ Training failed (exit ${ev.rc})`;
    logLine(ev.cancelled ? "[stop] training cancelled" : ok ? "[done] training complete" : `[error] exit code ${ev.rc}`, ok ? "success" : "error");
    if (ok) $("bar").style.width = "100%";
  }
}
// Python pushes event batches here; each event goes to the page that started the run (owner page)
window.rlweb = {
  handlers: { train: onTrainEvent },
  onEvents(events) { for (const ev of events) (this.handlers[ev.page] || onTrainEvent)(ev); },
};

// ------------------------------------------------------------------ actions
async function startTraining() {
  clearBad();
  $("start").disabled = true; $("runStatus").textContent = "Checking training data…";
  let r;
  try { r = await api().review_training(form()); } catch (e) { r = { ok: false, error: String(e) }; }
  $("start").disabled = busy;
  if (!r.ok) { $("runStatus").textContent = "Review failed"; toast(r.error, "err"); markBad(r.field); return; }
  $("runStatus").textContent = "Awaiting confirmation";
  $("mSummary").innerHTML = r.summary.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  $("mWarn").innerHTML = (r.existing ? `<div class="warnbox">⚠ <b>${$("name").value}</b> already exists at ${r.existing}.<br>The new run replaces the active version only after it succeeds; previous files are kept.</div>` : "") +
    r.warnings.map(w => `<div class="warnbox">${w}</div>`).join("");
  $("mOk").textContent = r.existing ? "Overwrite & start training" : "Start training";
  $("mCmd").textContent = r.command + "\n\nrecipe sha256: " + r.recipe_sha256;
  $("mRecipe").textContent = JSON.stringify(r.recipe, null, 2);
  $("modal").classList.add("on");
  $("mOk").onclick = async () => { $("modal").classList.remove("on"); const c = await api().confirm_training(r.token); if (!c.ok) toast(c.error, "err"); };
  $("mCancel").onclick = () => { $("modal").classList.remove("on"); $("runStatus").textContent = "Ready"; };
}

async function init() {
  CAT = await api().catalog();
  const d = CAT.defaults;
  setOptions($("algo"), Object.entries(CAT.algos), d.algo);
  setOptions($("reward_mode"), Object.entries(CAT.reward_modes), d.reward_mode);
  setOptions($("reward_profile"), CAT.reward_profiles.map(p => [p.key, p.label]), d.reward_profile);
  setOptions($("action_profile"), CAT.action_profiles.map(p => [p.key, `${p.label} (${p.actions.length} actions)`]), d.action_profile);
  setOptions($("viz_mode"), Object.entries(CAT.viz_modes), d.viz_mode);
  for (const k of ["name", "steps", "window", "max_hold", "train_pct", "mc_eval", "mc_skip", "corr_threshold", "entry_gate", "spread_pct", "commission", "seed"]) $(k).value = d[k];
  $("viz").checked = !!d.viz; $("formula").value = d.reward_formula;
  $("ppoFields").innerHTML = PPO_FIELDS.map(([k, l, h]) => `<div class="sl"><div class="t"><span>${l}</span><input id="ppo_${k}" value="${d[k]}"></div><div class="d">${h}</div></div>`).join("");
  document.querySelectorAll("[data-preset]").forEach(b => b.onclick = () => { const p = CAT.presets[b.dataset.preset]; for (const [k] of PPO_FIELDS) $("ppo_" + k).value = p[k]; toast(`preset: ${b.textContent.trim()}`, "ok"); });
  resetReward(); resetAction(); renderHealthShell(); drawTrend();
  const csvs = await api().list_csvs();
  setOptions($("csv"), csvs.map(c => [c, c]));
  if (!csvs.length) $("csv").innerHTML = `<option value="">no CSV in the project root</option>`;
  $("csv").onchange = refreshCsv; $("train_pct").oninput = refreshCsvSoon; $("window").oninput = refreshCsvSoon;
  refreshCsv();
  $("reward_profile").onchange = () => resetReward(true); $("action_profile").onchange = () => resetAction(true); $("reward_mode").onchange = renderRewardSliders;
  $("rwReset").onclick = () => resetReward(false); $("acReset").onclick = () => resetAction(false);
  $("formula_on").onchange = () => $("formulaBox").style.display = $("formula_on").checked ? "block" : "none";
  $("formula").onchange = async () => { const r = await api().validate_formula($("formula").value); $("formulaHint").textContent = r.ok ? "Formula valid; it will override realized reward shaping." : "Invalid formula: " + r.error; $("formulaHint").className = "hint " + (r.ok ? "ok" : "err"); };
  $("rwLoad").onclick = async () => { const r = await api().load_reward_json(); if (!r.ok) { if (!r.cancelled) toast(r.error, "err"); return; }
    $("reward_profile").value = r.base_profile; rewardJson = { path: r.path, overrides: r.overrides, has_formula: !!r.formula };
    const prof = CAT.reward_profiles.find(p => p.key === r.base_profile); rewardValues = Object.assign({}, prof.params, r.overrides);
    $("formula_on").checked = !!r.formula; $("formula").value = r.formula || CAT.defaults.reward_formula; $("formula_on").onchange();
    $("rwJsonHint").textContent = "JSON: " + r.path.split(/[\\/]/).pop(); renderRewardSliders(); toast(`Loaded ${r.name}`, "ok"); };
  $("rwSave").onclick = async () => { const r = await api().save_reward_json(form()); if (!r.ok) { if (!r.cancelled) { toast(r.error, "err"); markBad(r.field); } return; }
    rewardJson = { path: r.path, overrides: r.overrides, has_formula: r.has_formula }; $("rwJsonHint").textContent = "JSON: " + r.path.split(/[\\/]/).pop(); renderRewardSliders(); toast("Saved " + r.path, "ok"); };
  $("acLoad").onclick = async () => { const r = await api().load_action_json(); if (!r.ok) { if (!r.cancelled) toast(r.error, "err"); return; }
    $("action_profile").value = r.base_profile; actionJson = { path: r.path, params: r.params };
    const prof = CAT.action_profiles.find(p => p.key === r.base_profile); actionValues = Object.assign({}, prof.params, r.params);
    $("acJsonHint").textContent = "JSON: " + r.path.split(/[\\/]/).pop(); renderActionSliders(); toast(`Loaded ${r.name}`, "ok"); };
  $("acSave").onclick = async () => { const r = await api().save_action_json(form()); if (!r.ok) { if (!r.cancelled) { toast(r.error, "err"); markBad(r.field); } return; }
    actionJson = { path: r.path, params: r.params }; $("acJsonHint").textContent = "JSON: " + r.path.split(/[\\/]/).pop(); toast("Saved " + r.path, "ok"); };
  $("browse").onclick = async () => { const p = await api().browse_csv(); if (p) { $("csv").insertAdjacentHTML("afterbegin", `<option value="${p}">${p.split(/[\\/]/).pop()} (browsed)</option>`); $("csv").value = p; refreshCsv(); } };
  $("attach").onclick = async () => { const r = await api().attach_params($("csv").value); if (r.ok) { toast("Attached " + r.path, "ok"); refreshCsv(); } else if (!r.cancelled) toast(r.error, "err"); };
  $("start").onclick = startTraining;
  $("stop").onclick = async () => { logLine("Stopping…", "warn"); await api().stop_run(); };
  $("openViz").onclick = async () => { const r = await api().open_visual($("name").value, $("viz_mode").value); if (!r.ok) toast(r.error, "err"); };
  document.querySelectorAll("#side a").forEach(a => a.onclick = () => {
    const p = a.dataset.p; document.querySelectorAll("#side a").forEach(x => x.classList.toggle("on", x === a));
    $("pgTitle").textContent = PAGES[p];
    document.querySelectorAll(".page").forEach(s => s.classList.remove("on"));
    const live = window.rlpages.live;
    $("pills").classList.toggle("off", p !== "train" && !busy);   // training pills: Train page, or while a run is live
    if (p in live) { $("p-" + p).classList.add("on"); $("pgSub").textContent = live[p]; window.rlpages.show(p).catch(e => toast(String(e), "err")); }
    else { $("p-classic").classList.add("on"); $("clHdr").textContent = PAGES[p].toUpperCase(); $("pgSub").textContent = "not migrated yet — classic app"; }
  });
  $("openClassic").onclick = () => api().open_classic();
  await window.rlpages.init();
  if (window.rlbacktest) await window.rlbacktest.init();
  setBusy(await api().busy());
  window.__rlweb_ready = true;
}

(function waitBridge() { if (_raw() && _raw().catalog) init().catch(e => toast("init failed: " + e, "err")); else setTimeout(waitBridge, 100); })();
