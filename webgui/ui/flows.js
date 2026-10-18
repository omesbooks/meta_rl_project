// Pipeline, Walk-Forward and Fine-tune pages (phase 4). Commands, validation, the backtest
// range and the walk-forward verdict all come from Python (webgui/pipeline.py, pipeline_shared.py).
"use strict";
const FL = { cat: null, loaded: false, pipeModel: "", pipeStages: [], pipeTotal: 0, pipeRewards: [], wfExtras: {}, wfName: "" };
const PIPE_KEYS = ["model_name", "train_pct", "steps", "window", "conf", "bt_start", "risk", "max_pos", "sl", "tp",
  "reward_overrides", "reward_formula", "action_params"];
const WF_KEYS = ["windows", "steps", "window", "max_hold", "corr", "name"];
const FT_KEYS = ["mix", "lr", "steps", "name"];

// errors from Python name a field; map it to the element id of the page that owns it
function flowBad(prefix, field) {
  if (!field) return;
  const id = field.startsWith(prefix) ? field : prefix + field;
  if ($(id)) markBad(id);
  if (id === "pipe_hparams" || /^pipe_(lr|clip|ent|nsteps|nepochs|batch|gamma|gae|vf|max_hold|ep_len|net_arch|reward_overrides|reward_formula|action_params)$/.test(id)) $("pipeAdv").open = true;
}
function profileOptions(sel, list, value) { setOptions(sel, list.map(p => [p.key, esc(p.label)]), value); }
function hpCards(wrapId, prefix, fields) {
  $(wrapId).innerHTML = fields.map(([k, label, def, hint]) =>
    `<div class="sl"><div class="t"><span>${esc(label)}</span><input id="${prefix}${k}" value="${esc(def)}"></div>${hint ? `<div class="d">${esc(hint)}</div>` : ""}</div>`).join("");
}
function showReview(r, okLabel, onConfirm, onCancel) {
  $("mSummary").innerHTML = r.summary.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
  $("mWarn").innerHTML = (r.existing ? `<div class="warnbox">⚠ <b>${esc(r.model_name || "")}</b> already exists at ${esc(r.existing)}.<br>The new run replaces the active version only after it succeeds; previous files are kept.</div>` : "") +
    (r.warnings || []).map(w => `<div class="warnbox">${esc(w)}</div>`).join("");
  $("mOk").textContent = r.existing ? "Overwrite & " + okLabel.toLowerCase() : okLabel;
  $("mCmd").textContent = r.command + "\n\nrecipe sha256: " + r.recipe_sha256;
  $("mRecipe").textContent = JSON.stringify(r.recipe, null, 2);
  $("modal").classList.add("on"); setTimeout(() => $("mOk").focus(), 0);
  $("mOk").onclick = () => { $("modal").classList.remove("on"); onConfirm(); };
  $("mCancel").onclick = () => { $("modal").classList.remove("on"); onCancel && onCancel(); };
}

// ------------------------------------------------------------------ pipeline
function pipeForm() {
  const f = { csv: $("pipe_csv").value, bt_csv: $("pipe_bt_csv").value, use_m1: $("pipe_use_m1").checked,
    bt_mode: $("pipe_bt_mode").value, reward_profile: $("pipe_reward_profile").value, action_profile: $("pipe_action_profile").value,
    reward_mode: $("pipe_reward_mode").value };
  for (const k of PIPE_KEYS) f[k] = $("pipe_" + k).value;
  for (const [k] of FL.cat.pipe_hp) f[k] = $("pipe_" + k).value;
  return f;
}
async function pipeCsvInfo(follow) {
  const csv = $("pipe_csv").value;
  const r = await api().pipeline_csv_info(csv, $("pipe_train_pct").value, $("pipe_window").value);
  $("pipe_split").textContent = r.split || r.error || ""; $("pipe_split").className = "hint " + (r.level || "");
  if (!r.ok) { $("pipe_data_hint").textContent = r.error || "Rows: - | Suggested steps: -"; $("pipe_data_hint").className = "hint " + (r.error ? "err" : ""); $("pipe_use_steps").disabled = true; return; }
  if (r.suggest) {
    $("pipe_data_hint").textContent = `Rows: ${r.rows.toLocaleString()} | Train bars: ${r.suggest.train_bars.toLocaleString()} | Suggested steps: ${r.suggest.steps.toLocaleString()} (~${r.suggest.passes} passes)`;
    $("pipe_data_hint").className = "hint ok"; $("pipe_use_steps").disabled = false; $("pipe_use_steps").onclick = () => $("pipe_steps").value = r.suggest.steps;
  } else { $("pipe_data_hint").textContent = `Rows: ${r.rows.toLocaleString()} | Suggested steps: -`; $("pipe_data_hint").className = "hint"; $("pipe_use_steps").disabled = true; }
  if (follow && r.bt_csv && $("pipe_csv").value === csv) { ensureOption($("pipe_bt_csv"), r.bt_csv, r.bt_csv.split(/[\/]/).pop()); $("pipe_bt_csv").value = r.bt_csv; }
  pipeM1();
}
const pipeCsvSoon = debounce(() => pipeCsvInfo(false), 300);
let pipeM1Pref = true;
async function pipeM1() {
  const bt = $("pipe_bt_csv").value || $("pipe_csv").value;
  const r = await api().pipeline_m1(bt);
  const box = $("pipe_use_m1");
  if (r.m1) { $("pipe_m1_hint").textContent = `M1 replay: ${r.m1} (auto-detect)`; box.disabled = false; box.checked = pipeM1Pref; }
  else { $("pipe_m1_hint").textContent = "M1 replay: ไม่พบไฟล์ _m1 คู่กัน — ใช้ intrabar assumption"; box.checked = false; box.disabled = true; }
}
function pipeHealthShell() {
  $("pipe_health").innerHTML = Object.keys(CAT.metrics).map(n => `<div class="kpi" id="pk_${n}"><div class="k">${CAT.metrics[n].label.toUpperCase()}</div><div class="v">—</div><div class="u">good ${CAT.metrics[n].good.map(v => v == null ? "∞" : v).join(" … ")}</div></div>`).join("");
  FL.pipeRewards = []; pipeTrend();
}
function pipeTrend() {
  const r = FL.pipeRewards, w = 600, h = 140, pad = 10, n = r.length;
  if (n < 2) { $("pipe_trend").innerHTML = `<text x="10" y="70" fill="#5b6b85" font-size="13">waiting for PPO training metrics …</text>`; return; }
  const lo = Math.min(...r), hi = Math.max(...r), span = hi - lo || 1;
  const X = i => pad + i * (w - 2 * pad) / (n - 1), Y = v => h - pad - (v - lo) / span * (h - 2 * pad - 14);
  $("pipe_trend").innerHTML = (lo < 0 && hi > 0 ? `<line x1="${pad}" x2="${w - pad}" y1="${Y(0)}" y2="${Y(0)}" stroke="#5b6b85" stroke-dasharray="3 3"/>` : "") +
    `<polyline fill="none" stroke="var(--acc)" stroke-width="2.4" points="${r.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")}"/>` +
    `<text x="${w - pad}" y="14" text-anchor="end" fill="#0a66ff" font-size="13" font-weight="600">${r[n - 1].toFixed(3)}</text><text x="${pad}" y="14" fill="#5b6b85" font-size="11">${n} updates</text>`;
}
function pipeStage(idx, state) {   // state: on | done | failed
  [...$("pipe_stages").children].forEach((el, i) => {
    el.className = state === "failed" && i === idx ? "failed" : i < idx || (state === "done" && i === idx) ? "done" : i === idx ? "on" : "";
  });
}
function onPipelineEvent(ev) {
  const total = FL.pipeStages.length || 3;
  if (ev.type === "start") {
    setBusy(true, "pipeline"); FL.pipeModel = ev.model; FL.pipeStages = ev.stages; FL.pipeTotal = ev.total; FL.stageIdx = 0;
    $("pipe_log").innerHTML = ""; pipeHealthShell(); pipeStage(-1); $("pipe_bar").style.width = "0";
    $("pipe_status").textContent = "Starting pipeline…"; $("pipe_open_report").disabled = $("pipe_open_chart").disabled = true;
  } else if (ev.type === "stage") {
    FL.stageIdx = ev.index; pipeStage(ev.index, "on");
    logLine("", "", "pipe_log"); logLine(`=== ${ev.name} ===`, "metric", "pipe_log"); logLine("$ " + ev.command, "cmd", "pipe_log");
    $("pipe_bar").style.width = (100 * ev.index / total).toFixed(1) + "%"; $("pipe_status").textContent = `${ev.name} started`;
  } else if (ev.type === "stage_done") {
    pipeStage(ev.index + 1, "on"); logLine(`${ev.name} done.`, "success", "pipe_log");
    $("pipe_bar").style.width = (100 * (ev.index + 1) / total).toFixed(1) + "%"; $("pipe_status").textContent = `${ev.name} done`;
  } else if (ev.type === "line") {
    logLine(ev.line, ev.cls, "pipe_log");
    if (ev.progress) {
      const p = Math.min(1, ev.progress.step / ev.progress.total), i = FL.stageIdx || 0;
      $("pipe_bar").style.width = (100 * (i + p) / total).toFixed(1) + "%";
      $("pipe_status").textContent = `${FL.pipeStages[i] || "stage"}: ${ev.progress.step.toLocaleString()} / ${ev.progress.total.toLocaleString()} (${(100 * p).toFixed(1)}%)`;
    }
    if (ev.metric) {
      const m = ev.metric, k = $("pk_" + m.name);
      if (k) { k.className = "kpi " + (m.status || ""); k.querySelector(".v").textContent = fmtNum(m.value, Math.abs(m.value) < 0.1 ? 4 : 3); }
      if (m.name === "ep_rew_mean") { FL.pipeRewards.push(m.value); pipeTrend(); }
    }
  } else if (ev.type === "done") {
    setBusy(false);
    const ok = ev.rc === 0 && !ev.cancelled;
    if (ok) { pipeStage(total - 1, "done"); [...$("pipe_stages").children].forEach(e => e.className = "done"); $("pipe_bar").style.width = "100%"; }
    else pipeStage(Math.max(0, FL.pipeStages.indexOf(ev.failed_stage)), "failed");
    const msg = ok ? "Pipeline complete." : ev.cancelled ? "Pipeline stopped." : `${ev.failed_stage || "stage"} failed with exit code ${ev.rc}.`;
    $("pipe_status").textContent = ok ? "✓ " + msg : "Pipeline stopped/failed — " + msg;
    logLine(msg, ok ? "success" : "error", "pipe_log");
    $("pipe_open_report").disabled = $("pipe_open_chart").disabled = !ok;
    if (ok && typeof MD !== "undefined") MD.loaded = false;   // Models table refreshes on next visit
  }
}
async function runPipeline() {
  clearBad(); $("pipe_run").disabled = true; $("pipe_status").textContent = "Checking training data…";
  let r; try { r = await api().review_pipeline(pipeForm()); } catch (e) { r = { ok: false, error: String(e) }; }
  $("pipe_run").disabled = busy;
  if (!r.ok) { $("pipe_status").textContent = "Review failed"; toast(r.error, "err"); flowBad("pipe_", r.field); return; }
  $("pipe_model_name").value = r.model_name;                  // the CTk page shows the cleaned name too
  $("pipe_status").textContent = "Awaiting confirmation";
  showReview(r, "Start pipeline", async () => { const c = await api().confirm_pipeline(r.token); if (!c.ok) toast(c.error, "err"); },
    () => $("pipe_status").textContent = "Idle");
}

// ------------------------------------------------------------------ walk-forward
function wfForm() {
  const f = { csv: $("wf_csv").value, action_profile: $("wf_action_profile").value, reward_mode: $("wf_reward_mode").value,
    reward_profile: $("wf_reward_profile").value, ...FL.wfExtras };
  for (const k of WF_KEYS) f[k] = $("wf_" + k).value;
  for (const [k] of FL.cat.wf_ppo) f[k] = $("wf_" + k).value;
  return f;
}
function wfRecipeChanged(kind) {   // CTk _wf_recipe_changed: a manual profile change drops extras copied for the old one
  const changed = [], x = FL.wfExtras;
  if (kind === "action" && (x.action_params || x.action_profile_json)) { x.action_params = ""; x.action_profile_json = ""; changed.push("action params/JSON"); }
  if (kind === "reward" && (x.reward_overrides || x.reward_formula)) { x.reward_overrides = ""; x.reward_formula = ""; changed.push("reward overrides/formula"); }
  if (changed.length) $("wf_recipe_hint").textContent = `เปลี่ยน profile ด้วยมือ → ล้าง ${changed.join(" + ")} ที่ copy มา (กด ⧉ Copy settings from Train ใหม่ถ้าต้องการ)`;
}
async function wfCopy() {
  const r = await api().wf_from_train(form());
  if (!r.ok) { toast(r.error, "err"); return; }
  for (const k of ["window", "max_hold", "corr"]) $("wf_" + k).value = r[k];
  $("wf_reward_mode").value = r.reward_mode; $("wf_reward_profile").value = r.reward_profile; $("wf_action_profile").value = r.action_profile;
  for (const [k] of FL.cat.wf_ppo) $("wf_" + k).value = r[k];
  FL.wfExtras = { reward_overrides: r.reward_overrides, reward_formula: r.reward_formula, action_params: r.action_params, action_profile_json: r.action_profile_json };
  $("wf_recipe_hint").textContent = r.hint;
  logLine("Copied recipe from Train page: window/max hold/correlation threshold/reward/action/PPO (steps + output name ไม่ถูก copy — เป็นค่าเฉพาะของ Walk-Forward)", "info", "wf_log");
}
function wfVerdict(v) {
  $("wf_verdict").className = "verdict " + (v ? v.tone : "");
  $("wf_verdict").innerHTML = `<div class="ic">${v ? v.icon : "—"}</div><div><b>${esc(v ? v.title : "Running...")}</b><span>${esc(v ? v.sub : "")}</span></div>`;
}
function onWalkforwardEvent(ev) {
  if (ev.type === "start") {
    setBusy(true, "walk-forward"); FL.wfName = ev.name; logLine("$ " + ev.command, "cmd", "wf_log");
    $("wf_bar").style.width = "0"; $("wf_prog_label").textContent = "Progress: 0.0% — starting"; wfVerdict(null); $("wf_chart").innerHTML = "";
  } else if (ev.type === "line") {
    logLine(ev.line, ev.cls, "wf_log");
    if (ev.wf && ev.wf.progress) {
      const [cur, tot, label] = ev.wf.progress, p = tot > 0 ? Math.min(Math.max(cur / tot, 0), 1) : 0;
      $("wf_bar").style.width = (100 * p).toFixed(1) + "%"; $("wf_prog_label").textContent = `Progress: ${(100 * p).toFixed(1)}%${label ? " — " + label : ""}`;
    }
    if (ev.wf && ev.wf.verdict) wfVerdict(ev.wf.verdict);
  } else if (ev.type === "done") {
    setBusy(false);
    const ok = ev.rc === 0;
    logLine(ev.cancelled ? "[stop] cancelled" : ok ? "✓ Done" : `✗ Failed (exit ${ev.rc})`, ok ? "success" : "error", "wf_log");
    if (ok) { $("wf_bar").style.width = "100%"; $("wf_prog_label").textContent = "Progress: 100.0% — complete"; wfChart(ev.name); }
    else $("wf_prog_label").textContent = "Progress: stopped/failed";
  }
}
async function wfChart(name) {
  const r = await api().wf_chart(name || $("wf_name").value);
  $("wf_chart").innerHTML = r.ok ? `<img src="${r.url}" alt="${esc(r.name)}">` : `<div class="empty">${esc(r.error)}</div>`;
}

// ------------------------------------------------------------------ fine-tune
function ftForm() {
  const f = { base: $("ft_base").value, old_csv: $("ft_old_csv").value, new_csv: $("ft_new_csv").value, mode: $("ft_mode").value };
  for (const k of FT_KEYS) f[k] = $("ft_" + k).value;
  return f;
}
async function ftRecipe() { const r = await api().ft_recipe($("ft_base").value); $("ft_recipe").textContent = r.text; $("ft_recipe").className = "hint " + (r.level || ""); }
function onFinetuneEvent(ev) {
  if (ev.type === "start") { setBusy(true, "fine-tuning"); logLine("$ " + ev.command, "cmd", "ft_log"); $("ft_status").textContent = `Fine-tuning → ${ev.name}…`; $("ft_bar").style.width = "0"; }
  else if (ev.type === "line") {
    logLine(ev.line, ev.cls, "ft_log");
    if (ev.progress) { const p = Math.min(1, ev.progress.step / ev.progress.total); $("ft_bar").style.width = (100 * p).toFixed(1) + "%"; $("ft_status").textContent = `Step ${ev.progress.step.toLocaleString()} / ${ev.progress.total.toLocaleString()} (${(100 * p).toFixed(1)}%)`; }
  } else if (ev.type === "done") {
    setBusy(false);
    const ok = ev.rc === 0;
    $("ft_status").textContent = ev.cancelled ? "Stopped" : ok ? "✓ Fine-tune complete" : `✗ Failed (exit ${ev.rc})`;
    logLine(ev.cancelled ? "[stop] cancelled" : ok ? "✓ Done" : `✗ Failed (exit ${ev.rc})`, ok ? "success" : "error", "ft_log");
    if (ok) { $("ft_bar").style.width = "100%"; if (typeof MD !== "undefined") MD.loaded = false; }
  }
}

// ------------------------------------------------------------------ wiring
window.rlflows = {
  async init() {
    Object.assign(window.rlweb.handlers, { pipeline: onPipelineEvent, walkfwd: onWalkforwardEvent, finetune: onFinetuneEvent });
    Object.assign(window.rlpages.live, { pipeline: "Train PPO → Backtest → Chart, each stage gated on exit code",
      walkfwd: "rl_walkforward.py · retrain per window, verdict from the OOS windows", finetune: "rl_finetune.py · base model + old/new data" });
    const stop = async (log) => { logLine("Stopping…", "warn", log); await api().stop_run(); };
    $("pipe_run").onclick = runPipeline; $("pipe_stop").onclick = () => stop("pipe_log");
    $("pipe_csv").onchange = () => pipeCsvInfo(true); $("pipe_train_pct").oninput = pipeCsvSoon; $("pipe_window").oninput = pipeCsvSoon;
    $("pipe_bt_csv").onchange = pipeM1; $("pipe_use_m1").onchange = () => pipeM1Pref = $("pipe_use_m1").checked;
    $("pipe_open_report").onclick = async () => {
      document.querySelector("#side a[data-p=backtest]").click(); await window.rlbacktest.show();
      const sel = $("bt_model"); if (![...sel.options].some(o => o.value === FL.pipeModel)) sel.insertAdjacentHTML("beforeend", `<option value="${esc(FL.pipeModel)}">${esc(FL.pipeModel)}</option>`);
      sel.value = FL.pipeModel; sel.onchange();
    };
    $("pipe_open_chart").onclick = async () => { const r = await api().open_chart(FL.pipeModel); if (!r.ok) toast(r.error, "err"); };
    document.querySelectorAll("[data-pipe-preset]").forEach(b => b.onclick = () => { const p = CAT.presets[b.dataset.pipePreset]; for (const k of ["lr", "clip", "ent", "nsteps", "nepochs", "batch", "gamma", "gae", "vf"]) $("pipe_" + k).value = p[k]; toast(`preset: ${b.textContent.trim()}`, "ok"); });
    $("wf_run").onclick = async () => { clearBad(); const r = await api().start_walkforward(wfForm()); if (!r.ok) { toast(r.error, "err"); flowBad("wf_", r.field); } };
    $("wf_stop").onclick = () => stop("wf_log"); $("wf_copy").onclick = wfCopy;
    $("wf_action_profile").onchange = () => wfRecipeChanged("action");
    $("wf_reward_profile").onchange = $("wf_reward_mode").onchange = () => wfRecipeChanged("reward");
    $("ft_run").onclick = async () => { clearBad(); const r = await api().start_finetune(ftForm()); if (!r.ok) { toast(r.error, "err"); flowBad("ft_", r.field); } };
    $("ft_stop").onclick = () => stop("ft_log"); $("ft_base").onchange = ftRecipe;
    $("ft_mode").onchange = () => $("ft_mode_hint").textContent = FL.cat.ft_mode_hints[$("ft_mode").value] || "";
  },
  async show(page) {
    if (!FL.loaded) {
      const c = FL.cat = await api().flow_catalog(), csvs = c.csvs.map(x => [esc(x), esc(x)]), none = [["", "(none)"]];
      const pd = c.pipeline, wd = c.wf, fd = c.ft;
      setOptions($("pipe_csv"), csvs.length ? csvs : none); setOptions($("pipe_bt_csv"), csvs.length ? csvs : none);
      profileOptions($("pipe_reward_profile"), CAT.reward_profiles, pd.reward_profile);
      profileOptions($("pipe_action_profile"), CAT.action_profiles, pd.action_profile);
      for (const k of PIPE_KEYS) $("pipe_" + k).value = pd[k];
      $("pipe_bt_mode").value = pd.bt_mode; $("pipe_reward_mode").value = pd.reward_mode;
      hpCards("pipeHp", "pipe_", c.pipe_hp.map(([k, label, def, hint]) => [k, label, def, hint]));
      pipeHealthShell();
      setOptions($("wf_csv"), csvs.length ? csvs : none);
      profileOptions($("wf_action_profile"), CAT.action_profiles, wd.action_profile);
      profileOptions($("wf_reward_profile"), CAT.reward_profiles, wd.reward_profile);
      for (const k of WF_KEYS) $("wf_" + k).value = wd[k];
      $("wf_reward_mode").value = wd.reward_mode;
      hpCards("wfPpo", "wf_", c.wf_ppo.map(([k, label, def]) => [k, label, def, ""]));
      const models = c.models.map(m => [esc(m), esc(m)]);
      setOptions($("ft_base"), [["", "(none)"], ...models]); setOptions($("ft_old_csv"), [["", "(none)"], ...csvs]); setOptions($("ft_new_csv"), [["", "(none)"], ...csvs]);
      setOptions($("ft_mode"), Object.entries(c.ft_modes), fd.mode);
      for (const k of FT_KEYS) $("ft_" + k).value = fd[k];
      $("ft_mode_hint").textContent = c.ft_mode_hints[fd.mode];
      FL.loaded = true;
      pipeCsvInfo(true);
    }
  },
};
