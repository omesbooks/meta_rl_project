// Models + Settings pages (phase 2). Same rule as app.js: Python decides, the page paints.
"use strict";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ appearance (theme, accent, brand)
let SETTINGS = null;
function applyAppearance(st) {
  if (!st) return;
  document.body.classList.toggle("deep", st.web.theme === "deep");
  const acc = st.web.brand_accent ? st.colors.ACCENT : "";
  if (acc) document.documentElement.style.setProperty("--acc", acc); else document.documentElement.style.removeProperty("--acc");
  const b = st.branding;
  $("brandText").textContent = (b.logo_text || "RL Trading Studio").toUpperCase();
  $("brandSub").textContent = b.subtitle || "";
  $("brandLogo").innerHTML = st.logo ? `<img src="${st.logo}" alt="">` : `<span>${esc(b.logo_emoji || "🤖")}</span>`;
  if (b.window_title) document.title = b.window_title;
}

// ------------------------------------------------------------------ models
const MD = { rows: [], sel: null, sortCol: null, sortDir: -1, loaded: false };
function mdRowClass(r) { const v = r.sort[5]; return v == null ? "" : v >= 0 ? "pos" : "neg"; }
function renderModels() {
  const q = $("mdFilter").value.trim().toLowerCase();
  let rows = MD.rows.filter(r => !q || r.model.toLowerCase().includes(q) || r.source.toLowerCase().includes(q));
  if (MD.sortCol != null) {
    const c = MD.sortCol, d = MD.sortDir;
    const key = r => (c === 4 && r.pf_inf) ? Infinity : r.sort[c] == null ? (c === 9 ? r.cells[9] : null) : r.sort[c];
    rows = rows.slice().sort((a, b) => { const x = key(a), y = key(b); if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
      return (typeof x === "string" ? x.localeCompare(y) : x - y) * d; });
  }
  $("mdCount").textContent = `${rows.length} of ${MD.rows.length} results`;
  $("mdTable").querySelector("tbody").innerHTML = rows.map(r =>
    `<tr data-iid="${esc(r.iid)}" class="${r.iid === MD.sel ? "on" : ""}">` + r.cells.map((v, i) =>
      `<td class="${i === 5 || i === 6 ? mdRowClass(r) + (i === 6 ? " dd" : "") : ""}">${esc(v)}</td>`).join("") + "</tr>").join("") ||
    `<tr><td colspan="10" class="empty">No model results${q ? " match the filter" : " found — train one first"}.</td></tr>`;
  $("mdTable").querySelectorAll("tbody tr[data-iid]").forEach(tr => tr.onclick = () => selectModel(tr.dataset.iid));
}
async function loadModels() {
  $("mdCount").textContent = "loading …";
  const r = await api().list_models();
  MD.loaded = true; MD.rows = r.rows;
  $("mdTable").querySelector("thead").innerHTML = "<tr>" + r.headings.map((h, i) =>
    `<th data-c="${i}">${esc(h)}${MD.sortCol === i ? (MD.sortDir > 0 ? " ▲" : " ▼") : ""}</th>`).join("") + "</tr>";
  $("mdTable").querySelectorAll("th").forEach(th => th.onclick = () => {
    const c = +th.dataset.c; MD.sortDir = MD.sortCol === c ? -MD.sortDir : (c <= 1 || c === 7 ? 1 : -1); MD.sortCol = c;
    $("mdTable").querySelectorAll("th").forEach(t => t.textContent = t.textContent.replace(/ [▲▼]$/, ""));
    th.textContent += MD.sortDir > 0 ? " ▲" : " ▼"; renderModels();
  });
  if (!MD.rows.some(x => x.iid === MD.sel)) MD.sel = MD.rows.length ? MD.rows[0].iid : null;
  renderModels();
  if (MD.sel) selectModel(MD.sel); else { $("eqName").textContent = "–"; $("eqBox").innerHTML = `<div class="empty">No model results found yet.</div>`; }
}
async function selectModel(iid) {
  MD.sel = iid;
  $("mdTable").querySelectorAll("tbody tr").forEach(tr => tr.classList.toggle("on", tr.dataset.iid === iid));
  const row = MD.rows.find(x => x.iid === iid); if (!row) return;
  $("eqName").textContent = `${row.model} · ${row.source}`;
  $("mdChart").disabled = !row.has_chart;
  $("eqBox").innerHTML = `<div class="empty">loading equity curve …</div>`;
  const r = await api().model_equity(iid);
  if (MD.sel !== iid) return;                                   // user clicked another row meanwhile
  $("eqBox").innerHTML = r.ok ? `<img src="${r.url}" alt="${esc(r.name)}" title="${esc(r.name)}">` : `<div class="empty">${esc(r.error)}</div>`;
}
function initModels() {
  $("mdRefresh").onclick = loadModels;
  $("mdFilter").oninput = renderModels;
  $("mdEquity").onclick = async () => { if (!MD.sel) return toast("Select a model result first.", "err"); const r = await api().open_model_file(MD.sel, "equity"); if (!r.ok) toast(r.error, "err"); };
  $("mdChart").onclick = async () => { if (!MD.sel) return toast("Select a model result first.", "err"); const r = await api().open_model_file(MD.sel, "chart"); if (!r.ok) toast(r.error, "err"); };
  $("mdPdf").onclick = async () => { $("mdPdf").disabled = true; toast("Building PDF …"); const r = await api().export_models_pdf(MD.sel || ""); $("mdPdf").disabled = false; toast(r.ok ? "Saved report: " + r.path : r.error, r.ok ? "ok" : "err"); };
}

// ------------------------------------------------------------------ settings
function hex6(v) { v = String(v || "").trim(); return /^#[0-9a-f]{3}$/i.test(v) ? "#" + v.slice(1).split("").map(c => c + c).join("") : v; }
function renderSettings(st) {
  SETTINGS = st;
  setOptions($("webTheme"), Object.entries(st.web_themes), st.web.theme);
  $("webBrandAccent").checked = !!st.web.brand_accent;
  setOptions($("brPreset"), [["", "— choose a preset —"], ...Object.keys(st.presets).map(k => [esc(k), esc(k)])], "");
  $("brText").innerHTML = st.text_fields.map(([k, label, hint]) =>
    `<div class="fld"><label for="br_${k}">${esc(label)}</label><input id="br_${k}" value="${esc(st.branding[k])}" placeholder="${esc(hint)}"><span class="hint">${esc(hint)}</span></div>`).join("");
  $("brColors").innerHTML = st.color_fields.map(([k, label, hint]) =>
    `<div class="fld"><label for="color_${k}">${esc(label)}</label><div class="inl"><input type="color" id="pick_${k}" value="${hex6(st.colors[k])}"><input id="color_${k}" class="mono" value="${esc(st.colors[k])}"></div><span class="hint">${esc(hint)}</span></div>`).join("");
  for (const [k] of st.color_fields) {
    $("pick_" + k).oninput = () => { $("color_" + k).value = $("pick_" + k).value; $("color_" + k).classList.remove("bad"); };
    $("color_" + k).oninput = () => { const v = hex6($("color_" + k).value); if (/^#[0-9a-f]{6}$/i.test(v)) $("pick_" + k).value = v; };
  }
  keyStatus(st.keys);
  $("setWork").textContent = st.work_dir; $("setCfg").textContent = st.config_path + (st.saved ? "" : "  (not created yet — defaults)"); $("setPy").textContent = st.python;
  $("brStatus").textContent = st.saved ? "" : "using built-in defaults";
  applyAppearance(st);
}
function keyStatus(keys) {
  const g = keys.gemini;
  $("gemStatus").textContent = g.set ? `✓ key saved${g.tail ? " (…" + g.tail + ")" : ""}` : "no key set";
  $("gemStatus").className = "hint " + (g.set ? "ok" : "");
  $("gemClear").disabled = !g.set;
}
function brandingForm() {
  const f = { branding: {}, colors: {} };
  for (const [k] of SETTINGS.text_fields) f.branding[k] = $("br_" + k).value;
  for (const [k] of SETTINGS.color_fields) f.colors[k] = $("color_" + k).value;
  return f;
}
function initSettings() {
  $("webTheme").onchange = $("webBrandAccent").onchange = async () => {
    const r = await api().save_web_settings({ theme: $("webTheme").value, brand_accent: $("webBrandAccent").checked });
    SETTINGS.web = r.web; applyAppearance(SETTINGS);
  };
  $("brPreset").onchange = () => {
    const p = SETTINGS.presets[$("brPreset").value]; if (!p) return;
    for (const [k] of SETTINGS.color_fields) if (p[k]) { $("color_" + k).value = p[k]; $("pick_" + k).value = hex6(p[k]); }
    toast(`preset “${$("brPreset").value}” loaded — press Save to keep it`);
  };
  $("brSave").onclick = async () => {
    clearBad(); const r = await api().save_branding(brandingForm());
    if (!r.ok) { toast(r.error, "err"); markBad(r.field); return; }
    renderSettings(r); toast("✓ Branding saved — the classic app applies it after a restart", "ok");
  };
  $("brReset").onclick = async () => {
    if (!confirm("Delete branding_config.json and go back to the default branding?")) return;
    renderSettings(await api().reset_branding()); toast("✓ Branding reset to default", "ok");
  };
  $("brOpen").onclick = async () => { const r = await api().open_branding_config(); if (!r.ok) toast(r.error, "err"); };
  $("gemShow").onchange = () => $("gemini_key").type = $("gemShow").checked ? "text" : "password";
  $("gemSave").onclick = async () => {
    clearBad(); const r = await api().save_api_key($("gemini_key").value);
    if (!r.ok) { toast(r.error, "err"); markBad(r.field); return; }
    $("gemini_key").value = ""; keyStatus(r.keys); toast("✓ Gemini API key saved", "ok");
  };
  $("gemClear").onclick = async () => {
    if (!confirm("Remove the saved Gemini API key?")) return;
    const r = await api().clear_api_key(); keyStatus(r.keys); toast("Gemini API key cleared", "ok");
  };
  $("gemGet").onclick = () => api().open_gemini_page();
}

// ------------------------------------------------------------------ router hooks (called by app.js)
window.rlpages = {
  live: { train: "PPO · TradingEnv · rl_train.py", backtest: "backtest_live.py · stored report from the meta JSON", models: "results of every model · same table as the classic app", settings: "branding · API keys · web appearance" },
  async show(p) { if (p === "models" && !MD.loaded) await loadModels(); if (p === "backtest" && window.rlbacktest) await window.rlbacktest.show();
    if (["pipeline", "walkfwd", "finetune"].includes(p) && window.rlflows) await window.rlflows.show(p); },
  async init() {
    initModels(); initSettings();
    renderSettings(await api().settings_get());
  },
};
