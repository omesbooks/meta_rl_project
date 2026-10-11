// Dashboard display mode for the live visual (tools/viz/network3d.html).
//
// Event-driven: everything is drawn once per new snapshot / timeline pick /
// resize — no animation loop, so the window sits at ~0% CPU between updates.
// Panels: KPIs · learning curve · equity of every evaluation · action mix ·
// replay of the selected evaluation · neuron activity heatmaps · last trades.

const CSS = `
  #dash { position: fixed; inset: 64px 0 0 0; overflow: auto; padding: 8px 18px 84px; display: none; }
  body.dash #dash { display: block; }
  .dgrid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .dcard { background: rgba(8, 13, 28, .78); border: 1px solid rgba(25, 230, 255, .14); border-radius: 12px; padding: 10px 12px 12px; min-width: 0; }
  .dcard h3 { margin: 0 0 8px; font: 600 10px/1 var(--hud); letter-spacing: .28em; color: var(--dim); }
  .dcard h3 b { color: var(--fg); font-weight: 600; letter-spacing: .12em; }
  .span2 { grid-column: span 2; } .span3 { grid-column: span 3; }
  .kpis { grid-column: span 3; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; }
  .kpi { background: rgba(8, 13, 28, .78); border: 1px solid rgba(25, 230, 255, .14); border-radius: 10px; padding: 9px 12px; }
  .kpi .k { font: 600 9px/1 var(--hud); letter-spacing: .24em; color: var(--dim); }
  .kpi .v { font: 600 20px/1.3 var(--hud); font-variant-numeric: tabular-nums; text-shadow: 0 0 8px currentColor; }
  .dcard svg { display: block; width: 100%; overflow: visible; }
  .dlegend { display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 6px; font: 600 10px var(--hud); letter-spacing: .12em; }
  .dlegend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
  .heat { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 18px; }
  .heat .blk { min-width: 0; }
  .heat .lab { font: 600 9px/1.6 var(--hud); letter-spacing: .2em; color: var(--dim); }
  .heat canvas { width: 100%; image-rendering: pixelated; display: block; border-radius: 3px; }
  .dtable { width: 100%; border-collapse: collapse; font: 600 12px var(--hud); font-variant-numeric: tabular-nums; }
  .dtable td, .dtable th { padding: 4px 6px; text-align: right; border-bottom: 1px solid rgba(93, 107, 138, .18); }
  .dtable th { font-size: 9px; letter-spacing: .2em; color: var(--dim); font-weight: 600; }
  .dtable td:first-child, .dtable th:first-child { text-align: left; color: var(--dim); }
  .dnote { color: var(--dim); font: 600 11px var(--hud); letter-spacing: .1em; padding: 24px 0; text-align: center; }
  @media (max-width: 1100px) { .dgrid { grid-template-columns: 1fr; } .span2, .span3, .kpis { grid-column: span 1; }
                               .kpis { grid-template-columns: repeat(4, minmax(0, 1fr)); } .heat { grid-template-columns: 1fr; } }
`;

const C = { cyan: "#19e6ff", gold: "#ffc53d", green: "#00ffa3", mag: "#ff2e88", orange: "#ff8a1f", dim: "#5d6b8a", grid: "#0d2a4d" };
const fmt = (v, d = 2) => (v == null || !Number.isFinite(+v)) ? "–" : (+v).toFixed(d);
const pct = v => (v == null || !Number.isFinite(+v)) ? "–" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** Same fields as tools/viz/snapshot.py summary() — for runs written before it existed. */
export function summarize(D, points = 120) {
  const eq = D.eq || [], n = eq.length, tp = D.tpnl || [];
  const curve = [];
  const m = Math.min(points, n);
  for (let j = 0; j < m; j++) curve.push(eq[Math.round(j * (n - 1) / Math.max(1, m - 1))]);
  const counts = new Array((D.actions || []).length).fill(0);
  for (const a of D.act || []) if (a >= 0 && a < counts.length) counts[a]++;
  const closed = tp.filter(p => p !== 0), wins = closed.filter(p => p > 0).length;
  let peak = -Infinity, dd = 0;
  for (const e of eq) { peak = Math.max(peak, e); dd = Math.min(dd, (e - peak) / peak); }
  return { eq_curve: curve, action_counts: counts, trades: closed.length, wins,
           win_rate: closed.length ? wins / closed.length : null, equity_end: n ? eq[n - 1] : 1, max_dd: dd };
}

function closedTrades(D) {
  const out = []; let entry = null;
  for (let i = 0; i < D.frames; i++) {
    const prev = i > 0 ? D.pos[i - 1] : 0;
    if (D.tpnl[i] !== 0) out.push({ i, side: prev > 0 ? "LONG" : prev < 0 ? "SHORT" : "–", pnl: D.tpnl[i], bars: entry != null ? i - entry : null });
    if (D.pos[i] !== prev) entry = D.pos[i] !== 0 ? i : null;
  }
  return out;
}

export function createDashboard(root, { colors, fetchSnapshot }) {
  if (!document.getElementById("dash-css")) {
    const st = document.createElement("style"); st.id = "dash-css"; st.textContent = CSS; document.head.appendChild(st);
  }
  root.innerHTML = `
    <div class="dgrid">
      <div class="kpis" id="dKpi"></div>
      <div class="dcard span2"><h3>LEARNING CURVE · <b>eval reward per evaluation</b></h3><svg id="dLearn" height="190"></svg>
        <div class="dlegend"><span><i style="background:${C.cyan}"></i>EVAL REWARD</span><span><i style="background:${C.gold}"></i>BEST SO FAR</span></div></div>
      <div class="dcard"><h3>ACTION MIX · <b>share of bars per evaluation</b></h3><svg id="dMix" height="190"></svg><div class="dlegend" id="dMixLeg"></div></div>
      <div class="dcard span2"><h3>EQUITY ON THE SLICE · <b>every evaluation (gold = latest)</b></h3><svg id="dEq" height="210"></svg></div>
      <div class="dcard"><h3>LAST TRADES · <b id="dTradeHdr">selected evaluation</b></h3><div id="dTrades"></div></div>
      <div class="dcard span3"><h3>REPLAY · <b id="dReplayHdr">selected evaluation</b></h3><svg id="dReplay" height="230"></svg></div>
      <div class="dcard span3"><h3>NEURON ACTIVITY · <b>neurons × bars (orange +, cyan −)</b></h3><div class="heat" id="dHeat"></div></div>
    </div>`;
  const $ = id => root.querySelector("#" + id);
  const cache = new Map();          // eval i -> summary
  let pending = false, last = null;

  const width = el => Math.max(200, el.clientWidth || el.parentElement.clientWidth - 24);
  const actionColor = (name, i) => (colors && colors[name]) || ["#9fb3d9", "#00ffa3", "#ff2e88", "#ffc53d", "#b48cff", "#19e6ff"][i % 6];

  function summaryOf(entry) {
    if (entry.eq_curve && entry.action_counts) return entry;
    if (cache.has(entry.i)) return Object.assign({}, entry, cache.get(entry.i));
    return null;
  }

  async function fillMissing(index) {
    if (pending || !fetchSnapshot) return;
    const missing = index.filter(e => !summaryOf(e));
    if (!missing.length) return;
    pending = true;
    for (const e of missing) {
      try { cache.set(e.i, summarize(await fetchSnapshot(e))); } catch (err) { cache.set(e.i, null); }
    }
    pending = false;
    if (last) render(last);
  }

  function kpis(sel, D, entry) {
    const s = D ? summarize(D) : null, trades = D ? closedTrades(D) : [];
    const avg = trades.length ? trades.reduce((a, t) => a + t.pnl, 0) / trades.length : null;
    const tile = (k, v, col) => `<div class="kpi"><div class="k">${k}</div><div class="v" style="color:${col}">${v}</div></div>`;
    const m = (D && D.meta) || {};
    const step = entry ? entry.step : m.step ?? m.steps;
    $("dKpi").innerHTML = [
      tile("STEP", step != null ? Number(step).toLocaleString() : "–", "var(--fg)"),
      tile("EVAL REWARD", fmt(entry ? entry.eval_reward : m.eval_reward), C.cyan),
      tile("BEST REWARD", fmt(entry ? entry.best_reward : m.best_reward), C.gold),
      tile("TRADES", s ? s.trades : "–", C.cyan),
      tile("WIN RATE", s && s.win_rate != null ? Math.round(s.win_rate * 100) + "%" : "–", C.gold),
      tile("EQUITY", s ? pct(s.equity_end - 1) : "–", s && s.equity_end >= 1 ? C.green : C.mag),
      tile("MAX DD", s ? pct(s.max_dd) : "–", C.mag),
      tile("AVG TRADE", avg != null ? pct(avg) : "–", avg != null && avg >= 0 ? C.green : C.mag),
    ].join("");
  }

  function learning(index, selI) {
    const svg = $("dLearn"), W = width(svg), H = 190, P = { l: 46, r: 12, t: 10, b: 22 };
    const pts = index.filter(e => Number.isFinite(+e.eval_reward));
    if (pts.length < 1) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" fill="${C.dim}" text-anchor="middle" font-size="11">live runs only — one point per evaluation</text>`; return; }
    let best = -Infinity; const bestLine = pts.map(e => (best = Math.max(best, +e.eval_reward)));
    const xs = pts.map(e => +e.step), ys = pts.map(e => +e.eval_reward).concat(bestLine);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const X = v => P.l + (x1 > x0 ? (v - x0) / (x1 - x0) : 0.5) * (W - P.l - P.r);
    const Y = v => H - P.b - (y1 > y0 ? (v - y0) / (y1 - y0) : 0.5) * (H - P.t - P.b);
    const line = arr => arr.map((v, i) => `${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    let g = "";
    for (let k = 0; k <= 4; k++) { const v = y0 + (y1 - y0) * k / 4, y = Y(v); g += `<line x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" stroke="${C.grid}"/><text x="${P.l - 6}" y="${y + 3}" fill="${C.dim}" font-size="10" text-anchor="end">${fmt(v, 1)}</text>`; }
    if (y0 < 0 && y1 > 0) g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.dim}" stroke-dasharray="3 4"/>`;
    g += `<text x="${P.l}" y="${H - 4}" fill="${C.dim}" font-size="10">${x0.toLocaleString()}</text><text x="${W - P.r}" y="${H - 4}" fill="${C.dim}" font-size="10" text-anchor="end">${x1.toLocaleString()} steps</text>`;
    g += `<polyline points="${line(bestLine)}" fill="none" stroke="${C.gold}" stroke-width="1.4" stroke-dasharray="5 4" opacity=".8"/>`;
    g += `<polyline points="${line(pts.map(e => +e.eval_reward))}" fill="none" stroke="${C.cyan}" stroke-width="2" style="filter:drop-shadow(0 0 4px ${C.cyan})"/>`;
    pts.forEach((e, i) => { const sel = e.i === selI; g += `<circle cx="${X(xs[i])}" cy="${Y(+e.eval_reward)}" r="${sel ? 5 : 2.6}" fill="${sel ? C.gold : C.cyan}" ${sel ? `style="filter:drop-shadow(0 0 5px ${C.gold})"` : ""}/>`; });
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = g;
  }

  function equity(sums, selI, lastI) {
    const svg = $("dEq"), W = width(svg), H = 210, P = { l: 52, r: 12, t: 10, b: 18 };
    const curves = sums.filter(s => s && s.eq_curve && s.eq_curve.length);
    if (!curves.length) { svg.innerHTML = ""; return; }
    let lo = Infinity, hi = -Infinity;
    for (const s of curves) for (const v of s.eq_curve) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    lo = Math.min(lo, 1); hi = Math.max(hi, 1); const pad = (hi - lo) * 0.1 || 0.005; lo -= pad; hi += pad;
    const Y = v => H - P.b - (v - lo) / (hi - lo) * (H - P.t - P.b);
    let g = "";
    for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v); g += `<line x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" stroke="${C.grid}"/><text x="${P.l - 6}" y="${y + 3}" fill="${C.dim}" font-size="10" text-anchor="end">${pct(v - 1)}</text>`; }
    g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(1)}" y2="${Y(1)}" stroke="${C.dim}" stroke-dasharray="3 4"/>`;
    const n = curves.length;
    curves.forEach((s, j) => {
      const c = s.eq_curve, isLast = s.i === lastI, isSel = s.i === selI && !isLast;
      const pts = c.map((v, k) => `${(P.l + k / Math.max(1, c.length - 1) * (W - P.l - P.r)).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
      const op = isLast || isSel ? 1 : 0.12 + 0.45 * (j + 1) / n;
      const col = isLast ? C.gold : C.cyan;
      g += `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="${isLast || isSel ? 2.2 : 1}" opacity="${op}" ${isLast || isSel ? `style="filter:drop-shadow(0 0 4px ${col})"` : ""}/>`;
    });
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = g;
  }

  function mix(sums, names, labels, selI) {
    const svg = $("dMix"), W = width(svg), H = 190, P = { l: 4, r: 4, t: 6, b: 16 };
    const rows = sums.filter(s => s && s.action_counts);
    $("dMixLeg").innerHTML = names.map((nm, a) => `<span><i style="background:${actionColor(nm, a)}"></i>${esc(labels[a] || nm)}</span>`).join("");
    if (!rows.length) { svg.innerHTML = ""; return; }
    const bw = (W - P.l - P.r) / rows.length;
    let g = "";
    rows.forEach((s, j) => {
      const tot = s.action_counts.reduce((a, b) => a + b, 0) || 1; let y = H - P.b;
      s.action_counts.forEach((cnt, a) => {
        const h = cnt / tot * (H - P.t - P.b); y -= h;
        g += `<rect x="${(P.l + j * bw + bw * 0.12).toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, bw * 0.76).toFixed(1)}" height="${h.toFixed(1)}" fill="${actionColor(names[a], a)}" opacity="${s.i === selI ? 1 : 0.62}"/>`;
      });
      if (s.i === selI) g += `<rect x="${(P.l + j * bw + bw * 0.06).toFixed(1)}" y="${P.t - 3}" width="${(bw * 0.88).toFixed(1)}" height="${H - P.t - P.b + 3}" fill="none" stroke="${C.gold}" stroke-width="1.5" rx="2"/>`;
    });
    g += `<text x="${P.l}" y="${H - 3}" fill="${C.dim}" font-size="10">eval 1</text><text x="${W - P.r}" y="${H - 3}" fill="${C.dim}" font-size="10" text-anchor="end">eval ${rows.length}</text>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = g;
  }

  function replay(D) {
    const svg = $("dReplay"), W = width(svg), H = 230, P = { l: 56, r: 56, t: 10, b: 18 };
    const n = D.frames;
    let lo = Infinity, hi = -Infinity, elo = Infinity, ehi = -Infinity;
    for (let i = 0; i < n; i++) { lo = Math.min(lo, D.l[i]); hi = Math.max(hi, D.h[i]); elo = Math.min(elo, D.eq[i]); ehi = Math.max(ehi, D.eq[i]); }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad; const epad = Math.max(0.002, (ehi - elo) * 0.1); elo -= epad; ehi += epad;
    const sp = (W - P.l - P.r) / n, X = i => P.l + (i + 0.5) * sp;
    const Y = v => H - P.b - (v - lo) / (hi - lo) * (H - P.t - P.b), EY = v => H - P.b - (v - elo) / (ehi - elo) * (H - P.t - P.b);
    const dp = D.c[0] > 50 ? 2 : 5;
    let g = "";
    for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v); g += `<line x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" stroke="${C.grid}"/><text x="${P.l - 6}" y="${y + 3}" fill="${C.dim}" font-size="10" text-anchor="end">${v.toFixed(dp)}</text>`; }
    for (let k = 0; k <= 2; k++) { const v = elo + (ehi - elo) * k / 2; g += `<text x="${W - P.r + 6}" y="${EY(v) + 3}" fill="${C.gold}" font-size="10" opacity=".8">${pct(v - 1)}</text>`; }
    const bw = Math.max(1, sp * 0.6);
    for (let i = 0; i < n; i++) {
      const up = D.c[i] >= D.o[i], col = up ? C.green : C.mag, x = X(i);
      g += `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${Y(D.h[i]).toFixed(1)}" y2="${Y(D.l[i]).toFixed(1)}" stroke="${col}" opacity=".65"/>`;
      const y0 = Y(Math.max(D.o[i], D.c[i])), y1 = Y(Math.min(D.o[i], D.c[i]));
      g += `<rect x="${(x - bw / 2).toFixed(1)}" y="${y0.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, y1 - y0).toFixed(1)}" fill="${col}" opacity=".85"/>`;
    }
    g += `<polyline points="${D.eq.map((e, i) => `${X(i).toFixed(1)},${EY(e).toFixed(1)}`).join(" ")}" fill="none" stroke="${C.gold}" stroke-width="1.8" opacity=".9" style="filter:drop-shadow(0 0 3px ${C.gold})"/>`;
    for (let i = 0; i < n; i++) {
      const prev = i > 0 ? D.pos[i - 1] : 0, x = X(i);
      if (D.pos[i] !== prev && D.pos[i] !== 0) {
        const upT = D.pos[i] > 0, y = upT ? Y(D.l[i]) + 9 : Y(D.h[i]) - 9, s = 5;
        g += `<path d="M${x},${upT ? y - s : y + s} L${x - s},${upT ? y + s : y - s} L${x + s},${upT ? y + s : y - s}Z" fill="${upT ? C.green : C.mag}" style="filter:drop-shadow(0 0 3px ${upT ? C.green : C.mag})"/>`;
      }
      if (D.tpnl[i] !== 0) g += `<circle cx="${x}" cy="${Y(D.c[i])}" r="3.4" fill="${D.tpnl[i] > 0 ? C.gold : C.mag}" stroke="#02030a" stroke-width="1"/>`;
    }
    g += `<text x="${P.l}" y="${H - 3}" fill="${C.dim}" font-size="10">${esc(String(D.ts[0]).slice(0, 16))}</text><text x="${W - P.r}" y="${H - 3}" fill="${C.dim}" font-size="10" text-anchor="end">${esc(String(D.ts[n - 1]).slice(0, 16))}</text>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = g;
  }

  function heat(D) {
    const host = $("dHeat");
    const q = Uint8Array.from(atob(D.q), c => c.charCodeAt(0)), Q = new Int8Array(q.buffer), W = D.width, n = D.frames;
    let off = D.n_in;
    const towers = [["ACTOR", D.actor_layers], ["CRITIC", D.critic_layers]];
    const cols = [[], []];
    towers.forEach(([name, sizes], t) => sizes.forEach((sz, li) => { cols[t].push({ label: `${name} L${li + 1} · ${sz}${sz > 64 ? " (64 shown)" : ""}`, start: off, size: sz, maxRows: 64 }); off += sz; }));
    const NA = D.actions.length;
    cols[0].push({ label: "ACTION PROBABILITY · " + (D.action_labels || D.actions).join(" / "), start: off, size: NA, maxRows: NA, rowColors: D.actions.map((nm, a) => actionColor(nm, a)) });
    cols[1].push({ label: "CRITIC V(s)", start: off + NA, size: 1, maxRows: 1, signed: true });
    const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
    const POS = [255, 120, 30], NEG = [25, 190, 255];
    host.innerHTML = cols.map((c, t) => `<div class="blk">${c.map((b, k) => `<div class="lab">${esc(b.label)}</div><canvas data-t="${t}" data-k="${k}"></canvas>`).join("")}</div>`).join("");
    host.querySelectorAll("canvas").forEach(cv => {
      const b = cols[+cv.dataset.t][+cv.dataset.k];
      const rows = Math.min(b.size, b.maxRows), step = b.size / rows;
      const rowH = b.rowColors ? 6 : b.size === 1 ? 10 : 1;
      cv.width = n; cv.height = rows * rowH;
      cv.style.height = Math.min(140, Math.max(rows * rowH * 2, b.size === 1 ? 16 : 24)) + "px";
      const ctx = cv.getContext("2d"), img = ctx.createImageData(n, rows * rowH), px = img.data;
      for (let r = 0; r < rows; r++) {
        const col = b.start + Math.floor(r * step), rc = b.rowColors ? hex(b.rowColors[r]) : null;
        for (let f = 0; f < n; f++) {
          const v = Q[f * W + col] / 127, m = Math.min(1, Math.abs(v));
          const base = rc || (v >= 0 ? POS : NEG), k = b.rowColors ? m : Math.pow(m, 0.8);
          for (let y = 0; y < rowH; y++) {
            const o = ((r * rowH + y) * n + f) * 4;
            px[o] = 6 + base[0] * k; px[o + 1] = 9 + base[1] * k; px[o + 2] = 22 + base[2] * k; px[o + 3] = 255;
          }
        }
      }
      ctx.putImageData(img, 0, 0);
    });
  }

  function trades(D) {
    const list = closedTrades(D).slice(-14).reverse();
    if (!list.length) { $("dTrades").innerHTML = `<div class="dnote">no closed trades in this replay</div>`; return; }
    $("dTrades").innerHTML = `<table class="dtable"><tr><th>EXIT BAR</th><th>SIDE</th><th>BARS</th><th>PNL</th></tr>` +
      list.map(t => `<tr><td>${esc(String(D.ts[t.i]).slice(0, 16))}</td><td style="color:${t.side === "LONG" ? C.green : t.side === "SHORT" ? C.mag : C.dim}">${t.side}</td>` +
        `<td>${t.bars ?? "–"}</td><td style="color:${t.pnl > 0 ? C.green : C.mag}">${pct(t.pnl)}</td></tr>`).join("") + `</table>`;
  }

  function render(state) {
    last = state;
    const { index = [], snap: D = null, selectedI = -1 } = state;
    const entry = index.find(e => e.i === selectedI) || null;
    if (D) kpis(selectedI, D, entry);
    const history = index.length ? index : (D ? [Object.assign({ i: 0 }, summarize(D))] : []);
    const sums = history.map(e => { const s = summaryOf(e); return s ? Object.assign({ i: e.i }, s) : null; });
    if (D && entry && !summaryOf(entry)) cache.set(entry.i, summarize(D));
    const lastI = history.length ? history[history.length - 1].i : -1;
    learning(index, selectedI);
    equity(sums, selectedI, lastI);
    if (D) {
      mix(sums.length ? sums : [Object.assign({ i: selectedI }, summarize(D))], D.actions, D.action_labels || D.actions, selectedI);
      const hdr = entry ? `eval ${entry.i + 1} · step ${Number(entry.step).toLocaleString()}` : ((D.meta && D.meta.model) || "snapshot");
      $("dReplayHdr").textContent = `${hdr} · ${String(D.ts[0]).slice(0, 10)} → ${String(D.ts[D.frames - 1]).slice(0, 10)}`;
      $("dTradeHdr").textContent = hdr;
      replay(D); heat(D); trades(D);
    }
    fillMissing(index);
  }

  return { render, rerender: () => last && render(last) };
}
