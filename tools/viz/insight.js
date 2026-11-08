// Layer insight (2026-11-08): two read-outs that make the hidden layers say something you can act on.
//
//  1. layerHealth — per hidden layer, over the bars in the loaded snapshot:
//       dead   neurons whose output hardly moves (std < DEAD_STD): the same number on every bar, so they
//              tell the next layer nothing about the market (a constant bias). Many dead = unused width.
//       pinned share of activations stuck at |tanh| >= SAT: the tanh gradient is ~0 there, so those
//              neurons barely learn (weights / inputs too large, or the learning rate too high).
//       dims   effective dimension = participation ratio of the activation covariance eigenvalues,
//              (Σλ)² / Σλ²: how many independent directions the layer really uses, out of its size.
//  2. regimeMap — PCA of one layer's activations to 2D. Each bar is a dot; bars the network treats
//     alike land together. Colour by the action taken, by the price move over the next h bars, or by
//     the result of the trade opened on that bar, to see where (in the network's own terms) it wins
//     and where it loses. Click a dot to jump the replay there.
//
// Everything is computed here from the quantised activations every snapshot already carries (q / 127;
// tanh layers raw, other activations scaled per layer by their 99th percentile in snapshot.pack), so old
// snapshots work too. Pure functions are exported for test_viz_insight.py (run through node).

export const DEAD_STD = 0.02, SAT = 0.97;
export const DEAD_WARN = 0.10, DEAD_BAD = 0.25, SAT_WARN = 0.25, SAT_BAD = 0.5;
export const MAX_ROWS = 400;

const pct = v => (v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0) + "%";
const layerName = h => `${h.tower ? "CRITIC" : "ACTOR"} L${h.idx + 1}`;

// frames used for the statistics: all of them, or MAX_ROWS evenly spaced
export function sampleRows(nf, max = MAX_ROWS) {
  const m = Math.min(nf, max), r = new Int32Array(m);
  for (let t = 0; t < m; t++) r[t] = m === nf ? t : Math.round(t * (nf - 1) / Math.max(1, m - 1));
  return r;
}

// rows x size matrix of one layer's activations (Float64, row-major)
export function layerMatrix(s, L, rows = sampleRows(s.NF)) {
  const m = rows.length, n = L.size, X = new Float64Array(m * n);
  for (let t = 0; t < m; t++) { const o = rows[t] * s.W + L.off; for (let j = 0; j < n; j++) X[t * n + j] = s.Q[o + j] / 127; }
  return { X, m, n, rows };
}

export function centerCols(M) {
  const { X, m, n } = M, mu = new Float64Array(n);
  for (let t = 0; t < m; t++) for (let j = 0; j < n; j++) mu[j] += X[t * n + j];
  for (let j = 0; j < n; j++) mu[j] /= Math.max(1, m);
  for (let t = 0; t < m; t++) for (let j = 0; j < n; j++) X[t * n + j] -= mu[j];
  return M;
}

// participation ratio (Σλ)² / Σλ² of the covariance of a column-centred matrix. tr(C) = Σλ and
// ||C||_F² = Σλ²; the smaller Gram matrix (n x n or m x m) has the same non-zero eigenvalues.
export function participation(M) {
  const { X, m, n } = M;
  let tr = 0, fro = 0;
  if (n <= m) {
    for (let a = 0; a < n; a++) for (let b = a; b < n; b++) {
      let c = 0; for (let t = 0; t < m; t++) c += X[t * n + a] * X[t * n + b];
      if (a === b) { tr += c; fro += c * c; } else fro += 2 * c * c;
    }
  } else {
    for (let a = 0; a < m; a++) for (let b = a; b < m; b++) {
      let c = 0; const oa = a * n, ob = b * n; for (let j = 0; j < n; j++) c += X[oa + j] * X[ob + j];
      if (a === b) { tr += c; fro += c * c; } else fro += 2 * c * c;
    }
  }
  return fro > 1e-18 ? tr * tr / fro : 0;
}

function isTanh(s, L) {
  const acts = s.D && s.D.layer_act;           // written by snapshot.pack since 2026-11-08; older = SB3 default tanh
  if (!acts) return true;
  const a = (acts[L.tower ? "critic" : "actor"] || [])[L.idx];
  return a == null || a === "tanh";
}

export function layerHealth(s) {
  const rows = sampleRows(s.NF);
  return s.layers.map(L => {
    const M = layerMatrix(s, L, rows), { X, m, n } = M, tanh = isTanh(s, L);
    const sum = new Float64Array(n), sq = new Float64Array(n);
    let pinned = 0;
    for (let t = 0; t < m; t++) for (let j = 0; j < n; j++) {
      const v = X[t * n + j]; sum[j] += v; sq[j] += v * v; if (tanh && Math.abs(v) >= SAT) pinned++;
    }
    const dead = new Uint8Array(n); let nDead = 0;
    for (let j = 0; j < n; j++) {
      const mu = sum[j] / m, sd = Math.sqrt(Math.max(0, sq[j] / m - mu * mu));
      if (sd < DEAD_STD) { dead[j] = 1; nDead++; }
    }
    const dims = participation(centerCols(M));
    const deadShare = nDead / n, satShare = tanh ? pinned / (m * n) : null, sat = satShare || 0;
    const cls = deadShare >= DEAD_BAD || sat >= SAT_BAD ? "bad" : deadShare >= DEAD_WARN || sat >= SAT_WARN ? "warn" : "ok";
    return { tower: L.tower, idx: L.idx, size: n, bars: m, tanh, dead, nDead, deadShare, satShare, dims, cls };
  });
}

// plain-language findings, most important first (at most three lines: the panel has to stay on screen)
export function healthAdvice(H) {
  const out = [], names = hs => hs.map(layerName).join(", ");
  const dead = H.filter(h => h.deadShare >= DEAD_WARN), pinned = H.filter(h => h.satShare != null && h.satShare >= SAT_WARN);
  if (pinned.length) {
    const worst = pinned.reduce((a, b) => b.satShare > a.satShare ? b : a);
    out.push({ cls: worst.satShare >= SAT_BAD ? "bad" : "warn",
      t: `${names(pinned)} pinned at ±1 (${names([worst])} ${pct(worst.satShare)}): there tanh passes almost no gradient and the neuron acts as an on/off switch, so the layer carries less information and learns only through the unpinned rest. `
       + `Whether it still learns shows in the LEARNING tab (Δ weights). To A/B: activation_fn=ReLU, a lower learning rate or a smaller window.` });
  }
  if (dead.length) {
    const worst = dead.reduce((a, b) => b.deadShare > a.deadShare ? b : a);
    out.push({ cls: worst.deadShare >= DEAD_BAD ? "bad" : "warn",
      t: `${names(dead)}: dead neurons (${names([worst])} ${pct(worst.deadShare)}) give the same output on every bar — width this net does not use.` });
  }
  const actor = H.filter(h => h.tower === 0), last = actor[actor.length - 1];
  if (actor.length >= 3 && last && (last.deadShare >= DEAD_WARN || last.dims < 0.05 * last.size))
    out.push({ cls: "warn", t: `the deepest actor layer uses ${last.dims.toFixed(1)} of ${last.size} dimensions → a 2-layer or narrower net_arch is worth an A/B test` });
  if (!out.length) out.push({ cls: "ok", t: "no dead or pinned layers on these bars — every layer passes information on" });
  return out;
}

// top-2 principal components of a column-centred matrix by power iteration on X^T X (deterministic start)
export function pca2(M, iters = 80) {
  const { X, m, n } = M;
  let tot = 0; for (let i = 0; i < X.length; i++) tot += X[i] * X[i];
  const vecs = [], lam = [], xv = new Float64Array(m);
  for (let k = 0; k < 2; k++) {
    let v = new Float64Array(n);
    for (let j = 0; j < n; j++) v[j] = 1 + ((j * 7919 + k * 104729) % 997) / 997;
    for (let it = 0; it < iters; it++) {
      for (let t = 0; t < m; t++) { let a = 0; const o = t * n; for (let j = 0; j < n; j++) a += X[o + j] * v[j]; xv[t] = a; }
      const w = new Float64Array(n);
      for (let t = 0; t < m; t++) { const a = xv[t], o = t * n; if (a) for (let j = 0; j < n; j++) w[j] += X[o + j] * a; }
      for (const u of vecs) { let d = 0; for (let j = 0; j < n; j++) d += w[j] * u[j]; for (let j = 0; j < n; j++) w[j] -= d * u[j]; }
      let nr = 0; for (let j = 0; j < n; j++) nr += w[j] * w[j]; nr = Math.sqrt(nr);
      if (nr < 1e-15) break;
      for (let j = 0; j < n; j++) w[j] /= nr;
      v = w;
    }
    let big = 0; for (let j = 0; j < n; j++) if (Math.abs(v[j]) > Math.abs(big)) big = v[j];
    if (big < 0) for (let j = 0; j < n; j++) v[j] = -v[j];        // sign convention: largest loading positive
    let l = 0; for (let t = 0; t < m; t++) { let a = 0; const o = t * n; for (let j = 0; j < n; j++) a += X[o + j] * v[j]; l += a * a; }
    vecs.push(v); lam.push(l);
  }
  const xy = new Float32Array(m * 2);
  for (let t = 0; t < m; t++) for (let k = 0; k < 2; k++) { let a = 0; const o = t * n, v = vecs[k]; for (let j = 0; j < n; j++) a += X[o + j] * v[j]; xy[t * 2 + k] = a; }
  return { xy, pc: lam.map(l => tot > 0 ? l / tot : 0), explained: tot > 0 ? (lam[0] + lam[1]) / tot : 0 };
}

// entry bars (flat -> position, or a flip) and the P&L of the trade each one opened (NaN = still open)
export function entryResults(D) {
  const n = D.pos.length, pnl = new Float64Array(n).fill(NaN), side = new Int8Array(n), close = new Int32Array(n).fill(-1);
  for (let i = 1; i < n; i++) {
    if (D.pos[i] === 0 || D.pos[i] === D.pos[i - 1]) continue;
    side[i] = Math.sign(D.pos[i]);
    for (let j = i + 1; j < n; j++) if (D.tpnl[j] !== 0) { pnl[i] = D.tpnl[j]; close[i] = j; break; }
  }
  return { pnl, side, close };
}

// horizon for the "next move" colour: median bars held, else 6
export function moveHorizon(D, E = entryResults(D)) {
  const holds = [];
  for (let i = 0; i < E.close.length; i++) if (E.close[i] > 0) holds.push(E.close[i] - i);
  if (!holds.length) return 6;
  holds.sort((a, b) => a - b);
  return Math.max(1, Math.min(48, holds[Math.floor(holds.length / 2)]));
}

export function forwardMove(D, h) {
  const n = D.c.length, f = new Float64Array(n).fill(NaN);
  for (let i = 0; i + h < n; i++) if (D.c[i] > 0) f[i] = D.c[i + h] / D.c[i] - 1;
  return f;
}

export function regimeMap(s, L) {
  const rows = sampleRows(s.NF), M = centerCols(layerMatrix(s, L, rows));
  return Object.assign(pca2(M), { rows, layer: L });
}

// ------------------------------------------------------------------ DOM: health panel
export function renderHealth(el, s, H) {
  const rowsHtml = H.map(h => `<tr class="${h.cls}"><td>${layerName(h)}</td><td>${h.size}</td>
      <td><span class="hb" style="width:${Math.max(1, h.deadShare * 60).toFixed(0)}px"></span>${pct(h.deadShare)}</td>
      <td>${h.satShare == null ? "–" : `<span class="hb" style="width:${Math.max(1, h.satShare * 60).toFixed(0)}px"></span>${pct(h.satShare)}`}</td>
      <td>${h.dims.toFixed(h.dims < 10 ? 1 : 0)}</td></tr>`).join("");
  const adv = healthAdvice(H).map(a => `<div class="adv ${a.cls}">${a.t}</div>`).join("");
  const src = s.ROLL ? "training rollout" : ((s.D.meta && s.D.meta.slice) || "validation") + " replay";
  el.innerHTML = `<div class="k">LAYER HEALTH · ${H[0] ? H[0].bars : 0} bars of this ${src}</div>
    <table class="hl"><tr><th></th><th>size</th><th>dead</th><th>pinned ±1</th><th>dims used</th></tr>${rowsHtml}</table>${adv}
    <div class="note">dead = output std &lt; ${DEAD_STD} on these bars (grey dots in the layer) · pinned = |tanh| ≥ ${SAT} ·
    dims = how many independent directions the layer's activity spans (participation ratio). Few dims in a wide layer = more width than the data needs.</div>`;
}

// ------------------------------------------------------------------ DOM: regime map panel
export function createRegime(el, { onSeek, css }) {
  let s = null, R = null, E = null, fwd = null, h = 6, li = -1, colorBy = "trade", cur = -1, hover = -1;
  let dpr = 1, W = 316, Hh = 224, px = null;
  el.innerHTML = `<div class="k">REGIME MAP · <select class="rl"></select> colour <select class="rc">
      <option value="trade">trade result</option><option value="move">next move</option><option value="action">action</option></select></div>
    <canvas></canvas><div class="leg"></div><div class="tip">&nbsp;</div><div class="note"></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d"), selL = el.querySelector(".rl"), selC = el.querySelector(".rc");
  selC.value = colorBy;
  selL.onchange = () => { li = +selL.value; compute(); draw(); };
  selC.onchange = () => { colorBy = selC.value; draw(); };

  function compute() {
    const L = s.towerLayers[0][li] || s.towerLayers[0][s.towerLayers[0].length - 1];
    R = regimeMap(s, L);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let t = 0; t < R.rows.length; t++) { const x = R.xy[t * 2], y = R.xy[t * 2 + 1]; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const pad = 10, sx = (W - 2 * pad) / Math.max(1e-9, x1 - x0), sy = (Hh - 2 * pad) / Math.max(1e-9, y1 - y0);
    px = new Float32Array(R.rows.length * 2);
    for (let t = 0; t < R.rows.length; t++) { px[t * 2] = pad + (R.xy[t * 2] - x0) * sx; px[t * 2 + 1] = Hh - pad - (R.xy[t * 2 + 1] - y0) * sy; }
  }
  function moveScale() {
    const a = []; for (const v of fwd) if (Number.isFinite(v)) a.push(Math.abs(v));
    a.sort((x, y) => x - y); return a.length ? Math.max(1e-9, a[Math.floor((a.length - 1) * 0.95)]) : 1;
  }
  function draw() {
    if (!s || !R) return;
    const C = css(), n = R.rows.length;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, Hh);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, Hh - 1);
    const ms = colorBy === "move" ? moveScale() : 1;
    const dotsFirst = [], dotsTop = [];
    for (let t = 0; t < n; t++) {
      const f = R.rows[t];
      let col = C.dim, r = 2.1, a = 0.55, top = false;
      if (colorBy === "action") { col = s.colors[s.D.act[f]] || C.dim; a = 0.7; }
      else if (colorBy === "move") {
        const v = fwd[f];
        if (Number.isFinite(v)) { const m = Math.min(1, Math.abs(v) / ms); col = v >= 0 ? C.up : C.dn; a = 0.15 + 0.75 * m; r = 1.6 + m * 1.6; }
        else a = 0.2;
      } else {
        if (E.side[f]) { top = true; r = 3.6; a = 0.95; const p = E.pnl[f]; col = Number.isFinite(p) ? (p > 0 ? C.up : C.dn) : C.gold; }
        else { a = 0.22; r = 1.5; }
      }
      (top ? dotsTop : dotsFirst).push([t, col, r, a, E.side[f], E.pnl[f]]);
    }
    for (const d of [...dotsFirst, ...dotsTop]) {
      const [t, col, r, a, side, p] = d, x = px[t * 2], y = px[t * 2 + 1];
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.strokeStyle = col; ctx.beginPath();
      if (colorBy === "trade" && side) {            // ▲ long entry, ▼ short entry; hollow = still open
        ctx.moveTo(x, y - side * r); ctx.lineTo(x - r, y + side * r * 0.8); ctx.lineTo(x + r, y + side * r * 0.8); ctx.closePath();
        if (Number.isFinite(p)) ctx.fill(); else { ctx.lineWidth = 1.2; ctx.stroke(); }
      } else { ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }
    }
    ctx.globalAlpha = 1;
    // current bar: short trail + ring
    const ci = rowOf(cur);
    if (ci >= 0) {
      ctx.strokeStyle = C.fg; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.55; ctx.beginPath();
      for (let t = Math.max(0, ci - 8); t <= ci; t++) { const x = px[t * 2], y = px[t * 2 + 1]; t === Math.max(0, ci - 8) ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke(); ctx.globalAlpha = 1;
      ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(px[ci * 2], px[ci * 2 + 1], 6, 0, Math.PI * 2); ctx.stroke();
    }
    if (hover >= 0) { ctx.strokeStyle = C.gold; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px[hover * 2], px[hover * 2 + 1], 5, 0, Math.PI * 2); ctx.stroke(); }
    legend(C);
  }
  function legend(C) {
    const leg = el.querySelector(".leg"), sw = (c, t, shape = "●") => `<span style="color:${c}">${shape}</span> ${t}`;
    if (colorBy === "action") leg.innerHTML = s.labels.map((l, a) => sw(s.colors[a], l)).join(" &nbsp; ");
    else if (colorBy === "move") leg.innerHTML = `${sw(C.up, "price up")} &nbsp; ${sw(C.dn, "price down")} over the next ${h} bars (strength = size)`;
    else {
      let w = 0, l = 0, o = 0;
      for (let f = 0; f < E.side.length; f++) if (E.side[f]) { const p = E.pnl[f]; if (!Number.isFinite(p)) o++; else if (p > 0) w++; else l++; }
      leg.innerHTML = `${sw(C.up, `win ${w}`, "▲")} &nbsp; ${sw(C.dn, `loss ${l}`, "▲")} &nbsp; ${sw(C.gold, `open ${o}`, "△")} &nbsp; ▲ long ▼ short · grey = no entry`;
    }
    const L = R.layer;
    el.querySelector(".note").textContent = `Each dot is one bar, placed by the PCA of ACTOR L${L.idx + 1}'s ${L.size} activations `
      + `(the two axes keep ${pct(R.explained)} of its variance${R.explained < 0.4 ? " — a rough map, distances are approximate" : ""}). `
      + `Bars the network treats alike sit together: a corner where one colour dominates is a market state it has learned to tell apart; `
      + `a corner of red entries is where a filter would help. Click a dot to jump there.`;
  }
  function rowOf(f) {
    if (!R || f < 0) return -1;
    const m = R.rows.length; return m === s.NF ? Math.min(m - 1, f) : Math.round(f * (m - 1) / Math.max(1, s.NF - 1));
  }
  function nearest(ev) {
    const b = cv.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top;
    let best = -1, bd = 64;
    for (let t = 0; t < R.rows.length; t++) { const dx = px[t * 2] - x, dy = px[t * 2 + 1] - y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = t; } }
    return best;
  }
  function tip(t) {
    const T = el.querySelector(".tip");
    if (t < 0) { T.innerHTML = "&nbsp;"; return; }
    const f = R.rows[t], D = s.D, mv = fwd[f], p = E.pnl[f];
    const parts = [`bar ${f + 1}`, String(D.ts[f]).slice(0, 16), s.labels[D.act[f]] || D.act[f]];
    if (Number.isFinite(mv)) parts.push(`next ${h}: ${mv >= 0 ? "+" : ""}${(mv * 100).toFixed(2)}%`);
    if (E.side[f]) parts.push(Number.isFinite(p) ? `${E.side[f] > 0 ? "long" : "short"} ${p >= 0 ? "+" : ""}${(p * 1e4).toFixed(1)} bp` : `${E.side[f] > 0 ? "long" : "short"} open`);
    T.textContent = parts.join(" · ");
  }
  cv.onmousemove = ev => { if (!R) return; const t = nearest(ev); if (t !== hover) { hover = t; tip(t); draw(); } cv.style.cursor = t >= 0 ? "pointer" : "default"; };
  cv.onmouseleave = () => { hover = -1; tip(-1); draw(); };
  cv.onclick = ev => { if (!R) return; const t = nearest(ev); if (t >= 0 && onSeek) onSeek(R.rows[t]); };

  function size() {
    dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W * dpr; cv.height = Hh * dpr; cv.style.width = W + "px"; cv.style.height = Hh + "px";
  }
  size();
  return {
    set(snap) {
      s = snap; E = entryResults(s.D); h = moveHorizon(s.D, E); fwd = forwardMove(s.D, h);
      const tl = s.towerLayers[0];
      selL.innerHTML = tl.map((L, i) => `<option value="${i}">L${i + 1} · ${L.size}</option>`).join("");
      if (li < 0 || li >= tl.length) li = tl.length - 1;
      selL.value = String(li); hover = -1; compute(); draw();
    },
    frame(f) { if (f !== cur) { cur = f; draw(); } },
    redraw: draw,
    pxOf(t) { return px && t >= 0 && t < R.rows.length ? { x: px[t * 2], y: px[t * 2 + 1], bar: R.rows[t] } : null; },   // tests
    get n() { return R ? R.rows.length : 0; },
  };
}

// ================================================================== 2026-11-08 (b): LEARNING · INPUTS · CRITIC tabs
// Each reads either the loaded snapshot or the per-evaluation history (one summarizeEval row per eval
// snapshot of the run, fetched by the viewer when the tab opens).

// names of the usage groups: every feature (all lags together), then the state inputs
export function groupNames(s) {
  const n = s.D.input_names || [], out = [];
  for (let f = 0; f < s.NFEAT; f++) out.push((n[f] || `f${f}`).split("@")[0]);
  for (let i = s.WIN * s.NFEAT; i < s.NIN; i++) out.push(n[i] || `in ${i}`);
  return out;
}

// ---- critic check: does V(s) predict what actually came after the bar?
// R on a training rollout = GAE return (advantage + V), exactly the target the critic is trained on;
// on an eval replay = discounted reward sum (gamma from meta, default 0.99), bootstrapped with V at the last bar.
export function discountedReturns(reward, values, gamma) {
  const n = reward.length, R = new Float64Array(n);
  let g = values.length ? values[n - 1] : 0;
  for (let t = n - 1; t >= 0; t--) { g = reward[t] + gamma * g; R[t] = g; }
  return R;
}
export function explainedVariance(V, R) {
  const n = Math.min(V.length, R.length);
  if (n < 3) return null;
  let mr = 0, md = 0; for (let t = 0; t < n; t++) { mr += R[t]; md += R[t] - V[t]; } mr /= n; md /= n;
  let vr = 0, vd = 0; for (let t = 0; t < n; t++) { vr += (R[t] - mr) ** 2; const d = R[t] - V[t] - md; vd += d * d; }
  return vr > 1e-18 ? 1 - vd / vr : null;
}
function corr(a, b) {
  const n = Math.min(a.length, b.length); let ma = 0, mb = 0;
  for (let t = 0; t < n; t++) { ma += a[t]; mb += b[t]; } ma /= n; mb /= n;
  let sab = 0, sa = 0, sb = 0; for (let t = 0; t < n; t++) { const x = a[t] - ma, y = b[t] - mb; sab += x * y; sa += x * x; sb += y * y; }
  return sa > 0 && sb > 0 ? sab / Math.sqrt(sa * sb) : null;
}
export function criticCheck(s) {
  const D = s.D, n = s.NF, V = new Float64Array(n), vmax = D.value_max || 1;
  for (let f = 0; f < n; f++) V[f] = s.Q[f * s.W + s.OFF_VAL] / 127 * vmax;
  let R, kind;
  if (D.adv && D.adv.length === n) { R = Float64Array.from(D.adv, (a, f) => a + V[f]); kind = "gae"; }
  else if (D.reward && D.reward.length === n) { R = discountedReturns(D.reward, V, (D.meta && D.meta.gamma) || 0.99); kind = "mc"; }
  else return null;
  return { V, R, kind, ev: explainedVariance(V, R), corr: corr(V, R), gamma: (D.meta && D.meta.gamma) || 0.99 };
}
export function criticVerdict(ev) {
  if (ev == null) return { cls: "warn", t: "not enough bars to judge" };
  if (ev < 0) return { cls: "bad", t: `explained variance ${ev.toFixed(2)} < 0: V(s) is worse than predicting the average return. The advantages that steer the policy are mostly noise, so more steps help little — check the reward scale, vf_coef or the critic's size first.` };
  if (ev < 0.2) return { cls: "warn", t: `explained variance ${ev.toFixed(2)}: the critic explains little of what follows a bar — advantages are noisy and the policy learns slowly.` };
  if (ev < 0.6) return { cls: "ok", t: `explained variance ${ev.toFixed(2)}: the critic tracks part of the return — usable advantages.` };
  return { cls: "ok", t: `explained variance ${ev.toFixed(2)}: the critic predicts the return well. If the policy still does not improve, look at the reward or the features, not the critic.` };
}

// ---- one row per evaluation snapshot (history for LEARNING and INPUTS)
export function summarizeEval(s) {
  const m = s.D.meta || {}, H = s.health || layerHealth(s), C = criticCheck(s);
  return { i: m.eval_index != null ? m.eval_index : -1, step: m.step != null ? m.step : null,
           reward: Number.isFinite(m.eval_reward) ? m.eval_reward : null,
           layers: H.map(h => ({ tower: h.tower, idx: h.idx, size: h.size, sat: h.satShare, dead: h.deadShare, dims: h.dims })),
           delta: m.layer_delta || null, use: s.D.use_flip || null, names: groupNames(s), ev: C ? C.ev : null };
}

export function learningAdvice(rows) {
  const out = [], last = rows[rows.length - 1];
  if (!last) return out;
  const withD = rows.filter(r => r.delta);
  const lname = (tw, i) => `${tw === "critic" ? "CRITIC" : "ACTOR"} L${i + 1}`;
  if (withD.length >= 2) {
    const first = withD[0].delta, now = withD[withD.length - 1].delta, ratios = [];
    for (const tw of ["actor", "critic"]) (now[tw] || []).forEach((v, i) => { const f = (first[tw] || [])[i]; if (f > 0) ratios.push([lname(tw, i), v / f, v]); });
    const rew = rows.map(r => r.reward), best = rew.reduce((b, v, k) => v != null && (b < 0 || v > rew[b]) ? k : b, -1);
    const stale = best >= 0 && rows.length - 1 - best >= 3;
    if (ratios.length && ratios.every(r => r[1] < 0.25) && stale)
      out.push({ cls: "bad", t: `learning has stalled: every layer now moves < 25% of its first-evaluation change and the best eval reward is from eval ${rows[best].i + 1} (${rows.length - 1 - best} evals ago). More steps are unlikely to help — stop, or change the learning rate / architecture.` });
    const all = ratios.map(r => r[2]).sort((a, b) => a - b), med = all[Math.floor(all.length / 2)] || 0;
    for (const [nm, , v] of ratios) if (med > 0 && v < 0.2 * med)
      out.push({ cls: "warn", t: `${nm} moves ${(med / Math.max(v, 1e-12)).toFixed(0)}× less than the median layer — it is (nearly) frozen while the others still learn.` });
  } else out.push({ cls: "warn", t: "weight change per layer is recorded by runs trained from 2026-11-08 on (rl_train --viz); this run shows the pinned trend only." });
  const firstRow = rows[0];
  if (rows.length >= 2) last.layers.forEach((h, k) => {
    const a = firstRow.layers[k] && firstRow.layers[k].sat, b = h.sat;
    const nm = `${h.tower ? "CRITIC" : "ACTOR"} L${h.idx + 1}`;
    if (a != null && b != null && b - a >= 0.2)
      out.push({ cls: b >= SAT_BAD ? "bad" : "warn", t: `${nm} saturated while it trained: pinned ${pct(a)} at eval ${firstRow.i + 1} → ${pct(b)} now.` });
    else if (a != null && b != null && a >= SAT_BAD && b >= SAT_BAD)
      out.push({ cls: "bad", t: `${nm} was already ${pct(a)} pinned at eval ${firstRow.i + 1} (step ${firstRow.step != null ? firstRow.step.toLocaleString() : "?"}) and is ${pct(b)} now: it saturated early; compare with its Δ weights to see whether it still learns.` });
  });
  if (!out.length) out.push({ cls: "ok", t: "every layer is still moving and none saturated over training" });
  return out;
}

export function usageAdvice(rows, used = 0.01, strong = 0.05) {
  const R = rows.filter(r => r.use), out = [];
  if (R.length < 2) return [{ cls: "warn", t: R.length ? "one evaluation with a decision test so far — the trend needs two or more." : "no decision test in these snapshots (made before 2026-10-19)." }];
  const names = R[R.length - 1].names, n = R[R.length - 1].use.length;
  const never = [], dropped = [], rising = [];
  for (let f = 0; f < n; f++) {
    const v = R.map(r => r.use[f] || 0), mx = Math.max(...v), lastV = v[v.length - 1], firstV = v[0];
    if (mx < used) never.push(names[f]);
    else if (Math.max(...v.slice(0, Math.ceil(v.length / 2))) >= strong && lastV < used) dropped.push(names[f]);
    else if (firstV < used && lastV >= strong) rising.push(names[f]);
  }
  const list = a => a.slice(0, 6).join(", ") + (a.length > 6 ? ` +${a.length - 6}` : "");
  if (never.length) out.push({ cls: "warn", t: `${never.length} of ${n} inputs never changed ≥${pct(used)} of decisions at any evaluation (${list(never)}) — candidates to drop; confirm with obs_ablation.` });
  if (dropped.length) out.push({ cls: "warn", t: `${dropped.length} inputs mattered early and are ignored now (${list(dropped)}).` });
  if (rising.length) out.push({ cls: "ok", t: `${rising.length} inputs the policy learned to use during training (${list(rising)}).` });
  if (!never.length) out.push({ cls: "ok", t: `every input changes ≥${pct(used)} of decisions at some evaluation. On an indecisive policy almost any shuffle flips a decision, so "used" ≠ "carries edge".` });
  return out;
}

// ---- small DOM helpers
function sparkSvg(vals, { w = 74, h = 18, color = "currentColor", mark = -1, lo = null, hi = null } = {}) {
  const f = vals.filter(v => v != null && Number.isFinite(v));
  if (!f.length) return `<svg width="${w}" height="${h}"></svg>`;
  const a = lo != null ? lo : Math.min(...f), b0 = hi != null ? hi : Math.max(...f), b = b0 === a ? a + 1e-9 : b0;
  const n = vals.length, x = k => (2 + k * (w - 4) / Math.max(1, n - 1)).toFixed(1);
  const pts = []; vals.forEach((v, k) => { if (v != null && Number.isFinite(v)) pts.push(`${x(k)},${(h - 2 - (v - a) / (b - a) * (h - 4)).toFixed(1)}`); });
  const mk = mark >= 0 ? `<line x1="${x(mark)}" x2="${x(mark)}" y1="1" y2="${h - 1}" stroke="currentColor" stroke-opacity=".35"/>` : "";
  return `<svg width="${w}" height="${h}" style="vertical-align:middle">${mk}<polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="1.4"/></svg>`;
}
const advHtml = A => A.map(a => `<div class="adv ${a.cls}">${a.t}</div>`).join("");

// ---- LEARNING tab: per layer, how much the weights moved each evaluation + how saturated it got
export function renderLearning(el, rows, curI, status) {
  if (!rows.length) { el.innerHTML = `<div class="k">LAYER LEARNING</div><div class="note">${status || "no evaluations yet"}</div>`; return; }
  const last = rows[rows.length - 1], mark = rows.findIndex(r => r.i === curI);
  const lines = last.layers.map((h, k) => {
    const tw = h.tower ? "critic" : "actor", d = rows.map(r => r.delta && r.delta[tw] ? r.delta[tw][h.idx] : null);
    const sat = rows.map(r => r.layers[k] ? r.layers[k].sat : null), dNow = d[mark >= 0 ? mark : d.length - 1];
    const cls = h.sat != null && h.sat >= SAT_BAD ? "bad" : h.sat != null && h.sat >= SAT_WARN ? "warn" : "";
    return `<tr class="${cls}"><td>${h.tower ? "CRITIC" : "ACTOR"} L${h.idx + 1}</td><td>${dNow == null ? "–" : (dNow * 100).toFixed(2) + "%"}</td>
      <td>${sparkSvg(d, { mark, lo: 0 })}</td><td>${sparkSvg(sat, { mark, lo: 0, hi: 1 })}</td></tr>`;
  }).join("");
  el.innerHTML = `<div class="k">LAYER LEARNING · ${rows.length} evaluations${status ? " · " + status : ""}</div>
    <table class="hl"><tr><th></th><th>Δ weights</th><th>Δ per eval</th><th>pinned ±1 (0–100%)</th></tr>${lines}</table>${advHtml(learningAdvice(rows))}
    <div class="note">Δ weights = ‖W − W<sub>previous eval</sub>‖ / ‖W<sub>previous eval</sub>‖: how far each layer moved since the previous evaluation (≈0 = it stopped learning). The vertical tick is the evaluation shown in 3D; the layer labels in 3D show the same numbers.</div>`;
}

// ---- INPUTS tab: decision-test usage of every input at every evaluation (heatmap) + verdicts
export function createUsageMap(el, { css, onPickEval }) {
  let rows = [], curI = -1, hover = null, W = 316, H = 190, dpr = 1, order = [];
  el.innerHTML = `<div class="k"><span class="ttl">INPUT USAGE OVER TRAINING</span></div><canvas></canvas><div class="tip">&nbsp;</div><div class="advs"></div>
    <div class="note">Rows = inputs (most used at the latest eval on top), columns = evaluations, brightness = share of decisions that change when that input is shuffled.
    The outlined column is the eval shown in 3D (its usage also colours the input layer); click a column to load that evaluation.</div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d");
  const R = () => rows.filter(r => r.use);
  function geom() { const rr = R(), n = rr.length ? rr[rr.length - 1].use.length : 0; return { rr, n, cw: rr.length ? W / rr.length : W, rh: n ? H / n : H }; }
  function draw() {
    dpr = Math.min(2, devicePixelRatio || 1); cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
    const C = css(), { rr, n, cw, rh } = geom();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    if (!rr.length) return;
    const last = rr[rr.length - 1].use; order = [...Array(n).keys()].sort((a, b) => (last[b] || 0) - (last[a] || 0));
    let mx = 0; for (const r of rr) for (const v of r.use) mx = Math.max(mx, v || 0);
    for (let c = 0; c < rr.length; c++) for (let k = 0; k < n; k++) {
      const v = rr[c].use[order[k]] || 0;
      ctx.globalAlpha = v <= 0 ? 0.06 : 0.12 + 0.88 * Math.sqrt(v / (mx || 1)); ctx.fillStyle = v <= 0 ? C.dim : C.use;
      ctx.fillRect(c * cw + 0.5, k * rh, Math.max(1, cw - 1), Math.max(0.8, rh));
    }
    ctx.globalAlpha = 1;
    const ci = rr.findIndex(r => r.i === curI);
    if (ci >= 0) { ctx.strokeStyle = C.fg; ctx.lineWidth = 1.5; ctx.strokeRect(ci * cw + 0.75, 0.75, cw - 1.5, H - 1.5); }
    if (hover) { ctx.strokeStyle = C.gold; ctx.lineWidth = 1; ctx.strokeRect(0.5, hover.k * rh, W - 1, Math.max(1.5, rh)); }
  }
  cv.onmousemove = ev => {
    const { rr, n, cw, rh } = geom(); if (!rr.length) return;
    const b = cv.getBoundingClientRect(), c = Math.min(rr.length - 1, Math.floor((ev.clientX - b.left) / cw)), k = Math.min(n - 1, Math.floor((ev.clientY - b.top) / rh));
    hover = { c, k }; const f = order[k], r = rr[c];
    el.querySelector(".tip").textContent = `${r.names[f] || "in " + f} · eval ${r.i + 1}${r.step != null ? " (step " + r.step.toLocaleString() + ")" : ""} · changes ${pct(r.use[f] || 0)} of decisions`;
    cv.style.cursor = "pointer"; draw();
  };
  cv.onmouseleave = () => { hover = null; el.querySelector(".tip").innerHTML = "&nbsp;"; draw(); };
  cv.onclick = () => { const { rr } = geom(); if (hover && rr[hover.c] && onPickEval) onPickEval(rr[hover.c].i); };
  return {
    set(r, cur, status) {
      rows = r; curI = cur; const rr = R();
      el.querySelector(".ttl").textContent = `INPUT USAGE OVER TRAINING · ${rr.length} evaluations${status ? " · " + status : ""}`;
      el.querySelector(".advs").innerHTML = advHtml(usageAdvice(rows)); draw();
    },
  };
}

// ---- CRITIC tab: V(s) against the realised return over the bars, with the replay cursor
export function createCritic(el, { css }) {
  let s = null, C = null, cur = -1, W = 316, H = 130, dpr = 1, hist = [];
  el.innerHTML = `<div class="k">CRITIC CHECK · <span class="ttl"></span></div><div class="big"></div><canvas></canvas><div class="leg"></div><div class="advs"></div><div class="note"></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d");
  function draw() {
    dpr = Math.min(2, devicePixelRatio || 1); cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    if (!C) return;
    const K = css(), n = C.V.length;
    let lo = Infinity, hi = -Infinity; for (let t = 0; t < n; t++) { lo = Math.min(lo, C.V[t], C.R[t]); hi = Math.max(hi, C.V[t], C.R[t]); }
    const pad = (hi - lo) * 0.08 || 1e-3; lo -= pad; hi += pad;
    const X = t => 4 + t * (W - 8) / Math.max(1, n - 1), Y = v => H - 4 - (v - lo) / (hi - lo) * (H - 8);
    ctx.strokeStyle = K.grid; ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
    if (lo < 0 && hi > 0) { ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke(); ctx.setLineDash([]); }
    for (const [arr, col, lw] of [[C.R, K.gold, 1.2], [C.V, K.cyan, 1.6]]) {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
      for (let t = 0; t < n; t++) t ? ctx.lineTo(X(t), Y(arr[t])) : ctx.moveTo(X(t), Y(arr[t])); ctx.stroke();
    }
    if (cur >= 0 && cur < n) { ctx.strokeStyle = K.fg; ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(X(cur), 0); ctx.lineTo(X(cur), H); ctx.stroke(); ctx.globalAlpha = 1;
      for (const [arr, col] of [[C.R, K.gold], [C.V, K.cyan]]) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(cur), Y(arr[cur]), 3, 0, Math.PI * 2); ctx.fill(); } }
  }
  function text() {
    const K = css();
    el.querySelector(".ttl").textContent = s.ROLL ? "training rollout (what the critic trains on)" : "eval replay";
    if (!C) {
      el.querySelector(".big").innerHTML = "";
      el.querySelector(".advs").innerHTML = advHtml([{ cls: "warn", t: "This eval snapshot has no per-bar rewards (captured before 2026-11-08). Switch source to Training (live): rollouts always carry the returns the critic trains on." }]);
      el.querySelector(".leg").innerHTML = ""; el.querySelector(".note").textContent = ""; return;
    }
    const evs = hist.map(r => r.ev), ev = C.ev;
    el.querySelector(".big").innerHTML = `<b>${ev == null ? "–" : ev.toFixed(2)}</b> explained variance · corr ${C.corr == null ? "–" : C.corr.toFixed(2)} · ${C.V.length} bars`
      + (evs.filter(v => v != null).length >= 2 ? ` · over evals ${sparkSvg(evs, { w: 70, h: 16 })}` : "");
    el.querySelector(".leg").innerHTML = `<span style="color:${K.cyan}">━</span> V(s) predicted &nbsp; <span style="color:${K.gold}">━</span> realised return R`;
    el.querySelector(".advs").innerHTML = advHtml([criticVerdict(ev)]);
    el.querySelector(".note").textContent = C.kind === "gae"
      ? "R = advantage + V(s) from the rollout buffer (the GAE return the critic is fitted to); explained variance = 1 − Var(R − V) / Var(R), the same number SB3 logs as train/explained_variance. The V(s) label in 3D shows the realised R next to the prediction."
      : `R = discounted sum of the env rewards that followed each bar (γ = ${C.gamma}), bootstrapped with V at the last bar. The greedy replay is not the policy the critic was trained on, so expect it lower than on a rollout. The V(s) label in 3D shows the realised R next to the prediction.`;
  }
  return {
    set(snap, history) { s = snap; C = criticCheck(s); hist = history || []; cur = -1; text(); draw(); },
    frame(f) { if (f !== cur) { cur = f; draw(); } },
    real(f) { return C && f >= 0 && f < C.R.length ? C.R[f] : null; },
  };
}
