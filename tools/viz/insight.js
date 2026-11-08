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
      t: `${names(pinned)} pinned at ±1 (${names([worst])} ${pct(worst.satShare)}): tanh gradients vanish there, so the layer acts as on/off switches and hardly learns any more. `
       + `Try activation_fn=ReLU, a lower learning rate or a smaller window, and A/B against this run.` });
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
