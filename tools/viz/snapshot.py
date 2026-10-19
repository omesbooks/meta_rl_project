"""Neon-viewer snapshots of an actor-critic policy.

Two sources, one format (read by tools/viz/network3d.html + dashboard.js):

* **Eval replay** — `capture()`: after each evaluation, walk the current
  policy deterministically (argmax) over the validation slice.
  Written by rl_train.py --viz as viz/snap_NNN.json + index.json.
* **Training rollout** — `make_rollout_callback()`: after each PPO/A2C
  rollout, take the steps the agent ACTUALLY played from the rollout buffer —
  its sampled (exploring) actions on training data, the rewards it received
  and the advantages PPO is about to learn from — and re-run those
  observations through the network once (weights are not updated yet, so the
  activations are exactly the ones that produced the actions).
  Written as viz/rollout_latest.json + rollout_meta.json (overwritten).

Each snapshot holds per bar: OHLC + timestamp, action, greedy action,
position, equity, closed-trade pnl, every hidden activation of both towers,
action probabilities and V(s) (+ reward/advantage for rollouts); plus each
neuron's K most influential incoming connections, ranked by
|weight| x std(source) over the captured bars (raw |weight| over-draws inputs
on a tiny scale, e.g. the env's unrealized pnl, std ~0.005). Any net_arch,
activation and action profile; actor-critic policies only (PPO / A2C).
"""
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

SCHEMA = "metafxclub.viz_snapshot.v2"

SHORT_NAMES = {"hold": "HOLD", "buy": "BUY", "sell": "SELL", "close": "CLOSE",
               "move_sl_breakeven": "BE", "trail_sl_atr": "TRAIL"}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def action_names(profile_cfg):
    """Primitive names ("hold", "buy", ...) from a resolved action profile."""
    acts = (profile_cfg or {}).get("actions") or []
    return [a.get("name") if isinstance(a, dict) else str(a) for a in acts]


def supports(policy):
    return all(hasattr(policy, a) for a in ("mlp_extractor", "action_net", "value_net")) and \
        hasattr(policy.mlp_extractor, "policy_net") and hasattr(policy.mlp_extractor, "value_net")


def _record_points(seq):
    """Modules whose output is a hidden layer: the module right after each
    Linear (its activation) — or the Linear itself when nothing follows."""
    mods = list(seq)
    pts = []
    for i, m in enumerate(mods):
        if isinstance(m, torch.nn.Linear):
            j = i + 1 if i + 1 < len(mods) and not isinstance(mods[i + 1], torch.nn.Linear) else i
            pts.append(mods[j])
    return pts


def _forward(policy, obs):
    """One batched pass: hidden activations of both towers, action
    probabilities and V(s) for every row of `obs` (frames x inputs)."""
    towers = {"actor": policy.mlp_extractor.policy_net, "critic": policy.mlp_extractor.value_net}
    points = {name: _record_points(seq) for name, seq in towers.items()}
    acts, hooks = {}, []
    for name, pts in points.items():
        for li, m in enumerate(pts):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, key=(name, li): acts.__setitem__(key, out.detach().cpu().numpy())))
    device = getattr(policy, "device", torch.device("cpu"))
    try:
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
        with torch.no_grad():
            probs = policy.get_distribution(t).distribution.probs.cpu().numpy()
            values = policy.predict_values(t)[:, 0].cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    hidden = {name: [acts[(name, li)] for li in range(len(pts))] for name, pts in points.items()}
    tanh = {name: [isinstance(m, torch.nn.Tanh) for m in pts] for name, pts in points.items()}
    return towers, hidden, tanh, probs, values


EXTRA_INPUTS = ["position", "unrealized_pnl", "bars_in_position"]   # TradingEnv obs tail


def input_names(feature_names, window, n_in):
    """Name of every observation input in TradingEnv order: feature f at lag L (L = 0 is the
    current bar, L = window-1 the oldest), then the three state inputs."""
    nfeat = (n_in - 3) // window
    feats = list(feature_names or [])[:nfeat] or [f"f{i}" for i in range(nfeat)]
    out = []
    for r in range(window):
        lag = window - 1 - r
        out += [f"{f}@-{lag}" if lag else f"{f}@0" for f in feats]
    return out + EXTRA_INPUTS[:n_in - window * nfeat]


def attribution(policy, obs, actions):
    """Gradient x input of each frame's chosen-action logit with respect to the
    observation: how much every input pushed THAT decision (sign = direction).
    One batched forward/backward; frames are independent so summing the selected
    logits gives per-frame gradients."""
    device = getattr(policy, "device", torch.device("cpu"))
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).requires_grad_(True)
    feats = policy.extract_features(x)
    if isinstance(feats, tuple):
        feats = feats[0]
    logits = policy.action_net(policy.mlp_extractor.forward_actor(feats))
    idx = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=device).clamp(0, logits.shape[1] - 1)
    picked = logits.gather(1, idx[:, None]).sum()
    grad, = torch.autograd.grad(picked, x)
    return (grad * x).detach().cpu().numpy()


def attribution_summary(attr, window, n_in, feature_names=None, top=10):
    """Mean |attribution| share per feature (lags summed) and per lag, plus the top features."""
    nfeat = (n_in - 3) // window
    a = np.abs(np.asarray(attr, dtype=np.float64)).mean(axis=0)
    tot = a.sum() or 1.0
    per_feat = a[:window * nfeat].reshape(window, nfeat).sum(axis=0)
    extras = a[window * nfeat:]
    per_lag = a[:window * nfeat].reshape(window, nfeat).sum(axis=1)
    feats = list(feature_names or [])[:nfeat] or [f"f{i}" for i in range(nfeat)]
    names = feats + EXTRA_INPUTS[:len(extras)]
    shares = np.r_[per_feat, extras] / tot
    order = np.argsort(-shares)[:top]
    return {"attr_feature": [round(float(v), 5) for v in shares],
            "attr_lag": [round(float(v / tot), 5) for v in per_lag],          # index 0 = oldest bar
            "attr_top": [[names[i], round(float(shares[i]), 4)] for i in order]}


USAGE_REPEATS = 2           # shuffles per feature; the mean is stored
USAGE_STRONG = 0.05         # tiers the viewer paints (share of greedy decisions that change)
USAGE_USED = 0.01


def _greedy_probs(policy, obs):
    device = getattr(policy, "device", torch.device("cpu"))
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return policy.get_distribution(t).distribution.probs.cpu().numpy()


def feature_usage(policy, obs, window, repeats=USAGE_REPEATS, seed=0):
    """Permutation test: does the policy's DECISION depend on each input feature?

    For every feature (all its lags together, with one shuffle so its own history stays
    consistent) and each state input, shuffle its values across the frames, run the policy
    again and measure against the unshuffled pass:
      flip = share of frames whose greedy action changes,
      tv   = mean total-variation distance of the action probabilities (0..1).
    A feature whose shuffle never changes a decision is not used for these bars. It answers
    "does it matter" directly, where gradient x input only says how much it pushes locally.
    Returns (flip, tv) per feature in env order, then the state inputs."""
    obs = np.asarray(obs, dtype=np.float32)
    nf, n_in = obs.shape
    nfeat = (n_in - 3) // window
    groups = [np.arange(window) * nfeat + f for f in range(nfeat)]
    groups += [np.array([window * nfeat + e]) for e in range(n_in - window * nfeat)]
    base = _greedy_probs(policy, obs)
    g0 = base.argmax(axis=1)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(nf) for _ in range(max(1, int(repeats)))]
    flip, tv = np.zeros(len(groups)), np.zeros(len(groups))
    x = obs.copy()
    for gi, cols in enumerate(groups):
        for perm in perms:
            x[:, cols] = obs[perm][:, cols]
            pr = _greedy_probs(policy, x)
            flip[gi] += float((pr.argmax(axis=1) != g0).mean())
            tv[gi] += float(0.5 * np.abs(pr - base).sum(axis=1).mean())
        x[:, cols] = obs[:, cols]
    return flip / len(perms), tv / len(perms)


def pack(policy, obs, window, bars, k=16, names=None, meta=None, extra=None, feature_names=None,
         usage=True):
    """Build a snapshot dict from observations the policy saw.

    bars: dict of per-frame lists — ts, o, h, l, c, act, pos, eq, tpnl.
    extra: optional per-frame arrays added as-is (e.g. reward, adv).
    feature_names: env feature columns (for input attribution labels).
    usage: run the per-feature permutation test (feature_usage; eval snapshots only, the
    rollout writer turns it off to keep its overhead at ~1%).
    """
    if not supports(policy):
        raise ValueError("visual snapshots need an actor-critic policy (PPO / A2C)")
    obs = np.asarray(obs, dtype=np.float32)
    nf, n_in = obs.shape
    towers, hidden, tanh, probs, values = _forward(policy, obs)
    attr = attribution(policy, obs, bars["act"])
    attr_scale = np.abs(attr).max(axis=1); attr_scale[attr_scale == 0] = 1.0
    attr_q = np.round(attr / attr_scale[:, None] * 127).astype(np.int8)

    blocks = [np.clip(obs / 4.0, -1, 1)]                       # z-scores, clip at ±4
    sizes = {"actor": [], "critic": []}
    for name in ("actor", "critic"):
        for li, a in enumerate(hidden[name]):
            if not tanh[name][li]:                             # ReLU / linear: scale per layer
                a = a / max(1e-6, float(np.percentile(np.abs(a), 99)))
            blocks.append(np.clip(a, -1, 1))
            sizes[name].append(int(a.shape[1]))
    blocks.append(probs)
    vmax = float(np.abs(values).max()) or 1.0
    blocks.append((values / vmax)[:, None])
    q = np.round(np.concatenate(blocks, axis=1) * 127).astype(np.int8)

    # connections: each neuron's top-k by |w| x std(source) over these frames
    edges = {}
    obs_std = obs.std(axis=0)
    for name, head in (("actor", policy.action_net), ("critic", policy.value_net)):
        weights = [m.weight for m in towers[name] if isinstance(m, torch.nn.Linear)] + [head.weight]
        sources = [obs_std] + [a.std(axis=0) for a in hidden[name]]
        mats = []
        for W, src_std in zip(weights, sources):
            Wn = W.detach().cpu().numpy()
            kk = int(min(k, Wn.shape[1]))
            order = np.argsort(-(np.abs(Wn) * src_std[None, :]), axis=1, kind="stable")[:, :kk]
            mats.append({"rows": int(Wn.shape[0]), "cols": int(Wn.shape[1]), "k": kk,
                         "idx": b64(order.astype(np.int16)),
                         "w": b64(np.take_along_axis(Wn, order, 1).astype(np.float32))})
        edges[name] = mats

    n_actions = probs.shape[1]
    names = list(names or [])[:n_actions] or [f"A{i}" for i in range(n_actions)]
    snap = {
        "schema": SCHEMA, "frames": nf, "n_in": n_in, "window": int(window),
        "n_features": int((n_in - 3) // window), "actor_layers": sizes["actor"],
        "critic_layers": sizes["critic"], "actions": names,
        "action_labels": [SHORT_NAMES.get(n, n.upper()) for n in names],
        "width": int(q.shape[1]), "value_max": vmax, "q": b64(q),
        "ts": [str(x) for x in bars["ts"]],
        "o": [round(float(x), 5) for x in bars["o"]], "h": [round(float(x), 5) for x in bars["h"]],
        "l": [round(float(x), 5) for x in bars["l"]], "c": [round(float(x), 5) for x in bars["c"]],
        "act": [int(a) for a in bars["act"]], "greedy": [int(a) for a in probs.argmax(axis=1)],
        "pos": [int(p) for p in bars["pos"]], "eq": [round(float(e), 6) for e in bars["eq"]],
        "tpnl": [round(float(p), 6) for p in bars["tpnl"]],
        "edges": edges, "edge_rank": "abs_weight_x_source_std",
        # full actor head (actions x last hidden layer, ~1 KB): the viewer groups every last-layer
        # neuron by the action it feeds most; the top-k edges alone left about half ungrouped
        "head_w": b64(policy.action_net.weight.detach().cpu().numpy().astype(np.float32)),
        "head_shape": [int(d) for d in policy.action_net.weight.shape],
        "attr": b64(attr_q), "attr_scale": [round(float(v), 6) for v in attr_scale],
        "attr_method": "grad_x_input_of_chosen_action_logit",
        "input_names": input_names(feature_names, window, n_in),
        "meta": dict(meta or {}, created=time.strftime("%Y-%m-%d %H:%M:%S")),
    }
    snap.update(attribution_summary(attr, window, n_in, feature_names))
    if usage:
        t0 = time.perf_counter()
        flip, tv = feature_usage(policy, obs, window)
        snap.update({"use_flip": [round(float(v), 5) for v in flip], "use_tv": [round(float(v), 5) for v in tv],
                     "use_method": f"permute_feature_all_lags_x{USAGE_REPEATS}",
                     "use_tiers": {"strong": USAGE_STRONG, "used": USAGE_USED},
                     "use_s": round(time.perf_counter() - t0, 3)})
    for key, arr in (extra or {}).items():
        snap[key] = [round(float(x), 6) for x in arr]
    return snap


def capture(policy, env, window, frames, k=16, action_names=None, meta=None, feature_names=None):
    """Eval replay: walk `env` (a fresh, unwrapped TradingEnv on the slice to
    show) with the policy's argmax action and return the snapshot dict.

    The env's dataframe must still carry raw timestamp/open/high/low/close
    columns (feature normalisation leaves them untouched).
    """
    if not supports(policy):
        raise ValueError("visual snapshots need an actor-critic policy (PPO / A2C)")
    t_start = time.perf_counter()
    env.reset()
    env.t = window - 1
    env.start_t = env.t
    df = env.df
    obs = env._get_obs()
    device = getattr(policy, "device", torch.device("cpu"))
    rows, bar_idx, acts, pos, eq, tpnl, gates = [], [], [], [], [], [], []
    gate_arr = getattr(env, "_gate", None)
    for _ in range(int(frames)):
        t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            a = int(policy.get_distribution(t).distribution.probs[0].argmax())
        rows.append(obs.astype(np.float32))
        bar_idx.append(int(env.t))
        if gate_arr is not None:
            gates.append(float(gate_arr[env.t]))
        n_tr = len(env.trades)
        obs, _, term, trunc, _ = env.step(a)
        acts.append(a)
        pos.append(int(env.position))
        eq.append(float(env._curve[-1]))
        tpnl.append(float(env.trades[-1]["pnl"]) if len(env.trades) > n_tr else 0.0)
        if term or trunc:
            break
    bars = {"ts": df["timestamp"].astype(str).iloc[bar_idx].tolist(),
            "o": df["open"].iloc[bar_idx].tolist(), "h": df["high"].iloc[bar_idx].tolist(),
            "l": df["low"].iloc[bar_idx].tolist(), "c": df["close"].iloc[bar_idx].tolist(),
            "act": acts, "pos": pos, "eq": eq, "tpnl": tpnl}
    snap = pack(policy, np.stack(rows), window, bars, k=k, names=action_names,
                meta=dict(meta or {}, source="eval"), extra={"gate": gates} if gates else None,
                feature_names=feature_names or list(getattr(env, "feature_cols", []) or []))
    snap["meta"]["capture_s"] = round(time.perf_counter() - t_start, 3)
    return snap


def summary(snap, points=120):
    """Compact per-evaluation facts for index.json (the Dashboard reads these
    instead of downloading every snapshot): downsampled equity, action counts,
    trade stats, max drawdown on the slice."""
    eq = np.asarray(snap["eq"], dtype=float)
    n = len(eq)
    idx = np.linspace(0, n - 1, min(points, n)).round().astype(int) if n else np.array([], int)
    tp = np.asarray(snap["tpnl"], dtype=float)
    closed = tp[tp != 0]
    n_act = len(snap["actions"])
    counts = np.bincount(np.asarray(snap["act"], dtype=int), minlength=n_act)[:n_act]
    peak = np.maximum.accumulate(eq) if n else eq
    max_dd = float(((eq - peak) / peak).min()) if n else 0.0
    # behaviour: entries (flat -> position), hold bars (entry -> close), acceptance on gate bars
    pos = np.asarray(snap["pos"], dtype=int)
    prev = np.r_[0, pos[:-1]]
    entries = np.where((prev == 0) & (pos != 0))[0]
    exits = np.where(tp != 0)[0]
    holds = []
    for e in entries:
        nxt = exits[exits >= e]
        if len(nxt):
            holds.append(int(nxt[0] - e) + 1)
    out = {
        "eq_curve": [round(float(v), 5) for v in eq[idx]],
        "action_counts": [int(c) for c in counts],
        "trades": int(len(closed)), "wins": int((closed > 0).sum()),
        "win_rate": float((closed > 0).mean()) if len(closed) else None,
        "equity_end": float(eq[-1]) if n else 1.0, "max_dd": round(max_dd, 5),
        "entries": int(len(entries)), "bars": int(n),
        "hold_mean": round(float(np.mean(holds)), 2) if holds else None,
        "pnl_bp_mean": round(float(closed.mean()) * 1e4, 2) if len(closed) else None,
    }
    if snap.get("attr_top"):
        out["top_features"] = snap["attr_top"][:8]
    if snap.get("gate"):
        g = np.asarray(snap["gate"], dtype=float)
        sig = g != 0
        acc = sig & (prev == 0) & (pos != 0) & (np.sign(pos) == np.sign(g))
        out.update({"signal_bars": int(sig.sum()), "accepted": int(acc.sum()),
                    "acceptance": round(float(acc.sum()) / sig.sum(), 3) if sig.sum() else None})
    return out


def _atomic_write(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:      # Windows: the viewer's server has the file open
            if attempt == 19:
                raise
            time.sleep(0.05)


def write_live(viz_dir, i, snapshot, entry):
    """Store eval snapshot i and append its summary to index.json (atomic
    writes, so the live viewer never reads a half-written file)."""
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    name = f"snap_{i:03d}.json"
    _atomic_write(viz_dir / name, json.dumps(snapshot))
    idx_path = viz_dir / "index.json"
    try:
        index = json.loads(idx_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {"schema": SCHEMA, "snapshots": []}
    index["snapshots"] = [s for s in index.get("snapshots", []) if s.get("i") != i]
    index["snapshots"].append(dict(entry, i=i, file=name))
    index["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write(idx_path, json.dumps(index))
    return viz_dir / name


def write_rollout(viz_dir, snapshot, seq):
    """Overwrite the latest training-rollout snapshot; rollout_meta.json is the
    tiny file the viewer polls (so it never re-downloads an unchanged rollout)."""
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(viz_dir / "rollout_latest.json", json.dumps(snapshot))
    m = snapshot.get("meta", {})
    _atomic_write(viz_dir / "rollout_meta.json", json.dumps({
        "seq": seq, "step": m.get("step"), "total_steps": m.get("total_steps"),
        "file": "rollout_latest.json", "updated": time.strftime("%Y-%m-%d %H:%M:%S")}))
    return viz_dir / "rollout_latest.json"


# ------------------------------------------------------------------ greedy walks: decision stability + train/validation gap
STABILITY_FILE = "stability.json"
STABILITY_MAX_EVALS = 400
SETTLED_CHANGED = 0.10    # "settled" = every later evaluation changed <= 10% of the greedy decisions
GAP_WINDOW = 4            # evaluations the trend verdict looks back over
GAP_PP = 0.5              # percentage points of slice return that count as a move
GAP_MEMO_PP = 5.0         # train - validation gap that counts as memorising (with validation <= 0)
GAP_PEAK_PP = 1.0         # validation below its best by this much, >= 3 evals later = peaked


def greedy_walk(policy, env, frames):
    """Deterministic (argmax) walk of `frames` bars on a fresh, unwrapped TradingEnv.
    Per bar: action probabilities, greedy action, position, equity, closed-trade pnl and
    reward — no network capture, so it is cheap enough to run on two slices after every
    evaluation (~1 ms per bar on CPU)."""
    env.reset()
    env.t = env.window_size - 1
    env.start_t = env.t
    obs = env._get_obs()
    device = getattr(policy, "device", torch.device("cpu"))
    probs, acts, pos, eq, tpnl, rew, bar_idx = [], [], [], [], [], [], []
    for _ in range(int(frames)):
        t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            p = policy.get_distribution(t).distribution.probs[0].cpu().numpy().astype(np.float32)
        a = int(p.argmax())
        n_tr = len(env.trades)
        bar_idx.append(int(env.t))
        obs, r, term, trunc, _ = env.step(a)
        probs.append(p); acts.append(a); pos.append(int(env.position)); eq.append(float(env._curve[-1]))
        tpnl.append(float(env.trades[-1]["pnl"]) if len(env.trades) > n_tr else 0.0); rew.append(float(r))
        if term or trunc:
            break
    df = env.df
    out = {"probs": np.stack(probs) if probs else np.zeros((0, 1), np.float32),
           "act": np.asarray(acts, dtype=int), "pos": np.asarray(pos, dtype=int), "eq": np.asarray(eq, dtype=float),
           "tpnl": np.asarray(tpnl, dtype=float), "reward": np.asarray(rew, dtype=float), "bar_idx": bar_idx,
           "bars": {"ts": df["timestamp"].astype(str).iloc[bar_idx].tolist(),
                    "o": [round(float(x), 5) for x in df["open"].iloc[bar_idx]],
                    "h": [round(float(x), 5) for x in df["high"].iloc[bar_idx]],
                    "l": [round(float(x), 5) for x in df["low"].iloc[bar_idx]],
                    "c": [round(float(x), 5) for x in df["close"].iloc[bar_idx]]}}
    out["stats"] = walk_stats(out["eq"], out["tpnl"], out["reward"], out["pos"])
    return out


def walk_stats(eq, tpnl, reward, pos):
    """Slice-level facts of one greedy walk. gross_* are in bp so JS can rebuild PF (never
    Infinity in the JSON: PF is None when there is no losing trade)."""
    eq = np.asarray(eq, dtype=float); tpnl = np.asarray(tpnl, dtype=float); pos = np.asarray(pos, dtype=int)
    n = len(eq)
    closed = tpnl[tpnl != 0]
    gw = float(closed[closed > 0].sum()); gl = float(-closed[closed < 0].sum())
    peak = np.maximum.accumulate(eq) if n else eq
    prev = np.r_[0, pos[:-1]] if n else pos
    return {
        "bars": int(n), "return_pct": round(float(eq[-1] - 1) * 100, 4) if n else 0.0,
        "trades": int(len(closed)), "wins": int((closed > 0).sum()),
        "win_rate": round(float((closed > 0).mean()), 4) if len(closed) else None,
        "gross_win_bp": round(gw * 1e4, 2), "gross_loss_bp": round(gl * 1e4, 2),
        "pf": round(gw / gl, 4) if gl > 0 else None,
        "pnl_bp_mean": round(float(closed.mean()) * 1e4, 2) if len(closed) else None,
        "reward_sum": round(float(np.sum(reward)), 4), "reward_per_bar": round(float(np.mean(reward)), 6) if n else 0.0,
        "max_dd_pct": round(float(((eq - peak) / peak).min()) * 100, 4) if n else 0.0,
        "entries": int(((prev == 0) & (pos != 0)).sum()), "in_position": round(float((pos != 0).mean()), 4) if n else 0.0,
    }


def stability_row(i, step, walk, prev_greedy=None):
    """One evaluation's decisions on the fixed validation slice: probabilities (uint8),
    greedy action and position per bar, plus how many greedy decisions changed since the
    previous evaluation, mean confidence (max p) and entropy."""
    P = np.asarray(walk["probs"], dtype=np.float64)
    greedy = np.asarray(walk["act"], dtype=int)
    n = len(greedy)
    ent = float(-(P * np.log(P + 1e-12)).sum(axis=1).mean()) if n else 0.0
    changed = None
    if prev_greedy is not None and len(prev_greedy) == n and n:
        changed = round(float((greedy != np.asarray(prev_greedy)).mean()), 4)
    st = walk["stats"]
    return {"i": int(i), "step": int(step), "bars": int(n),
            "p": b64(np.round(np.clip(P, 0, 1) * 255).astype(np.uint8)), "n_actions": int(P.shape[1]) if n else 0,
            "greedy": greedy.tolist(), "pos": [int(x) for x in walk["pos"]],
            "changed": changed, "entropy": round(ent, 4), "confidence": round(float(P.max(axis=1).mean()), 4) if n else None,
            "in_position": st["in_position"], "trades": st["trades"], "return_pct": st["return_pct"]}


def settled_since(rows, thresh=SETTLED_CHANGED):
    """Index of the evaluation after which every later evaluation changed <= `thresh` of
    the decisions (needs at least 2 later evaluations); None when not settled."""
    ch = [r.get("changed") for r in rows]
    for k in range(len(rows) - 2):
        later = ch[k + 1:]
        if later and all(c is not None and c <= thresh for c in later):
            return int(rows[k]["i"])
    return None


def write_stability(viz_dir, row, bars=None, actions=None, meta=None):
    """Append one evaluation to stability.json (bars + action names stored once)."""
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    path = viz_dir / STABILITY_FILE
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {"schema": SCHEMA, "evals": []}
    if bars is not None and not st.get("bars"):
        st["bars"] = bars
    if actions is not None:
        st["actions"] = list(actions)
    st.update(meta or {})
    st["evals"] = sorted([e for e in st.get("evals", []) if e.get("i") != row["i"]] + [row], key=lambda e: e["i"])[-STABILITY_MAX_EVALS:]
    st["settled_since"] = settled_since(st["evals"])
    st["settled_thresh"] = SETTLED_CHANGED
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write(path, json.dumps(st))
    return path


def gap_verdict(gaps):
    """'Stop or keep going?' from the greedy returns on the fixed Train slice vs the
    validation slice, one pair per evaluation. gaps: [{i, step, train: {return_pct}, val: {return_pct}}].
    Returns {cls, t, d} like the Monitor chips; the text carries the numbers and thresholds."""
    n = len(gaps)
    if n < 3:
        return {"cls": "grey", "t": "TRAIN vs VALIDATION: COLLECTING", "d": f"{n} evaluation(s) so far · the trend verdict needs 3"}
    W = gaps[-GAP_WINDOW:]
    tr = [float(g["train"]["return_pct"]) for g in W]
    va = [float(g["val"]["return_pct"]) for g in W]
    d_tr, d_va, gap = tr[-1] - tr[0], va[-1] - va[0], tr[-1] - va[-1]
    all_va = [float(g["val"]["return_pct"]) for g in gaps]
    best_i = int(np.argmax(all_va)); best = all_va[best_i]
    k = len(W)
    if d_tr >= GAP_PP and d_va <= -GAP_PP:
        return {"cls": "red", "t": "GAP WIDENING — OVERFIT?",
                "d": f"over the last {k} evals train {tr[0]:+.1f}% → {tr[-1]:+.1f}% while validation {va[0]:+.1f}% → {va[-1]:+.1f}% · "
                     f"threshold train ≥ +{GAP_PP} pp and validation ≤ −{GAP_PP} pp · the best/ checkpoint keeps the best validation eval"}
    if d_va >= GAP_PP:
        return {"cls": "green", "t": "GENERALISING",
                "d": f"validation {va[0]:+.1f}% → {va[-1]:+.1f}% over the last {k} evals · train {tr[-1]:+.1f}% · gap {gap:+.1f} pp · threshold validation ≥ +{GAP_PP} pp"}
    if n - 1 - best_i >= 3 and best - va[-1] >= GAP_PEAK_PP:
        return {"cls": "amber", "t": f"VALIDATION PEAKED AT EVAL {best_i + 1}",
                "d": f"best {best:+.1f}% at step {int(gaps[best_i]['step']):,}, now {va[-1]:+.1f}% ({n - 1 - best_i} evals later) · "
                     f"threshold ≥ 3 evals and ≥ {GAP_PEAK_PP} pp below the best · the best/ checkpoint holds that eval's policy"}
    if gap >= GAP_MEMO_PP and va[-1] <= 0:
        return {"cls": "amber", "t": "MEMORISING THE TRAIN SLICE",
                "d": f"train {tr[-1]:+.1f}% vs validation {va[-1]:+.1f}% (gap {gap:+.1f} pp) · threshold gap ≥ {GAP_MEMO_PP} pp with validation ≤ 0"}
    return {"cls": "grey", "t": "NO CLEAR TREND",
            "d": f"train {d_tr:+.1f} pp, validation {d_va:+.1f} pp over the last {k} evals · within ±{GAP_PP} pp · train {tr[-1]:+.1f}% vs validation {va[-1]:+.1f}%"}


def _segment(starts, n, frames, min_len=150):
    """Most recent contiguous episode stretch of the rollout, capped at
    `frames` bars (an episode reset would make the price chart jump)."""
    begins = [0] + [i for i in range(1, n) if starts[i]]
    last = begins[-1]
    if n - last >= min_len or len(begins) == 1:
        return max(last, n - frames), n
    prev = begins[-2]
    return max(prev, last - frames), last


MONITOR_ROWS = 1000       # rollouts kept in monitor.json
CF_EXIT_BARS = 4          # fixed-rule counterfactual: enter on the signal bar, exit N bars later, no stop


def monitor_row(seq, step, total_steps, rec, aux, df, env, n_actions):
    """One row of training-behaviour diagnostics for a whole rollout.

    rec: [(t, pos_after, equity_after, close_pnl, reward)], aux: [(action, gate, prev_pos, close_info)]
    where close_info is None or (reason, bars, pnl) for a trade that closed on that step.
    The counterfactual takes every gate (signal) bar of the rollout in the signal's
    direction, exits CF_EXIT_BARS bars later at the close with the env's spread and
    commission, and splits the bars into those the agent accepted (opened in the
    signal direction on that bar) and those it skipped."""
    n = len(rec)
    acts = np.array([a[0] for a in aux], dtype=int)
    gate = np.array([a[1] if a[1] is not None else 0.0 for a in aux], dtype=float)
    has_gate = any(a[1] is not None for a in aux)
    prev_pos = np.array([a[2] for a in aux], dtype=int)
    pos = np.array([r[1] for r in rec], dtype=int)
    opened = (prev_pos == 0) & (pos != 0)
    closes = [a[3] for a in aux if a[3] is not None]
    pnls = np.array([c[2] for c in closes], dtype=float)
    holds = np.array([c[1] for c in closes], dtype=float)
    exits = {}
    for c in closes:
        exits[str(c[0])] = exits.get(str(c[0]), 0) + 1
    counts = np.bincount(acts[(acts >= 0) & (acts < n_actions)], minlength=n_actions).astype(float)
    pr = counts / max(1.0, counts.sum())
    entropy = float(-np.sum(pr[pr > 0] * np.log(pr[pr > 0])))
    row = {
        "seq": int(seq), "step": int(step), "total_steps": int(total_steps or 0), "bars": int(n),
        "trades_opened": int(opened.sum()), "trades_closed": int(len(closes)),
        "trades_per_100": round(100.0 * opened.sum() / max(1, n), 2),
        "hold_mean": round(float(holds.mean()), 2) if len(holds) else None,
        "exits": exits,
        "pnl_mean_bp": round(float(pnls.mean()) * 1e4, 2) if len(pnls) else None,
        "win_rate": round(float((pnls > 0).mean()), 3) if len(pnls) else None,
        "gross_win": round(float(pnls[pnls > 0].sum()) * 1e4, 2), "gross_loss": round(float(-pnls[pnls < 0].sum()) * 1e4, 2),
        "action_mix": counts.astype(int).tolist(), "action_entropy": round(entropy, 3),
        "reward_sum": round(float(sum(r[4] for r in rec)), 4),
        "gate": bool(has_gate),
    }
    if has_gate:
        ts = np.array([r[0] for r in rec], dtype=int)
        sig = gate != 0
        sgn = np.sign(gate)
        accepted = sig & opened & (np.sign(pos) == sgn)
        buy_id = env.action_ids.get("buy"); sell_id = env.action_ids.get("sell")
        attempt = (acts == buy_id) | (acts == sell_id)
        wrong = attempt & (prev_pos == 0) & ~((acts == buy_id) & (gate > 0)) & ~((acts == sell_id) & (gate < 0))
        close = df["close"].to_numpy(float)
        cost = 2.0 * float(env.spread) + float(env.commission)
        cf = np.full(n, np.nan)
        ok = sig & (ts + CF_EXIT_BARS < len(close))
        cf[ok] = sgn[ok] * (close[ts[ok] + CF_EXIT_BARS] - close[ts[ok]]) / close[ts[ok]] - cost
        def m(mask):
            v = cf[mask & np.isfinite(cf)]
            return (round(float(v.mean()) * 1e4, 2) if len(v) else None, int(len(v)))
        row.update({
            "signal_bars": int(sig.sum()), "signal_share": round(float(sig.mean()), 4),
            "accepted": int(accepted.sum()), "acceptance": round(float(accepted.sum()) / max(1, sig.sum()), 3),
            "blocked_attempts": int(wrong.sum()),
            "cf_all_bp": m(sig)[0], "cf_accepted_bp": m(accepted)[0], "cf_accepted_n": m(accepted)[1],
            "cf_skipped_bp": m(sig & ~accepted)[0], "cf_skipped_n": m(sig & ~accepted)[1],
        })
    return row


def write_monitor(viz_dir, row, extra=None):
    """Append one rollout row to monitor.json (small file the viewer polls)."""
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    path = viz_dir / "monitor.json"
    try:
        mon = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        mon = {"schema": SCHEMA, "rows": []}
    mon["rows"] = [r for r in mon.get("rows", []) if r.get("seq") != row["seq"]] + [row]
    mon["rows"] = mon["rows"][-MONITOR_ROWS:]
    mon.update(extra or {})
    mon["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write(path, json.dumps(mon))
    return path


def make_rollout_callback(viz_dir, window, names=None, frames=400, min_interval=4.0,
                          total_steps=None, model_name="", symbol="", gate_name="", feature_names=None):
    """SB3 callback: after every rollout (at most every `min_interval` s) write
    the real training stretch the agent just played. Prints
    `VIZ_ROLLOUT <seq> <step> <path>` for the first write and every 10th."""
    from stable_baselines3.common.callbacks import BaseCallback

    class RolloutViz(BaseCallback):
        def __init__(self):
            super().__init__()
            self.seq = 0            # heavy snapshot counter
            self.rollouts = 0       # every rollout (monitor rows)
            self.last_write = 0.0
            self.rec = []
            self.aux = []
            self.pre = None
            self.prev_ntr = 0

        def _env(self):
            e = self.training_env.envs[0]
            return getattr(e, "unwrapped", e)

        def _state(self):
            env = self._env()
            return int(env.t), float(env.equity)

        def _on_rollout_start(self):
            self.rec = []
            self.aux = []
            self.pre = self._state()
            self.prev_ntr = len(getattr(self._env(), "trades", []))

        def _on_step(self):
            # env.step has run (and an auto-reset, if the episode ended): `pre` is
            # the state the agent decided in; the info describes the outcome
            t, eq_before = self.pre
            info = self.locals["infos"][0]
            eq_after = float(info.get("equity", eq_before))
            # TradingEnv equity only moves when a trade closes
            pnl = eq_after / eq_before - 1.0 if abs(eq_after - eq_before) > 1e-12 else 0.0
            self.rec.append((t, int(info.get("position", 0)), eq_after, pnl,
                             float(np.asarray(self.locals["rewards"]).ravel()[0])))
            env = self._env()
            act = np.asarray(self.locals["actions"]).reshape(-1)
            action = int(round(float(act[0])))                       # hybrid: [index, sl]
            gate = float(env._gate[t]) if getattr(env, "_gate", None) is not None else None
            prev_pos = self.rec[-2][1] if len(self.rec) > 1 else 0
            ntr = int(info.get("n_trades", len(getattr(env, "trades", []))))
            close_info = None
            if ntr > self.prev_ntr and getattr(env, "trades", None):
                tr = env.trades[-1]
                close_info = (tr.get("reason", "?"), int(tr.get("bars", 0)), float(tr.get("pnl", 0.0)))
            elif ntr < self.prev_ntr and pnl != 0.0:                 # closed by an episode reset: trades list is gone
                close_info = ("episode_end", 0, float(pnl))
            self.prev_ntr = ntr
            self.aux.append((action, gate, prev_pos, close_info))
            self.pre = self._state()
            return True

        def _on_rollout_end(self):
            now = time.time()
            if self.rec:
                try:                                                  # cheap, every rollout
                    env = self._env()
                    row = monitor_row(self.rollouts, int(self.num_timesteps), total_steps, self.rec, self.aux,
                                      env.df, env, len(env.action_profile["actions"]))
                    try:                                              # which inputs drove this rollout's decisions
                        buf = self.model.rollout_buffer
                        nb = min(len(self.rec), buf.buffer_size if buf.full else buf.pos)
                        ob = np.asarray(buf.observations)[:nb, 0].reshape(nb, -1)
                        ac = np.asarray(buf.actions)[:nb, 0].reshape(nb, -1)[:, 0]
                        fn = feature_names or list(getattr(env, "feature_cols", []) or [])
                        row.update(attribution_summary(attribution(self.model.policy, ob, ac), window, ob.shape[1], fn))
                        row.pop("attr_feature", None)                 # keep the row small: lags + top features only
                    except Exception as exc:
                        row["attr_error"] = str(exc)[:120]
                    write_monitor(viz_dir, row, {"model": model_name, "symbol": symbol,
                                                 "actions": list(names or env.action_profile["actions"]),
                                                 "entry_gate": gate_name or getattr(env, "entry_gate", ""),
                                                 "max_hold": int(env.max_hold_bars), "cf_exit_bars": CF_EXIT_BARS})
                except Exception as exc:
                    print(f"\n[viz] monitor row failed: {exc}", flush=True)
                self.rollouts += 1
            if now - self.last_write < min_interval or not self.rec:
                return
            try:
                buf = self.model.rollout_buffer
                n = min(len(self.rec), buf.buffer_size if buf.full else buf.pos)
                starts = np.asarray(buf.episode_starts)[:n, 0]
                a, b = _segment(starts, n, frames)
                rec = self.rec[a:b]
                df = self._env().df
                tix = [r[0] for r in rec]
                bars = {"ts": df["timestamp"].astype(str).iloc[tix].tolist(),
                        "o": df["open"].iloc[tix].tolist(), "h": df["high"].iloc[tix].tolist(),
                        "l": df["low"].iloc[tix].tolist(), "c": df["close"].iloc[tix].tolist(),
                        "act": np.asarray(buf.actions)[a:b, 0].reshape(-1).astype(int).tolist(),
                        "pos": [r[1] for r in rec], "eq": [r[2] for r in rec], "tpnl": [r[3] for r in rec]}
                obs = np.asarray(buf.observations)[a:b, 0].reshape(b - a, -1)
                rewards = [r[4] for r in rec]
                adv = np.asarray(buf.advantages)[a:b, 0].reshape(-1)
                snap = pack(self.model.policy, obs, window, bars, names=names,
                            meta=dict(source="rollout", slice="train (exploring)", model=model_name,
                                      symbol=symbol, step=int(self.num_timesteps),
                                      total_steps=int(total_steps or 0), rollout=self.seq,
                                      reward_sum=round(float(np.sum(rewards)), 4)),
                            extra={"reward": rewards, "adv": adv}, usage=False,
                            feature_names=feature_names or list(getattr(self._env(), "feature_cols", []) or []))
                snap["meta"]["explore"] = round(
                    float(np.mean(np.array(snap["act"]) != np.array(snap["greedy"]))), 4)
                path = write_rollout(viz_dir, snap, self.seq)
                if self.seq % 10 == 0:
                    print(f"\nVIZ_ROLLOUT {self.seq} {self.num_timesteps} {path}", flush=True)
                self.seq += 1
                self.last_write = now
            except Exception as exc:                      # a visual must never stop training
                print(f"\n[viz] rollout snapshot failed: {exc}", flush=True)
                self.last_write = now

    return RolloutViz()
