"""Neon-viewer snapshots of an actor-critic policy.

Shared by tools/viz/capture_snapshot.py (one-off, after training) and
rl_train.py --viz (one snapshot after every evaluation, read live by
tools/viz/live_viewer.py).

A snapshot walks the policy deterministically over a slice of bars and records,
per bar: OHLC + timestamp, the argmax action, position, equity, closed-trade
pnl, every hidden activation of both towers (Actor + Critic), the action
probabilities and V(s); plus the K strongest incoming weights of every neuron.
Works for any net_arch depth/width, any activation function and any action
profile. Only actor-critic policies (PPO / A2C) are supported — DQN has a
single Q-network and no Actor/Critic split.
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


def capture(policy, env, window, frames, k=16, action_names=None, meta=None):
    """Walk `env` (a fresh, unwrapped TradingEnv on the slice to show) with the
    policy's argmax action and return the snapshot dict (JSON-serialisable).

    The env's dataframe must still carry raw timestamp/open/high/low/close
    columns (feature normalisation leaves them untouched).
    """
    if not supports(policy):
        raise ValueError("visual snapshots need an actor-critic policy (PPO / A2C)")
    t_start = time.perf_counter()
    towers = {"actor": policy.mlp_extractor.policy_net, "critic": policy.mlp_extractor.value_net}
    points = {name: _record_points(seq) for name, seq in towers.items()}
    tanh = {name: [isinstance(m, torch.nn.Tanh) for m in pts] for name, pts in points.items()}
    acts = {}
    hooks = []
    for name, pts in points.items():
        for li, m in enumerate(pts):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, key=(name, li): acts.__setitem__(key, out.detach())))

    env.reset()
    env.t = window - 1
    env.start_t = env.t
    df = env.df
    obs = env._get_obs()
    n_in = len(obs)
    device = getattr(policy, "device", torch.device("cpu"))
    obs_rows, hid = [], {key: [] for key in ((n, i) for n, p in points.items() for i in range(len(p)))}
    probs_rows, values, bar_idx, actions, position, equity, trade_pnl = [], [], [], [], [], [], []
    try:
        for _ in range(int(frames)):
            t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = policy.get_distribution(t).distribution.probs[0].cpu().numpy()
                value = float(policy.predict_values(t)[0, 0])
            obs_rows.append(obs.astype(np.float32))
            for key in hid:
                hid[key].append(acts[key][0].cpu().numpy())
            probs_rows.append(probs)
            values.append(value)
            bar_idx.append(int(env.t))
            a = int(np.argmax(probs))
            n_tr = len(env.trades)
            obs, _, term, trunc, _ = env.step(a)
            actions.append(a)
            position.append(int(env.position))
            equity.append(float(env._curve[-1]))
            trade_pnl.append(float(env.trades[-1]["pnl"]) if len(env.trades) > n_tr else 0.0)
            if term or trunc:
                break
    finally:
        for h in hooks:
            h.remove()

    nf = len(obs_rows)
    blocks = [np.clip(np.stack(obs_rows) / 4.0, -1, 1)]          # z-scores, clip at ±4
    sizes = {"actor": [], "critic": []}
    for name in ("actor", "critic"):
        for li in range(len(points[name])):
            a = np.stack(hid[(name, li)])
            if not tanh[name][li]:                               # ReLU / linear: scale per layer
                a = a / max(1e-6, float(np.percentile(np.abs(a), 99)))
            blocks.append(np.clip(a, -1, 1))
            sizes[name].append(int(a.shape[1]))
    blocks.append(np.stack(probs_rows))
    vals = np.array(values, dtype=np.float32)
    vmax = float(np.abs(vals).max()) or 1.0
    blocks.append((vals / vmax)[:, None])
    q = np.round(np.concatenate(blocks, axis=1) * 127).astype(np.int8)

    edges = {}
    for name, seq, head in (("actor", towers["actor"], policy.action_net),
                            ("critic", towers["critic"], policy.value_net)):
        mats = []
        for W in [m.weight for m in seq if isinstance(m, torch.nn.Linear)] + [head.weight]:
            Wn = W.detach().cpu().numpy()
            kk = int(min(k, Wn.shape[1]))
            order = np.argsort(-np.abs(Wn), axis=1)[:, :kk]
            mats.append({"rows": int(Wn.shape[0]), "cols": int(Wn.shape[1]), "k": kk,
                         "idx": b64(order.astype(np.int16)),
                         "w": b64(np.take_along_axis(Wn, order, 1).astype(np.float32))})
        edges[name] = mats

    n_actions = len(probs_rows[0])
    names = list(action_names or [])[:n_actions] or [f"A{i}" for i in range(n_actions)]
    ts = df["timestamp"].astype(str)
    col = lambda c: [round(float(df[c].iloc[i]), 5) for i in bar_idx]
    return {
        "schema": SCHEMA, "frames": nf, "n_in": n_in, "window": int(window),
        "n_features": int((n_in - 3) // window), "actor_layers": sizes["actor"],
        "critic_layers": sizes["critic"], "actions": names,
        "action_labels": [SHORT_NAMES.get(n, n.upper()) for n in names],
        "width": int(q.shape[1]), "value_max": vmax, "q": b64(q),
        "ts": [ts.iloc[i] for i in bar_idx], "o": col("open"), "h": col("high"),
        "l": col("low"), "c": col("close"), "act": actions, "pos": position,
        "eq": [round(e, 6) for e in equity], "tpnl": [round(p, 6) for p in trade_pnl],
        "edges": edges, "meta": dict(meta or {}, capture_s=round(time.perf_counter() - t_start, 3),
                                     created=time.strftime("%Y-%m-%d %H:%M:%S")),
    }


def _atomic_write(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_live(viz_dir, i, snapshot, entry):
    """Store snapshot i and append its summary to index.json (atomic writes,
    so the live viewer never reads a half-written file)."""
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
