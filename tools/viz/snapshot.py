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


def pack(policy, obs, window, bars, k=16, names=None, meta=None, extra=None):
    """Build a snapshot dict from observations the policy saw.

    bars: dict of per-frame lists — ts, o, h, l, c, act, pos, eq, tpnl.
    extra: optional per-frame arrays added as-is (e.g. reward, adv).
    """
    if not supports(policy):
        raise ValueError("visual snapshots need an actor-critic policy (PPO / A2C)")
    obs = np.asarray(obs, dtype=np.float32)
    nf, n_in = obs.shape
    towers, hidden, tanh, probs, values = _forward(policy, obs)

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
        "meta": dict(meta or {}, created=time.strftime("%Y-%m-%d %H:%M:%S")),
    }
    for key, arr in (extra or {}).items():
        snap[key] = [round(float(x), 6) for x in arr]
    return snap


def capture(policy, env, window, frames, k=16, action_names=None, meta=None):
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
    rows, bar_idx, acts, pos, eq, tpnl = [], [], [], [], [], []
    for _ in range(int(frames)):
        t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            a = int(policy.get_distribution(t).distribution.probs[0].argmax())
        rows.append(obs.astype(np.float32))
        bar_idx.append(int(env.t))
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
                meta=dict(meta or {}, source="eval"))
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
    return {
        "eq_curve": [round(float(v), 5) for v in eq[idx]],
        "action_counts": [int(c) for c in counts],
        "trades": int(len(closed)), "wins": int((closed > 0).sum()),
        "win_rate": float((closed > 0).mean()) if len(closed) else None,
        "equity_end": float(eq[-1]) if n else 1.0, "max_dd": round(max_dd, 5),
    }


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


def _segment(starts, n, frames, min_len=150):
    """Most recent contiguous episode stretch of the rollout, capped at
    `frames` bars (an episode reset would make the price chart jump)."""
    begins = [0] + [i for i in range(1, n) if starts[i]]
    last = begins[-1]
    if n - last >= min_len or len(begins) == 1:
        return max(last, n - frames), n
    prev = begins[-2]
    return max(prev, last - frames), last


def make_rollout_callback(viz_dir, window, names=None, frames=400, min_interval=4.0,
                          total_steps=None, model_name="", symbol=""):
    """SB3 callback: after every rollout (at most every `min_interval` s) write
    the real training stretch the agent just played. Prints
    `VIZ_ROLLOUT <seq> <step> <path>` for the first write and every 10th."""
    from stable_baselines3.common.callbacks import BaseCallback

    class RolloutViz(BaseCallback):
        def __init__(self):
            super().__init__()
            self.seq = 0
            self.last_write = 0.0
            self.rec = []
            self.pre = None

        def _env(self):
            e = self.training_env.envs[0]
            return getattr(e, "unwrapped", e)

        def _state(self):
            env = self._env()
            return int(env.t), float(env.equity)

        def _on_rollout_start(self):
            self.rec = []
            self.pre = self._state()

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
            self.pre = self._state()
            return True

        def _on_rollout_end(self):
            now = time.time()
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
                            extra={"reward": rewards, "adv": adv})
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
