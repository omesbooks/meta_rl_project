"""Capture a visual snapshot of a trained PPO policy for the neon 3D viewer.

Walks the model deterministically over a slice of its own dataset (default:
Validation) and records, per bar: OHLC, action, position, equity, closed-trade
pnl, every hidden-layer activation of both towers (Actor + Critic), the action
probabilities and V(s); plus the strongest incoming weights of every neuron.
The data is inlined into tools/viz/network3d.html -> one self-contained HTML
(Three.js loads from the jsDelivr CDN, so the first open needs internet).

NOTE: Validation is the slice used to pick the best checkpoint, so the equity
shown is a visual, not evidence of edge (use --slice test for unseen bars).

Usage (from the repo root):
    python tools/viz/capture_snapshot.py <model> [--frames 400] [--slice validation|test]
                                         [--offset 0] [--k 16] [--source final|best] [--out <html>]
Output default: artifacts/viz/<model>_network3d.html
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from artifact_paths import best_dir, final_model_path, train_meta_path  # noqa: E402
from training_data import apply_normalization, load_dataset  # noqa: E402
from trading_env import TradingEnv  # noqa: E402

TEMPLATE = Path(__file__).with_name("network3d.html")


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser(description="Capture a neon-viewer snapshot of a trained model")
    ap.add_argument("model")
    ap.add_argument("--frames", type=int, default=400, help="bars to walk (default 400)")
    ap.add_argument("--slice", choices=["validation", "test"], default="validation")
    ap.add_argument("--offset", type=int, default=0, help="bars to skip at the start of the slice")
    ap.add_argument("--k", type=int, default=16, help="strongest incoming weights kept per neuron")
    ap.add_argument("--source", choices=["final", "best"], default="final")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta = json.loads(Path(train_meta_path(args.model)).read_text(encoding="utf-8-sig"))
    features = meta["features"]
    hp = meta["hyperparameters"]
    window = int(hp["window"])
    zip_path = (Path(best_dir(args.model)) / "best_model.zip") if args.source == "best" \
        else Path(final_model_path(args.model))
    if not zip_path.is_file():
        sys.exit(f"model not found: {zip_path}")
    model = PPO.load(str(zip_path), device="cpu")
    pol = model.policy
    norm = pd.read_csv(Path(final_model_path(args.model)).with_name(f"{args.model}_norm.csv"), index_col=0)

    csv = Path(meta["train_csv"])
    csv = csv if csv.is_absolute() else ROOT / csv
    df = load_dataset(str(csv))
    s0 = int(meta["split_index"])
    v_rows = int(meta["validation_rows"])
    if args.slice == "validation":
        part = df.iloc[s0:s0 + v_rows]
    else:
        part = df.iloc[s0 + v_rows:]
    part = part.iloc[args.offset:].reset_index(drop=True)
    need = window + 4
    if len(part) < need:
        sys.exit(f"slice too short: {len(part)} rows")
    raw = part.copy()
    part = apply_normalization(part, features, norm)

    env = TradingEnv(part, features, window_size=window, random_start=False,
                     max_steps=len(part) - window - 3,
                     max_hold_bars=int(hp.get("max_hold", 35)),
                     reward_mode=hp.get("reward_mode", "realized"),
                     reward_profile=hp.get("reward_profile", "balanced"))
    env.reset()
    env.t = window - 1
    env.start_t = env.t

    acts = {}
    hooks = []
    for tower in ("policy_net", "value_net"):
        for i, layer in enumerate(getattr(pol.mlp_extractor, tower)):
            if isinstance(layer, torch.nn.Tanh):
                hooks.append(layer.register_forward_hook(
                    lambda m, inp, out, n=f"{tower}.{i}": acts.__setitem__(n, out.detach())))

    obs = env._get_obs()
    n_in = len(obs)
    rows, bar_idx, actions, position, equity, trade_pnl = [], [], [], [], [], []
    t0 = time.perf_counter()
    for _ in range(args.frames):
        t = torch.as_tensor(obs[None], dtype=torch.float32)
        with torch.no_grad():
            probs = pol.get_distribution(t).distribution.probs[0].numpy()
            value = float(pol.predict_values(t)[0, 0])
        names = sorted(acts)
        rows.append(np.concatenate([obs] + [acts[n][0].numpy() for n in names] + [probs, [value]]))
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
    capture_s = time.perf_counter() - t0
    for h in hooks:
        h.remove()

    names = sorted(acts)
    sizes = [int(acts[n].shape[1]) for n in names]
    F = np.stack(rows).astype(np.float32)
    n_hid = sum(sizes)
    vals = F[:, -1]
    vmax = float(np.abs(vals).max()) or 1.0
    Q = np.concatenate([np.clip(F[:, :n_in] / 4.0, -1, 1),             # z-scores, clip at ±4
                        np.clip(F[:, n_in:n_in + n_hid + 4], -1, 1),   # tanh activations + probs
                        (vals / vmax)[:, None]], axis=1)
    Q = np.round(Q * 127).astype(np.int8)

    edges = {}
    for tower in ("policy_net", "value_net"):
        lin = [m for m in getattr(pol.mlp_extractor, tower) if isinstance(m, torch.nn.Linear)]
        head = pol.action_net if tower == "policy_net" else pol.value_net
        mats = []
        for W in [l.weight for l in lin] + [head.weight]:
            Wn = W.detach().numpy()
            k = min(args.k, Wn.shape[1])
            order = np.argsort(-np.abs(Wn), axis=1)[:, :k]
            mats.append({"rows": int(Wn.shape[0]), "cols": int(Wn.shape[1]), "k": int(k),
                         "idx": b64(order.astype(np.int16)),
                         "w": b64(np.take_along_axis(Wn, order, 1).astype(np.float32))})
        edges[tower] = mats

    sym = re.split(r"[_\-.]", csv.name)[0].upper()
    ts = raw["timestamp"].astype(str).tolist()
    data = {
        "model": args.model, "source": args.source, "slice": args.slice, "symbol": sym,
        "window": window, "n_features": len(features), "n_in": n_in, "layers": sizes,
        "frames": len(rows), "width": int(Q.shape[1]), "value_max": vmax, "q": b64(Q),
        "ts": [ts[i] for i in bar_idx],
        "o": [round(float(raw["open"].iloc[i]), 5) for i in bar_idx],
        "h": [round(float(raw["high"].iloc[i]), 5) for i in bar_idx],
        "l": [round(float(raw["low"].iloc[i]), 5) for i in bar_idx],
        "c": [round(float(raw["close"].iloc[i]), 5) for i in bar_idx],
        "act": actions, "pos": position, "eq": [round(e, 6) for e in equity],
        "tpnl": [round(p, 6) for p in trade_pnl], "edges": edges,
    }
    out = Path(args.out) if args.out else ROOT / "artifacts" / "viz" / f"{args.model}_network3d.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__SNAPSHOT__*/null", json.dumps(data))
    out.write_text(html, encoding="utf-8")
    closed = [p for p in trade_pnl if p != 0.0]
    print(f"[viz] {args.model} ({args.source}) {args.slice}: {len(rows)} bars {data['ts'][0]} -> {data['ts'][-1]}")
    print(f"[viz] layers {sizes}, inputs {n_in}, edges kept {args.k}/neuron, capture {capture_s:.2f}s")
    print(f"[viz] trades closed {len(closed)}, equity end {equity[-1]:.4f}")
    print(f"[viz] -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
