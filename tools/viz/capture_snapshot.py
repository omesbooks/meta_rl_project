"""Capture a visual snapshot of a trained PPO/A2C policy for the neon 3D viewer.

Walks the model deterministically over a slice of its own dataset (default:
Validation) and inlines the snapshot into tools/viz/network3d.html -> one
self-contained HTML (Three.js is vendored under tools/viz/vendor, so the file
must stay next to that folder: the default output goes to tools/viz/out/).
For live snapshots during training use `rl_train.py --viz` instead.

NOTE: Validation is the slice used to pick the best checkpoint, so the equity
shown is a visual, not evidence of edge (use --slice test for unseen bars).

Usage (from the repo root):
    python tools/viz/capture_snapshot.py <model> [--frames 400] [--slice validation|test]
                                         [--offset 0] [--k 16] [--source final|best] [--out <html>]
"""
import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
VIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VIZ))

from stable_baselines3 import A2C, PPO  # noqa: E402

from action_profiles import get_action_profile  # noqa: E402
from artifact_paths import best_dir, final_model_path, train_meta_path  # noqa: E402
from training_data import apply_normalization, load_dataset  # noqa: E402
from trading_env import TradingEnv  # noqa: E402
import snapshot  # noqa: E402

TEMPLATE = VIZ / "network3d.html"


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
    algo = str(hp.get("algo", "ppo")).lower()
    if algo == "dqn":
        sys.exit("visual snapshots need an actor-critic policy (PPO / A2C); this model is DQN")
    model = (A2C if algo == "a2c" else PPO).load(str(zip_path), device="cpu")
    norm = pd.read_csv(Path(final_model_path(args.model)).with_name(f"{args.model}_norm.csv"), index_col=0)

    csv = Path(meta["train_csv"])
    csv = csv if csv.is_absolute() else ROOT / csv
    df = load_dataset(str(csv))
    s0 = int(meta["split_index"])
    v_rows = int(meta["validation_rows"])
    part = df.iloc[s0:s0 + v_rows] if args.slice == "validation" else df.iloc[s0 + v_rows:]
    part = part.iloc[args.offset:].reset_index(drop=True)
    if len(part) < window + 4:
        sys.exit(f"slice too short: {len(part)} rows")
    part = apply_normalization(part, features, norm)

    action_cfg = hp.get("action_profile_config") or         get_action_profile(hp.get("action_profile", "basic_4"), hp.get("action_params"))[1]
    env = TradingEnv(part, features, window_size=window, random_start=False,
                     max_steps=len(part) - window - 3,
                     max_hold_bars=int(hp.get("max_hold", 35)),
                     reward_mode=hp.get("reward_mode", "realized"),
                     reward_profile=hp.get("reward_profile", "balanced"),
                     action_profile=action_cfg)
    snap = snapshot.capture(model.policy, env, window, args.frames, k=args.k,
                            action_names=snapshot.action_names(action_cfg),
                            meta=dict(model=args.model, source=args.source, slice=args.slice,
                                      symbol=re.split(r"[_\-.]", csv.name)[0].upper(),
                                      steps=int(meta.get("actual_timesteps") or 0)))

    out = Path(args.out) if args.out else VIZ / "out" / f"{args.model}_network3d.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__SNAPSHOT__*/null", json.dumps(snap))
    # the template imports ./vendor/... and ./dashboard.js relative to tools/viz
    rel = Path(os.path.relpath(VIZ, out.parent.resolve())).as_posix()
    if rel != ".":
        html = html.replace('"./vendor/', f'"{rel}/vendor/').replace('"./dashboard.js"', f'"{rel}/dashboard.js"')
    out.write_text(html, encoding="utf-8")
    closed = [p for p in snap["tpnl"] if p != 0.0]
    print(f"[viz] {args.model} ({args.source}) {args.slice}: {snap['frames']} bars {snap['ts'][0]} -> {snap['ts'][-1]}")
    print(f"[viz] actor {snap['actor_layers']} critic {snap['critic_layers']}, inputs {snap['n_in']}, "
          f"actions {snap['action_labels']}, capture {snap['meta']['capture_s']}s")
    print(f"[viz] trades closed {len(closed)}, equity end {snap['eq'][-1]:.4f}")
    print(f"[viz] -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
