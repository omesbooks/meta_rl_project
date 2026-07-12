"""
RL Agent Confidence Analysis
----------------------------
ดูว่า agent ทายแม่นขึ้นเมื่อมั่นใจสูงไหม
ถ้าใช่ -> มี edge อยู่ในบางสถานการณ์ -> ใช้ confidence filter

Usage:
    python rl_analyze.py rl_v2_full EURUSD_H1.csv
"""
import sys
import io
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from trading_env import TradingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("csv")
    ap.add_argument("--start", type=float, default=0.8)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--max_hold", type=int, default=None,
                    help="default = read from model train metadata, then 30")
    args = ap.parse_args()

    # load + normalize
    df = pd.read_csv(args.csv)
    leaky = [c for c in df.columns if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))]
    if leaky:
        df = df.drop(columns=leaky)
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Resolve artifacts via artifact_paths (post-reorg layout with legacy fallback)
    from artifact_paths import find_norm_path, find_model_path, train_meta_path
    norm_path = find_norm_path(args.model)
    if norm_path is None or not Path(norm_path).exists():
        print(f"ERROR: norm stats not found for model '{args.model}' "
              f"(looked in artifacts/models/{args.model}/ and repo root)")
        sys.exit(1)
    norm = pd.read_csv(norm_path, index_col=0)
    for c in feature_cols:
        if c in norm.index:
            df[c] = (df[c] - norm.at[c, "mean"]) / norm.at[c, "std"]
    df = df.fillna(0).reset_index(drop=True)

    start = int(len(df) * args.start)
    test_df = df.iloc[start:].reset_index(drop=True)

    from stable_baselines3 import PPO
    import torch
    model_path = find_model_path(args.model, "final")
    if model_path is None or not Path(model_path).exists():
        print(f"ERROR: model zip not found for '{args.model}'")
        sys.exit(1)
    print(f"[load] {model_path}")
    model = PPO.load(str(model_path))

    # Resolve the training contract from metadata.  Older models may not have
    # metadata, so action-count detection remains the safe fallback.
    from action_profiles import get_action_profile, profile_for_action_count
    hparams = {}
    meta_path = train_meta_path(args.model)
    if meta_path.exists():
        try:
            hparams = json.loads(meta_path.read_text(encoding="utf-8-sig")).get(
                "hyperparameters", {})
        except Exception as exc:
            print(f"[contract] warn: could not read {meta_path}: {exc}")

    n_actions = int(model.action_space.n)
    action_value = hparams.get("action_profile_config") or hparams.get("action_profile")
    action_params = hparams.get("action_profile_params") or {}
    if action_value is None:
        action_key, action_profile = profile_for_action_count(n_actions)
        action_source = "model action count"
    else:
        action_key, action_profile = get_action_profile(action_value, action_params)
        action_source = "train metadata"
    if len(action_profile["actions"]) != n_actions:
        print(
            f"ERROR: action profile '{action_profile['label']}' has "
            f"{len(action_profile['actions'])} actions but model outputs {n_actions}."
        )
        sys.exit(1)

    max_hold = args.max_hold
    max_hold_source = "CLI"
    if max_hold is None:
        max_hold = int(hparams.get("max_hold", 30))
        max_hold_source = "train metadata" if "max_hold" in hparams else "fallback"
    action_display = hparams.get("action_profile") or action_key
    print(f"[contract] action_profile={action_display} ({n_actions} actions; {action_source})")
    print(f"[contract] max_hold={max_hold} bars ({max_hold_source})")

    # Auto-detect window from the model's observation space (same rule as
    # backtest_live): obs_dim = window * n_features + 3
    obs_dim = int(model.observation_space.shape[0])
    n_feat = len(feature_cols)
    detected = (obs_dim - 3) // n_feat if n_feat > 0 else 0
    if detected > 0 and (obs_dim - 3) % n_feat == 0:
        if args.window > 0 and args.window != detected:
            print(f"[window] --window {args.window} != model's {detected}; using {detected}")
        window = detected
    else:
        window = args.window if args.window > 0 else 10
        if window * n_feat + 3 != obs_dim:
            print(f"\nERROR: model/dataset feature mismatch")
            print(f"  model expects obs_dim={obs_dim}, dataset provides {n_feat} features")
            print(f"  -> pick the dataset this model was trained on")
            sys.exit(1)

    env = TradingEnv(test_df, feature_cols, window_size=window,
                     max_steps=len(test_df) - window - 2,
                     max_hold_bars=max_hold, reward_mode="realized",
                     action_profile=action_profile)
    obs, _ = env.reset()

    records = []
    done = False
    while not done:
        # Get action probabilities (not just argmax)
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).float()
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.numpy().flatten()
        action = int(np.argmax(probs))
        max_prob = float(probs[action])

        # Track entry trades for analysis
        is_entry = (action in (1, 2)) and env.position == 0
        entry_price = env.df.at[env.t, "close"] if is_entry else None

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if is_entry:
            records.append({
                "step": env.t,
                "action": action,
                "max_prob": max_prob,
                "entry": entry_price,
            })

    # Match entries to exits via env.trades
    print(f"\n[analyze] {len(env.trades)} trades")
    if not env.trades:
        return

    # Take top-K by confidence
    print("\n" + "=" * 70)
    print("  Performance by Confidence Bucket (entry decisions only)")
    print("=" * 70)

    # build trades dataframe with confidence
    trades_df = pd.DataFrame(env.trades)
    # Match — both have same order
    n = min(len(records), len(trades_df))
    trades_df = trades_df.iloc[:n].copy()
    trades_df["confidence"] = [r["max_prob"] for r in records[:n]]

    print(f"  {'Conf range':<15} {'#trades':>10} {'WinRate':>10} {'AvgPnL':>12} {'TotalPnL':>12}")
    print("-" * 70)

    buckets = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70),
               (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in buckets:
        sub = trades_df[(trades_df["confidence"] >= lo) & (trades_df["confidence"] < hi)]
        if len(sub) == 0:
            continue
        wr = (sub["pnl"] > 0).mean() * 100
        avg = sub["pnl"].mean() * 100
        tot = sub["pnl"].sum() * 100
        print(f"  [{lo:.2f}-{hi:.2f})    {len(sub):>10,} {wr:>9.2f}% {avg:>+11.4f}% {tot:>+11.2f}%")

    # threshold scan
    print("\n" + "=" * 70)
    print("  Threshold scan — เทรดเฉพาะตอนมั่นใจ >= X")
    print("=" * 70)
    print(f"  {'Conf>=':<10} {'#trades':>10} {'WinRate':>10} {'TotalPnL':>12} {'PF':>10}")
    print("-" * 70)

    best_pf = 0
    best_thr = None
    for thr in np.arange(0.30, 0.95, 0.05):
        sub = trades_df[trades_df["confidence"] >= thr]
        if len(sub) < 30:
            continue
        wr = (sub["pnl"] > 0).mean() * 100
        tot = sub["pnl"].sum() * 100
        wins = sub[sub["pnl"] > 0]["pnl"].sum()
        losses = abs(sub[sub["pnl"] <= 0]["pnl"].sum())
        pf = wins / losses if losses > 0 else float("inf")
        marker = " ⭐" if pf > best_pf else ""
        if pf > best_pf:
            best_pf = pf
            best_thr = thr
        print(f"  >={thr:.2f}    {len(sub):>10,} {wr:>9.2f}% {tot:>+11.2f}% {pf:>9.2f}{marker}")

    print("\n" + "=" * 70)
    if best_pf > 1.2:
        print(f"  ✅ Edge มีจริง! ใช้ conf >= {best_thr:.2f} → PF = {best_pf:.2f}")
    elif best_pf > 1.0:
        print(f"  🟡 มี edge เล็กน้อย: conf >= {best_thr:.2f} → PF = {best_pf:.2f}")
    else:
        print(f"  ❌ ไม่มี edge ที่ confidence level ใดเลย (best PF={best_pf:.2f})")
        print(f"     → ต้องเปลี่ยน features / instrument / target")
    print("=" * 70)


if __name__ == "__main__":
    main()
