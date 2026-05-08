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
    ap.add_argument("--max_hold", type=int, default=30)
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

    norm = pd.read_csv(f"{args.model}_norm.csv", index_col=0)
    for c in feature_cols:
        if c in norm.index:
            df[c] = (df[c] - norm.at[c, "mean"]) / (norm.at[c, "std"] + 1e-8)
    df = df.fillna(0).reset_index(drop=True)

    start = int(len(df) * args.start)
    test_df = df.iloc[start:].reset_index(drop=True)

    from stable_baselines3 import PPO
    import torch
    print(f"[load] {args.model}.zip")
    model = PPO.load(f"{args.model}.zip")

    env = TradingEnv(test_df, feature_cols, window_size=args.window,
                     max_steps=len(test_df) - args.window - 2,
                     max_hold_bars=args.max_hold, reward_mode="realized")
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
