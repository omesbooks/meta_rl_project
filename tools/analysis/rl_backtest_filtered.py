"""
RL Backtest with Confidence Filter
-----------------------------------
รัน agent บน test set แต่จะ "ไม่เปิด trade" ถ้า confidence < threshold
= simulate การเทรดจริงแบบเลือก setup ที่ดี

Usage:
    python rl_backtest_filtered.py rl_v3 training_data_v2_relabeled.csv --conf 0.85
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
    ap.add_argument("--conf", type=float, default=0.85,
                    help="confidence threshold — เปิด trade เฉพาะ conf >= นี้")
    ap.add_argument("--start", type=float, default=0.8)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--max_hold", type=int, default=30)
    args = ap.parse_args()

    print("=" * 60)
    print(f"  RL Backtest WITH FILTER — {args.model}")
    print(f"  Confidence threshold: {args.conf}")
    print("=" * 60)

    # load + normalize
    df = pd.read_csv(args.csv)
    leaky = [c for c in df.columns if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))]
    if leaky: df = df.drop(columns=leaky)
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
    print(f"[test] {len(test_df):,} rows")

    from stable_baselines3 import PPO
    import torch
    print(f"[load] {args.model}.zip")
    model = PPO.load(f"{args.model}.zip")

    env = TradingEnv(test_df, feature_cols, window_size=args.window,
                     max_steps=len(test_df) - args.window - 2,
                     max_hold_bars=args.max_hold, reward_mode="realized")
    obs, _ = env.reset()

    skipped = 0
    overridden = 0
    actions_log = []
    done = False

    while not done:
        # Get action probabilities
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).float()
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.numpy().flatten()
        action = int(np.argmax(probs))
        max_prob = float(probs[action])

        # ⭐ CONFIDENCE FILTER ⭐
        # ถ้าเป็น Buy/Sell แต่ confidence ต่ำกว่า threshold -> เปลี่ยนเป็น Hold
        if action in (1, 2) and env.position == 0:
            if max_prob < args.conf:
                action = 0  # override -> Hold
                skipped += 1
            else:
                overridden += 1

        actions_log.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    stats = env.get_stats()

    print(f"\n[filter] entries skipped (conf < {args.conf}): {skipped:,}")
    print(f"[filter] entries taken (conf >= {args.conf}): {overridden:,}")

    # ---------- print stats ----------
    print("\n" + "=" * 50)
    print(f"  Filtered Backtest: {args.model}")
    print(f"  Conf threshold: {args.conf}")
    print("=" * 50)
    print(f"  Total trades       : {stats['trades']:,}")
    if stats['trades'] > 0:
        print(f"  Win rate           : {stats['win_rate']:.2%}")
        print(f"  Profit factor      : {stats['profit_factor']:.2f}")
        print(f"  Avg win            : {stats['avg_win']:+.4%}")
        print(f"  Avg loss           : {stats['avg_loss']:+.4%}")
    print(f"  Return             : {stats['return']:+.2%}")
    print(f"  Max drawdown       : {stats['max_dd']:.2%}")

    from collections import Counter
    cnt = Counter(actions_log)
    names = {0: "Hold", 1: "Buy", 2: "Sell", 3: "Close"}
    print(f"\n  Action distribution:")
    for a in (0, 1, 2, 3):
        n = cnt.get(a, 0)
        pct = n / len(actions_log) * 100 if actions_log else 0
        print(f"    {names[a]:<6}: {n:>6,} ({pct:5.1f}%)")
    print("=" * 50)

    if env.trades:
        out_csv = Path(f"{args.model}_filtered_trades.csv")
        pd.DataFrame(env.trades).to_csv(out_csv, index=False)
        print(f"\n[save] -> {out_csv}")

    # equity curve
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
        eq = np.array(env._curve)
        axes[0].plot(eq, color="#7c3aed", linewidth=1.2)
        axes[0].fill_between(range(len(eq)), 1.0, eq,
                              where=(eq >= 1.0), color="#10b981", alpha=0.15)
        axes[0].fill_between(range(len(eq)), 1.0, eq,
                              where=(eq < 1.0), color="#ef4444", alpha=0.15)
        axes[0].axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
        axes[0].set_title(f"Filtered Equity (conf>={args.conf}) — {args.model}")
        axes[0].set_ylabel("Equity")
        axes[0].grid(True, alpha=0.3)

        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        axes[1].fill_between(range(len(dd)), 0, dd, color="#ef4444", alpha=0.5)
        axes[1].set_ylabel("DD (%)")
        axes[1].set_xlabel("Step")
        axes[1].grid(True, alpha=0.3)

        chart = f"{args.model}_filtered_equity.png"
        plt.tight_layout()
        plt.savefig(chart, dpi=110, bbox_inches="tight")
        print(f"[save] -> {chart}")
    except Exception as e:
        print(f"[chart] skipped: {e}")


if __name__ == "__main__":
    main()
