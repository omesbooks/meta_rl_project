"""
RL Backtest — รัน trained agent บน data + แสดงสถิติ + equity curve
-----------------------------------------------------------------
Usage:
    python rl_backtest.py rl_v1 EURUSD_H1.csv [--start 0.8]
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
    ap.add_argument("model", help="model name (e.g. rl_v1) — looks for rl_v1.zip")
    ap.add_argument("csv", help="CSV file")
    ap.add_argument("--start", type=float, default=0.8,
                    help="start fraction (0.8 = use last 20% as test)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "a2c"])
    ap.add_argument("--reward_mode", default="realized",
                    choices=["realized", "mtm"])
    ap.add_argument("--max_hold", type=int, default=30)
    args = ap.parse_args()

    print("=" * 60)
    print(f"  RL Backtest — {args.model}")
    print("=" * 60)

    # ---------- load data ----------
    df = pd.read_csv(args.csv)
    print(f"\n[data] {args.csv} — {len(df):,} rows")

    leaky = [c for c in df.columns
             if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))]
    if leaky:
        df = df.drop(columns=leaky)

    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # apply same normalization as training
    norm_file = Path(f"{args.model}_norm.csv")
    if norm_file.exists():
        norm = pd.read_csv(norm_file, index_col=0)
        for c in feature_cols:
            if c in norm.index:
                df[c] = (df[c] - norm.at[c, "mean"]) / (norm.at[c, "std"] + 1e-8)
        print(f"[norm] applied stats from {norm_file}")
    else:
        print(f"[norm] WARNING: {norm_file} not found — using raw features")

    df = df.fillna(0).reset_index(drop=True)

    # use last (1 - start) portion as test
    start = int(len(df) * args.start)
    test_df = df.iloc[start:].reset_index(drop=True)
    print(f"[test] rows {start} .. end ({len(test_df):,} rows)")

    # ---------- load model ----------
    from stable_baselines3 import PPO, DQN, A2C
    cls = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[args.algo]
    model_path = f"{args.model}.zip"
    print(f"[load] {model_path}")
    model = cls.load(model_path)

    # ---------- run ----------
    env = TradingEnv(test_df, feature_cols, window_size=args.window,
                     max_steps=len(test_df) - args.window - 2,
                     reward_mode=args.reward_mode,
                     max_hold_bars=args.max_hold)
    obs, _ = env.reset()

    actions_log = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        actions_log.append(int(action))
        done = terminated or truncated

    stats = env.get_stats()

    # ---------- print stats ----------
    print("\n" + "=" * 50)
    print(f"  Backtest result: {args.model}")
    print("=" * 50)
    print(f"  Total trades       : {stats['trades']:,}")
    if stats['trades'] > 0:
        print(f"  Win rate           : {stats['win_rate']:.2%}")
        print(f"  Profit factor      : {stats['profit_factor']:.2f}")
        print(f"  Avg win            : {stats['avg_win']:+.4%}")
        print(f"  Avg loss           : {stats['avg_loss']:+.4%}")
    print(f"  Return             : {stats['return']:+.2%}")
    print(f"  Max drawdown       : {stats['max_dd']:.2%}")

    # action distribution
    from collections import Counter
    cnt = Counter(actions_log)
    names = {0: "Hold", 1: "Buy", 2: "Sell", 3: "Close"}
    print(f"\n  Action distribution:")
    for a in (0, 1, 2, 3):
        n = cnt.get(a, 0)
        pct = n / len(actions_log) * 100 if actions_log else 0
        print(f"    {names[a]:<6}: {n:>6,} ({pct:5.1f}%)")
    print("=" * 50)

    # ---------- save trades CSV ----------
    if env.trades:
        out_csv = Path(f"{args.model}_trades.csv")
        pd.DataFrame(env.trades).to_csv(out_csv, index=False)
        print(f"\n[save] trades -> {out_csv}")

    # ---------- equity curve chart ----------
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
        eq = np.array(env._curve)
        axes[0].plot(eq, color="#2563eb", linewidth=1.2)
        axes[0].fill_between(range(len(eq)), 1.0, eq,
                              where=(eq >= 1.0), color="#10b981", alpha=0.15)
        axes[0].fill_between(range(len(eq)), 1.0, eq,
                              where=(eq < 1.0), color="#ef4444", alpha=0.15)
        axes[0].axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
        axes[0].set_title(f"RL Equity Curve — {args.model}")
        axes[0].set_ylabel("Equity (normalized)")
        axes[0].grid(True, alpha=0.3)

        # drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        axes[1].fill_between(range(len(dd)), 0, dd, color="#ef4444", alpha=0.5)
        axes[1].set_ylabel("DD (%)")
        axes[1].set_xlabel("Step")
        axes[1].grid(True, alpha=0.3)

        chart_path = f"{args.model}_equity.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=110, bbox_inches="tight")
        print(f"[save] equity chart -> {chart_path}")
    except Exception as e:
        print(f"[chart] skipped: {e}")


if __name__ == "__main__":
    main()
