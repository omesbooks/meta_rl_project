"""
Walk-Forward Validation สำหรับ RL Trading Agent
------------------------------------------------
เทรน + เทสต์ ซ้ำหลายช่วง เพื่อพิสูจน์ว่า edge ของ model robust จริง
ไม่ใช่ over-fit ของ luck เฉพาะช่วง

วิธี:
  Window 1: train [0%-50%]   -> test [50%-60%]
  Window 2: train [10%-60%]  -> test [60%-70%]
  Window 3: train [20%-70%]  -> test [70%-80%]
  Window 4: train [30%-80%]  -> test [80%-90%]
  Window 5: train [40%-90%]  -> test [90%-100%]

(rolling window — ทั้ง train และ test เลื่อนไปข้างหน้า)

เกณฑ์ robust:
  ✅ ถ้า WR > 50% และ PF > 1.0 ในทุก window
  ⚠️ ถ้า มี 1-2 windows ที่ใต้เส้น
  ❌ ถ้า > half ของ windows แย่

Usage:
    python rl_walkforward.py training_data_v2_relabeled.csv \\
        --windows 5 --steps 50000 --window 10 --name wf_xau

ใช้เวลา: ~5 windows × 8-15 นาที = 40-75 นาที
"""
import sys
import io
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from trading_env import TradingEnv


def _fallback_output_path(path):
    path = Path(path)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def _save_csv_with_fallback(df, path):
    path = Path(path)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError as e:
        fallback = _fallback_output_path(path)
        print(f"  [warn] cannot write {path}: {e}")
        try:
            df.to_csv(fallback, index=False)
            return fallback
        except PermissionError as e2:
            print(f"  [warn] cannot write fallback {fallback}: {e2}")
            return None


def _save_fig_with_fallback(fig, path, **kwargs):
    path = Path(path)
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError as e:
        fallback = _fallback_output_path(path)
        print(f"  [warn] cannot write {path}: {e}")
        try:
            fig.savefig(fallback, **kwargs)
            return fallback
        except PermissionError as e2:
            print(f"  [warn] cannot write fallback {fallback}: {e2}")
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--windows", type=int, default=5,
                    help="จำนวน walk-forward windows")
    ap.add_argument("--steps", type=int, default=50000,
                    help="training steps per window")
    ap.add_argument("--window", type=int, default=10,
                    help="state window size for env")
    ap.add_argument("--max_hold", type=int, default=30)
    ap.add_argument("--ep_len", type=int, default=2000)
    ap.add_argument("--name", default="wf",
                    help="output prefix")
    ap.add_argument("--mode", default="rolling",
                    choices=["rolling", "anchored"],
                    help="rolling=train window slides / anchored=train always from start")
    args = ap.parse_args()

    # =================================================================
    # Load data
    # =================================================================
    print("=" * 60)
    print("  Walk-Forward Validation")
    print("=" * 60)

    df = pd.read_csv(args.csv)
    leaky = [c for c in df.columns if any(k in c.lower()
             for k in ("future_", "forward_", "next_", "target"))]
    if leaky:
        df = df.drop(columns=leaky)
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low",
            "close", "volume"}
    feature_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"\n[data] {args.csv}")
    print(f"  rows: {len(df):,}")
    print(f"  features: {len(feature_cols)}")
    print(f"  windows: {args.windows} ({args.mode})")
    print(f"  steps/window: {args.steps:,}")

    n = len(df)
    if args.windows <= 0:
        print("ERROR: --windows must be greater than 0")
        return 1

    # Define windows.
    # The last 50% of the dataset is split into N non-overlapping test slices.
    # In rolling mode the train slice overlaps by design, while test slices do not.
    test_start = int(n * 0.50)
    train_span = test_start
    boundaries = np.linspace(test_start, n, args.windows + 1, dtype=int)
    windows = []
    for i in range(args.windows):
        train_end = int(boundaries[i])
        test_end = int(boundaries[i + 1])
        if test_end <= train_end:
            continue
        if args.mode == "rolling":
            train_start = max(0, train_end - train_span)
        else:  # anchored
            train_start = 0
        windows.append((train_start, train_end, train_end, test_end))

    if not windows:
        print("ERROR: no valid walk-forward windows. Reduce --windows or use more rows.")
        return 1

    min_rows = args.window + 3
    too_small = [(i + 1, ve - vs) for i, (_, _, vs, ve) in enumerate(windows)
                 if (ve - vs) < min_rows]
    if too_small:
        first_idx, rows = too_small[0]
        print(
            f"ERROR: window {first_idx} test slice has only {rows} rows; "
            f"need at least {min_rows}. Reduce --windows or window size."
        )
        return 1

    total_windows = len(windows)

    print(f"\n[windows]")
    for k, (ts, te, vs, ve) in enumerate(windows, 1):
        print(f"  {k}: train[{ts:>6}:{te:>6}] ({te-ts:>6} rows) "
              f"-> test[{vs:>6}:{ve:>6}] ({ve-vs:>6} rows)")
    print("  note: test slices do not overlap; rolling train slices overlap by design")

    # =================================================================
    # Setup PPO
    # =================================================================
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    class WFProgressCallback(BaseCallback):
        def __init__(self, window_idx, total_windows, steps_per_window):
            super().__init__()
            self.window_idx = window_idx
            self.total_windows = total_windows
            self.steps_per_window = max(int(steps_per_window), 1)
            self.units_per_window = self.steps_per_window + 1
            self.total_units = self.total_windows * self.units_per_window
            self.emit_every = max(1000, self.steps_per_window // 20)
            self.next_emit = 0
            self.final_emitted = False

        def _emit(self, current_step):
            current_step = min(max(int(current_step), 0), self.steps_per_window)
            global_step = (self.window_idx - 1) * self.units_per_window + current_step
            print(
                f"WF_PROGRESS {global_step} {self.total_units} "
                f"Window {self.window_idx}/{self.total_windows} train "
                f"{current_step:,}/{self.steps_per_window:,}",
                flush=True,
            )

        def _on_training_start(self):
            self._emit(0)

        def _on_step(self):
            current = min(int(self.num_timesteps), self.steps_per_window)
            if current >= self.steps_per_window:
                if not self.final_emitted:
                    self._emit(self.steps_per_window)
                    self.final_emitted = True
                return True
            if current >= self.next_emit:
                self._emit(current)
                self.next_emit = current + self.emit_every
            return True

    # Auto NN size
    if args.window >= 50:
        net_arch = [512, 256, 128]
    elif args.window >= 20:
        net_arch = [256, 128, 64]
    else:
        net_arch = [128, 64]
    print(f"  net_arch: {net_arch}")

    # =================================================================
    # Run each window
    # =================================================================
    results = []
    all_curves = []

    total_start = time.time()
    progress_units = total_windows * (max(int(args.steps), 1) + 1)
    print(f"WF_PROGRESS 0 {progress_units} Starting walk-forward", flush=True)

    for w_idx, (ts, te, vs, ve) in enumerate(windows, 1):
        print(f"\n{'=' * 60}")
        print(f"  Window {w_idx}/{total_windows}")
        print(f"{'=' * 60}")
        win_start = time.time()

        train_df = df.iloc[ts:te].reset_index(drop=True).copy()
        test_df = df.iloc[vs:ve].reset_index(drop=True).copy()

        # Normalize on train, apply to test
        feat_mean = train_df[feature_cols].mean()
        feat_std = train_df[feature_cols].std() + 1e-8
        train_df[feature_cols] = (train_df[feature_cols] - feat_mean) / feat_std
        test_df[feature_cols] = (test_df[feature_cols] - feat_mean) / feat_std
        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)

        # Train env
        def make_env(df_in, ep):
            return Monitor(TradingEnv(
                df_in, feature_cols,
                window_size=args.window,
                max_steps=ep,
                reward_mode="realized",
                max_hold_bars=args.max_hold,
            ))

        train_env = DummyVecEnv([lambda: make_env(train_df, args.ep_len)])

        # Train
        print(f"  [train] {args.steps:,} steps ...")
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            policy_kwargs=dict(net_arch=net_arch),
        )
        progress_cb = WFProgressCallback(w_idx, total_windows, args.steps)
        model.learn(total_timesteps=args.steps, callback=progress_cb, progress_bar=False)
        progress_at_test = (w_idx - 1) * (max(int(args.steps), 1) + 1) + max(int(args.steps), 1)
        print(
            f"WF_PROGRESS {progress_at_test} {progress_units} "
            f"Window {w_idx}/{total_windows} test",
            flush=True,
        )

        # Test
        print(f"  [test]  out-of-sample ...")
        eval_max_steps = max(1, len(test_df) - args.window - 3)
        test_env_raw = TradingEnv(
            test_df, feature_cols,
            window_size=args.window,
            max_steps=eval_max_steps,
            reward_mode="realized",
            max_hold_bars=args.max_hold,
        )
        obs, _ = test_env_raw.reset()
        # Evaluation should cover the whole test slice from the start.
        # TradingEnv randomizes reset for training; force deterministic test.
        test_env_raw.t = args.window - 1
        test_env_raw.start_t = test_env_raw.t
        obs = test_env_raw._get_obs()
        done = False
        expected_obs_shape = test_env_raw.observation_space.shape
        while not done:
            if obs.shape != expected_obs_shape:
                print(
                    f"  [warn] invalid obs shape {obs.shape}; "
                    f"expected {expected_obs_shape}. Stopping window test safely."
                )
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = test_env_raw.step(int(action))
            done = terminated or truncated

        stats = test_env_raw.get_stats()
        elapsed = time.time() - win_start
        stats_clean = {
            "window": w_idx,
            "train_rows": te - ts,
            "test_rows": ve - vs,
            "trades": stats["trades"],
            "win_rate": stats.get("win_rate", 0),
            "profit_factor": stats.get("profit_factor", 0),
            "return": stats["return"],
            "max_dd": stats["max_dd"],
            "elapsed_s": round(elapsed, 1),
        }
        results.append(stats_clean)
        all_curves.append(test_env_raw._curve)

        print(f"\n  Window {w_idx} result:")
        print(f"    Trades  : {stats['trades']}")
        print(f"    WR      : {stats.get('win_rate', 0):.2%}")
        print(f"    PF      : {stats.get('profit_factor', 0):.2f}")
        print(f"    Return  : {stats['return']:+.2%}")
        print(f"    MaxDD   : {stats['max_dd']:.2%}")
        print(f"    Elapsed : {elapsed:.0f}s")
        print(
            f"WF_PROGRESS {w_idx * (max(int(args.steps), 1) + 1)} {progress_units} "
            f"Window {w_idx}/{total_windows} done",
            flush=True,
        )

    total_elapsed = time.time() - total_start

    # =================================================================
    # Aggregate report
    # =================================================================
    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward SUMMARY ({total_elapsed/60:.1f} min total)")
    print(f"{'=' * 60}")

    res_df = pd.DataFrame(results)
    print()
    cols_to_show = ["window", "trades", "win_rate", "profit_factor", "return", "max_dd"]
    print(res_df[cols_to_show].to_string(index=False,
        formatters={
            "win_rate": lambda v: f"{v:.2%}",
            "profit_factor": lambda v: f"{v:.2f}",
            "return": lambda v: f"{v:+.2%}",
            "max_dd": lambda v: f"{v:.2%}",
        }))

    # Stability stats
    wr_arr = res_df["win_rate"].values
    pf_arr = np.clip(res_df["profit_factor"].values, 0, 10)  # cap inf
    ret_arr = res_df["return"].values
    dd_arr = res_df["max_dd"].values

    print(f"\n  📊 Aggregate metrics:")
    print(f"    WR   : mean={wr_arr.mean():.2%} | std={wr_arr.std():.4f} | "
          f"min={wr_arr.min():.2%} | max={wr_arr.max():.2%}")
    print(f"    PF   : mean={pf_arr.mean():.2f} | std={pf_arr.std():.2f} | "
          f"min={pf_arr.min():.2f} | max={pf_arr.max():.2f}")
    print(f"    Ret  : mean={ret_arr.mean():+.2%} | std={ret_arr.std():.4f} | "
          f"min={ret_arr.min():+.2%} | max={ret_arr.max():+.2%}")
    print(f"    DD   : mean={dd_arr.mean():.2%} | worst={dd_arr.min():.2%}")

    # Robustness verdict
    print(f"\n  🎯 Robustness verdict:")
    n_pos = sum(1 for x in pf_arr if x > 1.0)
    n_wr = sum(1 for x in wr_arr if x > 0.5)
    if n_pos == total_windows and n_wr == total_windows:
        verdict = "✅ ROBUST — ทุก window ผ่าน (PF>1 และ WR>50%) → edge จริง"
    elif n_pos >= total_windows * 0.7:
        verdict = f"🟡 MOSTLY ROBUST — {n_pos}/{total_windows} windows มี PF>1 → ใช้ระวัง"
    elif n_pos >= total_windows * 0.5:
        verdict = f"⚠️ UNSTABLE — แค่ {n_pos}/{total_windows} windows ผ่าน → over-fit เป็นไปได้สูง"
    else:
        verdict = f"❌ NOT ROBUST — {n_pos}/{total_windows} windows ผ่าน → edge ไม่จริง / ขึ้นอยู่กับ luck"
    print(f"    {verdict}")
    print(f"WF_PROGRESS {progress_units} {progress_units} Complete", flush=True)

    # Save CSV
    out_csv = f"{args.name}_results.csv"
    out_csv = _save_csv_with_fallback(res_df, out_csv)
    if out_csv:
        print(f"\n  [save] -> {out_csv}")
    else:
        print("\n  [save] results CSV skipped (permission denied)")

    # Equity curve chart
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                                  gridspec_kw={"height_ratios": [3, 2]})

        # Top: equity curves of each window
        colors = plt.cm.viridis(np.linspace(0, 0.9, total_windows))
        for i, curve in enumerate(all_curves):
            axes[0].plot(curve, color=colors[i], linewidth=1.2,
                         label=f"W{i+1}: PF={results[i]['profit_factor']:.2f}, "
                               f"Ret={results[i]['return']:+.1%}")
        axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
        axes[0].set_title(f"Walk-Forward Equity Curves — {total_windows} windows")
        axes[0].set_ylabel("Equity (normalized)")
        axes[0].legend(loc="best", fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Bottom: WR + PF bar chart
        x = np.arange(total_windows) + 1
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        bars1 = ax2.bar(x - 0.2, wr_arr * 100, 0.4,
                        color="#3b82f6", label="Win Rate (%)")
        bars2 = ax2_twin.bar(x + 0.2, pf_arr, 0.4,
                              color="#10b981", label="Profit Factor")
        ax2.axhline(50, color="#3b82f6", linestyle="--", linewidth=0.5, alpha=0.5)
        ax2_twin.axhline(1.0, color="#10b981", linestyle="--", linewidth=0.5, alpha=0.5)
        ax2.set_xlabel("Window")
        ax2.set_ylabel("Win Rate (%)", color="#3b82f6")
        ax2_twin.set_ylabel("Profit Factor", color="#10b981")
        ax2.set_title("Per-Window Performance")
        ax2.set_xticks(x)
        ax2.grid(True, alpha=0.3)

        chart = f"{args.name}_chart.png"
        plt.tight_layout()
        chart = _save_fig_with_fallback(
            plt, chart, dpi=110, bbox_inches="tight")
        if chart:
            print(f"  [save] -> {chart}")
        else:
            print("  [save] chart skipped (permission denied)")
    except Exception as e:
        print(f"  [chart] skipped: {e}")


if __name__ == "__main__":
    sys.exit(main() or 0)
