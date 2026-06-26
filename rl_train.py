"""
RL Trainer — PPO + TradingEnv
------------------------------
เทรน RL agent บนข้อมูลประวัติศาสตร์ EURUSD H1

ติดตั้ง:
    pip install stable-baselines3[extra] gymnasium

วิธีใช้:
    python rl_train.py EURUSD_H1.csv --steps 200000 --name rl_v1

output:
    artifacts/models/rl_v1/rl_v1.zip
    artifacts/models/rl_v1/rl_v1_norm.csv
    artifacts/models/rl_v1/rl_v1.train.json
    artifacts/models/rl_v1/best/best_model.zip
    artifacts/models/rl_v1/logs/        (tensorboard logs)
"""
import sys
import io
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Windows console UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from artifact_paths import (
    best_dir,
    ensure_model_dirs,
    final_model_path,
    logs_dir,
    norm_path as artifact_norm_path,
    params_path as artifact_params_path,
    train_meta_path,
)
from trading_env import TradingEnv


def _period(df):
    if "timestamp" not in df.columns or len(df) == 0:
        return None
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return None
    return {
        "start": ts.iloc[0].isoformat(),
        "end": ts.iloc[-1].isoformat(),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file with OHLC + features")
    ap.add_argument("--steps", type=int, default=200_000, help="training steps")
    ap.add_argument("--name", default="rl_v1", help="model name (no .zip)")
    ap.add_argument("--window", type=int, default=10, help="state window size")
    ap.add_argument("--train_pct", type=float, default=0.8, help="train/test split")
    ap.add_argument("--eval_csv", default="", help="optional separate CSV for eval/test")
    ap.add_argument("--ep_len", type=int, default=2000, help="bars per episode")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "a2c"])
    ap.add_argument("--reward_mode", default="realized",
                    choices=["realized", "mtm"],
                    help="realized=ปิดแล้วถึงให้ reward (แนะนำ), mtm=ทุก step (buy-hold trap)")
    ap.add_argument("--max_hold", type=int, default=30,
                    help="บังคับปิด position ถ้าถือเกินกี่ bars")
    ap.add_argument("--net_arch", default="auto",
                    help="NN layers e.g. '128,64' or 'auto' (scale by window)")

    # Advanced PPO hyperparameters
    ap.add_argument("--learning_rate", type=float, default=3e-4,
                    help="learning rate (default 3e-4) — ลดถ้า kl/clip สูง")
    ap.add_argument("--clip_range", type=float, default=0.2,
                    help="PPO clip range (default 0.2)")
    ap.add_argument("--ent_coef", type=float, default=0.01,
                    help="entropy coefficient (default 0.01) — เพิ่มถ้า exploration หาย")
    # Extended PPO hyperparameters (rollout / optim / value)
    ap.add_argument("--n_steps", type=int, default=2048,
                    help="rollout buffer size per env (default 2048)")
    ap.add_argument("--n_epochs", type=int, default=10,
                    help="PPO update epochs per rollout (default 10)")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="minibatch size (default 64) — ต้อง <= n_steps")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="discount factor (default 0.99)")
    ap.add_argument("--gae_lambda", type=float, default=0.95,
                    help="GAE lambda — advantage smoothing (default 0.95)")
    ap.add_argument("--vf_coef", type=float, default=0.5,
                    help="value function loss coefficient (default 0.5)")
    args = ap.parse_args()
    model_root = ensure_model_dirs(args.name)

    # Auto-scale NN by window size
    if args.net_arch == "auto":
        if args.window >= 50:
            net_arch = [512, 256, 128]   # window ใหญ่ -> NN ใหญ่
        elif args.window >= 20:
            net_arch = [256, 128, 64]
        else:
            net_arch = [128, 64]
    else:
        net_arch = [int(x) for x in args.net_arch.split(",")]
    print(f"[net] arch = {net_arch}")

    print("=" * 60)
    print("  RL Trainer — TradingEnv + PPO")
    print("=" * 60)

    # ---------- load data ----------
    print(f"\n[load] {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  rows: {len(df):,}")
    source_rows = len(df)

    # drop leaky columns + non-feature columns
    leaky = [c for c in df.columns
             if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))]
    if leaky:
        print(f"  drop leaky/target: {leaky}")
        df = df.drop(columns=leaky)

    # detect feature columns (numeric, exclude OHLC + ids)
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  features ({len(feature_cols)}): {feature_cols}")

    # parse timestamp + sort
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # ---------- split index (computed BEFORE normalize to avoid test leakage) ----------
    train_pct = min(max(float(args.train_pct), 0.01), 1.0)
    split = int(len(df) * train_pct)

    # normalize features using TRAIN-ONLY stats (RL ชอบ scaled input)
    # mean/std from the train slice only -> no look-ahead leakage from test.
    # With --eval_csv, train_pct is typically 1.0 so stats cover the full train
    # file, and the separate eval set is normalized with these same stats below.
    print("[normalize] z-score features (train-only stats) ...")
    train_slice = df[feature_cols].iloc[:split]
    feat_mean = train_slice.mean()
    feat_std = train_slice.std() + 1e-8
    df[feature_cols] = (df[feature_cols] - feat_mean) / feat_std
    # fill NaN
    df = df.fillna(0).reset_index(drop=True)

    # save normalization stats so inference (EA) uses identical scaling
    norm_path = artifact_norm_path(args.name)
    pd.DataFrame({"mean": feat_mean, "std": feat_std}).to_csv(norm_path)
    print(f"  saved norm stats -> {norm_path}")

    # Forward params.json sidecar from input CSV → model artifacts (for export)
    import shutil as _shutil
    src_params = Path(args.csv).with_suffix("").with_suffix(".params.json")
    if not src_params.exists():
        src_params = Path(args.csv).parent / (Path(args.csv).stem + ".params.json")
    if src_params.exists():
        dst_params = artifact_params_path(args.name)
        _shutil.copy(src_params, dst_params)
        print(f"  forwarded params -> {dst_params}")

    # ---------- split ----------
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    eval_source = "internal split"

    if args.eval_csv:
        print(f"\n[load] eval: {args.eval_csv}")
        test_df = pd.read_csv(args.eval_csv)
        print(f"  rows: {len(test_df):,}")

        leaky_eval = [c for c in test_df.columns
                      if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))]
        if leaky_eval:
            print(f"  drop eval leaky/target: {leaky_eval}")
            test_df = test_df.drop(columns=leaky_eval)

        if "timestamp" in test_df.columns:
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")
            test_df = test_df.sort_values("timestamp").reset_index(drop=True)

        for col in feature_cols:
            if col not in test_df.columns:
                print(f"  [warn] eval missing feature {col}; filling 0")
                test_df[col] = 0.0
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        test_df[feature_cols] = (test_df[feature_cols] - feat_mean) / feat_std
        test_df = test_df.fillna(0).reset_index(drop=True)
        eval_source = Path(args.eval_csv).name
    elif len(test_df) <= args.window + 2:
        fallback_rows = min(len(train_df), max(args.window + 3, min(args.ep_len, len(train_df))))
        test_df = train_df.tail(fallback_rows).copy().reset_index(drop=True)
        eval_source = "train tail fallback"
        print("  [warn] eval split is empty/small; using train tail for EvalCallback")

    if len(train_df) <= args.window + 2:
        raise SystemExit(
            f"ERROR: train data too small ({len(train_df)} rows). Need > window+2 rows.")
    if len(test_df) <= args.window + 2:
        raise SystemExit(
            f"ERROR: eval/test data too small ({len(test_df)} rows). Need > window+2 rows.")

    print(f"\n[split] train: {len(train_df):,} | eval: {len(test_df):,} ({eval_source})")
    meta_path = train_meta_path(args.name)
    meta = {
        "model": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_csv": args.csv,
        "eval_csv": args.eval_csv or "",
        "eval_source": eval_source,
        "train_pct": train_pct,
        "split_index": split,
        "source_rows": source_rows,
        "train_rows": len(train_df),
        "eval_rows": len(test_df),
        "train_period": _period(train_df),
        "eval_period": _period(test_df),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "artifacts_dir": str(model_root),
        "artifacts": {
            "model": str(final_model_path(args.name)),
            "best_model": str(best_dir(args.name) / "best_model.zip"),
            "norm": str(norm_path),
            "params": str(artifact_params_path(args.name)),
            "logs": str(logs_dir(args.name)),
            "train_meta": str(meta_path),
        },
        "hyperparameters": {
            "steps": args.steps,
            "window": args.window,
            "ep_len": args.ep_len,
            "algo": args.algo,
            "reward_mode": args.reward_mode,
            "max_hold": args.max_hold,
            "net_arch": net_arch,
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "n_steps": args.n_steps,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "vf_coef": args.vf_coef,
        },
    }
    meta_path.write_text(json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[meta] -> {meta_path}")

    # ---------- create environments ----------
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_train_env():
        return Monitor(TradingEnv(
            train_df, feature_cols,
            window_size=args.window,
            max_steps=args.ep_len,
            reward_mode=args.reward_mode,
            max_hold_bars=args.max_hold,
        ))

    def make_eval_env():
        return Monitor(TradingEnv(
            test_df, feature_cols,
            window_size=args.window,
            max_steps=len(test_df) - args.window - 2,
            reward_mode=args.reward_mode,
            max_hold_bars=args.max_hold,
        ))

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    # ---------- create model ----------
    print(f"\n[model] {args.algo.upper()}")
    log_dir = str(logs_dir(args.name))

    if args.algo == "ppo":
        # Sanity: batch_size must be <= n_steps and n_steps % batch_size == 0
        if args.batch_size > args.n_steps:
            print(f"[warn] batch_size({args.batch_size}) > n_steps({args.n_steps}); clamping to n_steps")
            args.batch_size = args.n_steps

        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,           # encourage exploration
            vf_coef=args.vf_coef,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=net_arch),
        )
        print(f"[hyper] lr={args.learning_rate}, clip={args.clip_range}, ent={args.ent_coef}")
        print(f"[hyper] n_steps={args.n_steps}, batch={args.batch_size}, epochs={args.n_epochs}")
        print(f"[hyper] gamma={args.gamma}, gae_lambda={args.gae_lambda}, vf_coef={args.vf_coef}")
    elif args.algo == "dqn":
        model = DQN(
            "MlpPolicy", train_env,
            learning_rate=1e-4,
            buffer_size=50_000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=500,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=net_arch),
        )
    else:  # a2c
        model = A2C("MlpPolicy", train_env, learning_rate=7e-4,
                    verbose=1, tensorboard_log=log_dir)

    # ---------- train ----------
    print(f"\n[train] {args.steps:,} steps ...")
    print(f"(progress bar will appear; tensorboard --logdir {log_dir} to monitor)")

    from stable_baselines3.common.callbacks import EvalCallback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir(args.name)),
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=True)

    # ---------- save ----------
    save_path = final_model_path(args.name)
    model.save(save_path)
    print(f"\n[save] -> {save_path}")

    # ---------- quick test ----------
    print("\n[eval] quick run on test set ...")
    test_env_raw = TradingEnv(test_df, feature_cols, window_size=args.window,
                              max_steps=len(test_df) - args.window - 2,
                              reward_mode=args.reward_mode,
                              max_hold_bars=args.max_hold)
    obs, _ = test_env_raw.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_raw.step(int(action))
        done = terminated or truncated

    stats = test_env_raw.get_stats()
    meta["quick_eval_stats"] = stats
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[meta] updated -> {meta_path}")
    print("\n" + "=" * 50)
    print("  Quick test on out-of-sample")
    print("=" * 50)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v}")
    print("=" * 50)
    print(f"\nรัน: python rl_backtest.py {args.name} {args.csv}  เพื่อดู backtest เต็ม")


if __name__ == "__main__":
    main()
