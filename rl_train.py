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
from reward_profiles import (
    REWARD_PROFILES,
    coerce_reward_overrides,
    get_reward_profile,
    load_reward_profile_json,
    reward_profile_label,
)
from reward_formula import validate_reward_formula
from action_profiles import (
    ACTION_PROFILES,
    action_profile_label,
    coerce_action_params,
    get_action_profile,
    load_action_profile_json,
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
    ap.add_argument("--reward_profile", default="balanced",
                    choices=list(REWARD_PROFILES.keys()),
                    help="reward preset: balanced, anti_overtrade, low_drawdown, trend_follower, scalper")
    ap.add_argument("--reward_overrides", default="",
                    help="JSON object of safe reward slider overrides, e.g. '{\"trade_penalty\":0.008}'")
    ap.add_argument("--reward_profile_json", default="",
                    help="path to reward profile JSON recipe; CLI overrides win if both are provided")
    ap.add_argument("--reward_formula", default="",
                    help="developer mode safe reward expression; only used with reward_mode=realized")
    ap.add_argument("--action_profile", default="basic_4",
                    choices=list(ACTION_PROFILES.keys()),
                    help="action preset: basic_4 or manage_6")
    ap.add_argument("--action_params", default="",
                    help="JSON object of safe action parameter overrides, e.g. '{\"trail_atr_mult\":2.5}'")
    ap.add_argument("--action_profile_json", default="",
                    help="path to limited action profile JSON recipe; CLI action_params win")
    ap.add_argument("--max_hold", type=int, default=30,
                    help="บังคับปิด position ถ้าถือเกินกี่ bars")
    ap.add_argument("--net_arch", default="auto",
                    help="NN layers e.g. '128,64' or 'auto' (scale by window)")
    ap.add_argument("--mc_eval", type=int, default=1000,
                    help="MC robustness runs on the quick eval (0 = off)")
    ap.add_argument("--mc_skip_frac", type=float, default=0.10,
                    help="fraction of trades dropped per skip-MC run (0.10 = 10%%)")

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
    try:
        reward_overrides = {}
        reward_profile_json_meta = None
        cli_reward_formula = args.reward_formula.strip()
        if args.reward_profile_json.strip():
            reward_profile_json_meta = load_reward_profile_json(args.reward_profile_json.strip())
            args.reward_profile = reward_profile_json_meta["base_profile"]
            reward_overrides.update(reward_profile_json_meta["overrides"])
            if reward_profile_json_meta.get("formula") and not cli_reward_formula:
                args.reward_formula = reward_profile_json_meta["formula"]
        if cli_reward_formula:
            args.reward_formula = cli_reward_formula
        if args.reward_overrides.strip():
            reward_overrides.update(coerce_reward_overrides(json.loads(args.reward_overrides)))
        if args.reward_formula.strip():
            validate_reward_formula(args.reward_formula.strip())
        reward_profile_key, reward_profile_cfg = get_reward_profile(args.reward_profile, reward_overrides)
    except Exception as exc:
        print(f"ERROR: invalid reward overrides: {exc}")
        sys.exit(1)
    args.reward_profile = reward_profile_key

    try:
        action_params = {}
        action_profile_json_meta = None
        action_profile_value = args.action_profile
        if args.action_profile_json.strip():
            action_profile_json_meta = load_action_profile_json(args.action_profile_json.strip())
            action_profile_value = action_profile_json_meta["profile"]
            action_params.update(action_profile_json_meta["params"])
        if args.action_params.strip():
            action_params.update(coerce_action_params(json.loads(args.action_params)))
        action_profile_key, action_profile_cfg = get_action_profile(action_profile_value, action_params)
    except Exception as exc:
        print(f"ERROR: invalid action profile: {exc}")
        sys.exit(1)
    args.action_profile = action_profile_key
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
            "reward_profile": args.reward_profile,
            "reward_profile_label": reward_profile_label(args.reward_profile),
            "reward_profile_json": reward_profile_json_meta,
            "reward_profile_overrides": reward_overrides,
            "reward_profile_config": reward_profile_cfg,
            "reward_formula": args.reward_formula.strip(),
            "action_profile": args.action_profile,
            "action_profile_label": action_profile_cfg["label"] if args.action_profile == "custom" else action_profile_label(args.action_profile),
            "action_profile_json": action_profile_json_meta,
            "action_profile_params": action_profile_cfg["params"],
            "action_profile_config": action_profile_cfg,
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
            reward_profile=args.reward_profile,
            reward_formula=args.reward_formula,
            action_profile=action_profile_cfg,
            max_hold_bars=args.max_hold,
        ))

    def make_eval_env():
        return Monitor(TradingEnv(
            test_df, feature_cols,
            window_size=args.window,
            max_steps=len(test_df) - args.window - 2,
            reward_mode=args.reward_mode,
            reward_profile=args.reward_profile,
            reward_formula=args.reward_formula,
            action_profile=action_profile_cfg,
            max_hold_bars=args.max_hold,
        ))

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    # ---------- create model ----------
    print(f"\n[model] {args.algo.upper()}")
    print(f"[reward] mode={args.reward_mode} | profile={reward_profile_label(args.reward_profile)}")
    print(f"[action] profile={action_profile_cfg['label']} | actions={len(action_profile_cfg['actions'])}")
    if args.reward_formula.strip():
        print("[reward] developer formula enabled")
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
                              reward_profile=args.reward_profile,
                              reward_formula=args.reward_formula,
                              action_profile=action_profile_cfg,
                              max_hold_bars=args.max_hold)
    obs, _ = test_env_raw.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_raw.step(int(action))
        done = terminated or truncated

    stats = test_env_raw.get_stats()
    meta["quick_eval_stats"] = stats

    # ---------- MC robustness — measured from the very first training ----------
    # Two Monte Carlo probes on the quick-eval trade list so EVERY model gets
    # robustness numbers at birth (not only when someone runs a full backtest):
    #   1) order-shuffle -> drawdown distribution   (sizing / survival risk)
    #   2) skip 10% of trades -> profit retention   (is the edge concentrated
    #      in a few lucky trades? SQX-style criterion: retention >= ~70%)
    mc_meta = None
    trade_rets = np.array([t.get("pnl", 0.0) for t in test_env_raw.trades], dtype=float)
    n_mc = int(getattr(args, 'mc_eval', 1000) or 0)
    skip_frac = min(max(float(getattr(args, 'mc_skip_frac', 0.10)), 0.01), 0.5)
    if n_mc > 0 and len(trade_rets) >= 10:
        rng = np.random.default_rng(7)
        dds = np.empty(n_mc)
        for _k in range(n_mc):
            eq = np.concatenate(([1.0], np.cumprod(1.0 + rng.permutation(trade_rets))))
            pk = np.maximum.accumulate(eq)
            dds[_k] = ((eq - pk) / pk).min()

        base_ret = float(np.prod(1.0 + trade_rets) - 1)
        keep = max(1, int(len(trade_rets) * (1.0 - skip_frac)))
        finals = np.empty(n_mc)
        for _k in range(n_mc):
            pick = rng.choice(trade_rets, size=keep, replace=False)
            finals[_k] = np.prod(1.0 + pick) - 1
        p95_worst_ret = float(np.percentile(finals, 5))
        flip_rate = float((finals <= 0).mean()) if base_ret > 0 else float((finals > 0).mean())
        retention = (p95_worst_ret / base_ret) if base_ret > 0 else None

        print("\n" + "=" * 50)
        print(f"  MC robustness (quick eval, {n_mc:,} runs)")
        print("=" * 50)
        print(f"  DD across orderings : median {np.median(dds):.2%} | "
              f"95% worst {np.percentile(dds, 5):.2%} | worst {dds.min():.2%}")
        print(f"  Skip {skip_frac:.0%} of trades : base {base_ret:+.2%} | "
              f"p95-worst {p95_worst_ret:+.2%}"
              + (f" | retention {retention:.0%}" if retention is not None else ""))
        if base_ret > 0:
            print(f"  P(profit flips to loss when {skip_frac:.0%} skipped): {flip_rate:.1%}")
            if retention is not None and retention >= 0.7 and flip_rate < 0.05:
                print(f"  verdict: edge is well-distributed across trades")
            elif retention is not None and retention < 0.4:
                print(f"  verdict: edge CONCENTRATED in few trades — fragile")
            else:
                print(f"  verdict: moderate concentration — check on full backtest")
        else:
            print(f"  (strategy not profitable on quick eval — "
                  f"retention criterion not applicable)")

        mc_meta = {
            "n_trades": int(len(trade_rets)),
            "dd_median": float(np.median(dds)),
            "dd_p95_worst": float(np.percentile(dds, 5)),
            "dd_worst": float(dds.min()),
            "skip_frac": skip_frac,
            "base_return": base_ret,
            "skip_p95_worst_return": p95_worst_ret,
            "profit_retention": retention,
            "flip_rate": flip_rate,
        }
    meta["quick_eval_mc"] = mc_meta

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
