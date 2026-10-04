"""
RL Trainer — PPO + TradingEnv
------------------------------
เทรน RL agent บนข้อมูลประวัติศาสตร์ EURUSD H1

ติดตั้ง:
    pip install stable-baselines3[extra] gymnasium

วิธีใช้:
    python rl_train.py EURUSD_H1.csv --steps 200000 --name rl_v1

output:
    artifacts/models/rl_v1/current.json  (published run pointer)
    artifacts/models/rl_v1/runs/<run_id>/rl_v1.zip
    artifacts/models/rl_v1/runs/<run_id>/rl_v1_norm.csv
    artifacts/models/rl_v1/runs/<run_id>/rl_v1.train.json
    artifacts/models/rl_v1/runs/<run_id>/best/best_model.zip
    artifacts/models/rl_v1/runs/<run_id>/logs/  (tensorboard logs)
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
if __name__ == "__main__":
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
from artifact_paths import ArtifactRun
from training_data import (load_dataset, feature_columns, split_training_data,
                           fit_preprocessing, apply_normalization,
                           check_selection_scope, validate_hyperparameters)


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


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file with OHLC + features")
    ap.add_argument("--steps", type=int, default=200_000, help="training steps")
    ap.add_argument("--name", default="rl_v1", help="model name (no .zip)")
    ap.add_argument("--window", type=int, default=10, help="state window size")
    ap.add_argument("--train_pct", type=float, default=0.8, help="train/test split")
    ap.add_argument("--eval_csv", default="", help="optional separate CSV for eval/test")
    ap.add_argument("--corr_threshold", type=float, default=1.0,
                    help="train-only correlation pruning (1 = constant features only)")
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
    ap.add_argument("--expected_recipe_sha256", default="", help="confirmed dashboard recipe fingerprint")
    return ap


def main():
    args = build_parser().parse_args()
    prepared = prepare_training(args)
    with ArtifactRun(args.name) as run:
        return train(args, run, prepared)


def _input_fingerprints(args):
    import hashlib
    files = {}
    for value in (args.csv, args.eval_csv, args.reward_profile_json, args.action_profile_json,
                  str(Path(args.csv).with_suffix(".params.json")),
                  str(Path(args.csv).with_suffix(".features.json"))):
        if not value:
            continue
        path = Path(value).resolve()
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            files[str(path)] = digest.hexdigest()
        else:
            files[str(path)] = None
    return files


def prepare_training(args):
    """Resolve and validate the exact recipe without writing training artifacts."""
    import copy
    args = copy.deepcopy(args)
    validate_hyperparameters(args)
    from artifact_paths import _model_home
    _model_home(args.name)
    files = _input_fingerprints(args)
    params_file = Path(args.csv).with_suffix(".params.json")
    if params_file.exists() and not isinstance(json.loads(params_file.read_text(encoding="utf-8-sig")), dict):
        raise ValueError("Collector sidecar must contain a JSON object")
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
        raise ValueError(f"Invalid reward overrides: {exc}") from exc
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
        raise ValueError(f"Invalid action profile: {exc}") from exc
    args.action_profile = action_profile_key

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
    if not net_arch or any(n <= 0 for n in net_arch):
        raise ValueError("Network layers must be positive integers")
    print(f"[net] arch = {net_arch}")

    print("=" * 60)
    print("  RL Trainer — TradingEnv + PPO")
    print("=" * 60)

    # ---------- load data ----------
    print(f"\n[load] {args.csv}")
    df = load_dataset(args.csv)
    source_rows = len(df)
    candidates = feature_columns(df)
    external = load_dataset(args.eval_csv, candidates) if args.eval_csv else None
    train_pct = float(args.train_pct)
    raw_train, raw_validation, raw_test, split = split_training_data(
        df, train_pct, args.window, external)
    check_selection_scope(args.csv, raw_train)
    feature_cols, norm, preprocessing = fit_preprocessing(
        raw_train, candidates, args.corr_threshold)
    selection_path = Path(args.csv).with_suffix(".features.json")
    if selection_path.exists():
        preprocessing["upstream_selection"] = json.loads(selection_path.read_text(encoding="utf-8"))
    feat_mean, feat_std = norm["mean"], norm["std"]
    train_df = apply_normalization(raw_train, feature_cols, norm)
    validation_df = (apply_normalization(raw_validation, feature_cols, norm)
                     if raw_validation is not None else None)
    if raw_test is None:
        test_df = train_df.tail(min(len(train_df), max(args.ep_len, args.window + 3))).copy()
        eval_source = "in-sample diagnostic; no unseen Test"
    else:
        test_df = apply_normalization(raw_test, feature_cols, norm)
        eval_source = "held-out Test (not used by EvalCallback)"

    import hashlib
    if files != _input_fingerprints(args):
        raise ValueError("Data or recipe changed during preparation. Review the training summary again.")
    recipe = dict(
        schema="metafxclub.confirmed_training.v1",
        settings={k: v for k, v in vars(args).items() if k != "expected_recipe_sha256"},
        net_arch=net_arch, effective_episode_steps=min(args.ep_len, len(train_df)-args.window-2),
        train_period=_period(train_df), validation_period=_period(validation_df) if validation_df is not None else None,
        test_period=_period(raw_test) if raw_test is not None else None,
        train_rows=len(train_df), validation_rows=len(validation_df) if validation_df is not None else 0,
        test_rows=len(raw_test) if raw_test is not None else 0,
        evaluation_role=eval_source, features=feature_cols, preprocessing=preprocessing,
        normalization=norm.to_dict(), reward_config=reward_profile_cfg,
        reward_formula=args.reward_formula, action_config=action_profile_cfg,
        input_sha256=files,
        output_root=str((__import__("artifact_paths").MODELS_DIR / args.name / "runs").resolve()),
    )
    recipe = _jsonable(recipe)
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if args.expected_recipe_sha256 and args.expected_recipe_sha256 != digest:
        raise ValueError("Data or recipe changed after confirmation. Review the training summary again.")
    return {
        "args": args, "net_arch": net_arch, "source_rows": source_rows, "candidates": candidates,
        "train_pct": train_pct, "raw_train": raw_train, "raw_validation": raw_validation,
        "raw_test": raw_test, "split": split, "feature_cols": feature_cols, "norm": norm,
        "preprocessing": preprocessing, "feat_mean": feat_mean, "feat_std": feat_std,
        "train_df": train_df, "validation_df": validation_df, "test_df": test_df,
        "eval_source": eval_source, "reward_overrides": reward_overrides,
        "reward_profile_json_meta": reward_profile_json_meta, "reward_profile_cfg": reward_profile_cfg,
        "action_profile_json_meta": action_profile_json_meta, "action_profile_cfg": action_profile_cfg,
        "recipe": recipe, "recipe_sha256": digest,
    }


def train(args, run, prepared=None):
    prepared = prepared or prepare_training(args)
    args = prepared["args"]
    net_arch = prepared["net_arch"]
    source_rows = prepared["source_rows"]
    candidates = prepared["candidates"]
    train_pct = prepared["train_pct"]
    raw_train = prepared["raw_train"]
    raw_validation = prepared["raw_validation"]
    raw_test = prepared["raw_test"]
    split = prepared["split"]
    feature_cols = prepared["feature_cols"]
    norm = prepared["norm"]
    preprocessing = prepared["preprocessing"]
    feat_mean = prepared["feat_mean"]
    feat_std = prepared["feat_std"]
    train_df = prepared["train_df"]
    validation_df = prepared["validation_df"]
    test_df = prepared["test_df"]
    eval_source = prepared["eval_source"]
    reward_overrides = prepared["reward_overrides"]
    reward_profile_json_meta = prepared["reward_profile_json_meta"]
    reward_profile_cfg = prepared["reward_profile_cfg"]
    action_profile_json_meta = prepared["action_profile_json_meta"]
    action_profile_cfg = prepared["action_profile_cfg"]
    model_root = ensure_model_dirs(args.name)
    norm_path = artifact_norm_path(args.name)
    norm.to_csv(norm_path)
    import shutil as _shutil
    src_params = Path(args.csv).with_suffix(".params.json")
    if src_params.exists():
        payload = json.loads(src_params.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Collector sidecar must contain a JSON object")
        _shutil.copy(src_params, artifact_params_path(args.name))
    print(f"[features] {len(candidates)} -> {len(feature_cols)}; dropped: {preprocessing['dropped']}")
    print(f"[split] train: {len(train_df):,} | validation: {len(validation_df) if validation_df is not None else 0:,} | test: {len(raw_test) if raw_test is not None else 0:,}")
    print(f"[eval] {eval_source}")
    print(f"[period] Train: {_period(train_df)}")
    print(f"[period] Validation: {_period(validation_df) if validation_df is not None else 'none'}")
    print(f"[period] Test/diagnostic: {_period(test_df)}")
    algo_hparams = {
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range if args.algo == "ppo" else None,
        "ent_coef": args.ent_coef if args.algo in ("ppo", "a2c") else None,
        "n_steps": args.n_steps if args.algo in ("ppo", "a2c") else None,
        "n_epochs": args.n_epochs if args.algo == "ppo" else None,
        "batch_size": args.batch_size if args.algo in ("ppo", "dqn") else None,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda if args.algo in ("ppo", "a2c") else None,
        "vf_coef": args.vf_coef if args.algo in ("ppo", "a2c") else None,
    }
    meta_path = train_meta_path(args.name)
    meta = {
        "model": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_csv": args.csv,
        "eval_csv": args.eval_csv or "",
        "eval_source": eval_source,
        "environment_version": "2.0-episode-liquidation",
        "confirmed_recipe": prepared["recipe"],
        "recipe_sha256": prepared["recipe_sha256"],
        "preprocessing": preprocessing,
        "validation_rows": len(validation_df) if validation_df is not None else 0,
        "validation_period": _period(validation_df) if validation_df is not None else None,
        "test_is_unseen": raw_test is not None,
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
            "reward_profile_active": args.reward_mode != "mtm",
            "action_profile": args.action_profile,
            "action_profile_label": action_profile_cfg["label"] if args.action_profile == "custom" else action_profile_label(args.action_profile),
            "action_profile_json": action_profile_json_meta,
            "action_profile_params": action_profile_cfg["params"],
            "action_profile_config": action_profile_cfg,
            "max_hold": args.max_hold,
            "net_arch": net_arch,
            **algo_hparams,
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
            reward_overrides=reward_overrides,
            reward_formula=args.reward_formula,
            action_profile=action_profile_cfg,
            max_hold_bars=args.max_hold,
        ))

    def make_eval_env():
        return Monitor(TradingEnv(
            validation_df, feature_cols,
            window_size=args.window,
            max_steps=len(validation_df) - args.window - 2,
            random_start=False,
            reward_mode=args.reward_mode,
            reward_profile=args.reward_profile,
            reward_overrides=reward_overrides,
            reward_formula=args.reward_formula,
            action_profile=action_profile_cfg,
            max_hold_bars=args.max_hold,
        ))

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env]) if validation_df is not None else None

    # ---------- create model ----------
    print(f"\n[model] {args.algo.upper()}")
    print(f"[reward] mode={args.reward_mode} | profile={reward_profile_label(args.reward_profile)}")
    if args.reward_mode == "mtm" and (reward_overrides or args.reward_formula.strip()):
        print("[warn] MTM reward mode ignores reward profile overrides/formula")
    print(f"[action] profile={action_profile_cfg['label']} | actions={len(action_profile_cfg['actions'])}")
    if args.reward_formula.strip():
        print("[reward] developer formula enabled")
    log_dir = str(logs_dir(args.name))

    if args.algo == "ppo":
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
            learning_rate=args.learning_rate,
            buffer_size=50_000,
            batch_size=args.batch_size,
            gamma=args.gamma,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=500,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=net_arch),
        )
        print(f"[hyper] lr={args.learning_rate}, batch={args.batch_size}, gamma={args.gamma}")
        print("[hyper] DQN ignores PPO-only clip/epochs/GAE/entropy/value settings")
    else:  # a2c
        model = A2C(
            "MlpPolicy", train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=net_arch),
        )
        print(f"[hyper] lr={args.learning_rate}, n_steps={args.n_steps}, gamma={args.gamma}")
        print(f"[hyper] gae_lambda={args.gae_lambda}, ent={args.ent_coef}, vf={args.vf_coef}")
        print("[hyper] A2C ignores PPO-only clip/epochs and GUI batch size")

    # ---------- train ----------
    print(f"\n[train] {args.steps:,} steps ...")
    print(f"(progress bar will appear; tensorboard --logdir {log_dir} to monitor)")

    from stable_baselines3.common.callbacks import EvalCallback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir(args.name)),
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=0,
    ) if eval_env is not None else None

    model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=True)

    # ---------- save ----------
    save_path = final_model_path(args.name)
    model.save(save_path)
    print(f"\n[save] -> {save_path}")

    # ---------- quick test ----------
    print("\n[eval] quick run on test set ...")
    test_env_raw = TradingEnv(test_df, feature_cols, window_size=args.window,
                              random_start=False,
                              max_steps=len(test_df) - args.window - 2,
                              reward_mode=args.reward_mode,
                              reward_profile=args.reward_profile,
                              reward_overrides=reward_overrides,
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
    meta["actual_timesteps"] = model.num_timesteps
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
    run.publish(model)
    train_env.close()
    if eval_env is not None:
        eval_env.close()
    print("\n" + "=" * 50)
    print(f"  Quick evaluation: {eval_source}")
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
