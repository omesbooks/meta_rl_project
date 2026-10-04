"""
RL fine-tuning for an existing PPO model.

Fine-tune must keep the base model contract intact:
- same feature columns
- same window / observation shape
- same action profile and output dimension
- same reward settings unless the model is retrained from scratch

This script reads the base model metadata when available and writes the new
model into artifacts/models/<name>/ so Backtest and Export can discover it.
"""
import argparse
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from action_profiles import get_action_profile, profile_for_action_count
from artifact_paths import (
    ensure_model_dirs,
    final_model_path,
    find_model_path,
    find_norm_path,
    find_params_path,
    logs_dir,
    norm_path as artifact_norm_path,
    params_path as artifact_params_path,
    train_meta_path,
)
from trading_env import TradingEnv
from trading_env import SegmentedTradingEnv
from artifact_paths import ArtifactRun
from training_data import load_dataset, apply_normalization
from reward_profiles import REWARD_PARAM_SPEC_BY_KEY, get_reward_profile


def _period(df):
    if "timestamp" not in df.columns or len(df) == 0:
        return None
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return None
    return {"start": ts.iloc[0].isoformat(), "end": ts.iloc[-1].isoformat()}


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


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path):
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load_csv(path):
    df = load_dataset(path)
    leaky = [
        c for c in df.columns
        if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))
    ]
    if leaky:
        print(f"  drop leaky/target: {leaky}")
        df = df.drop(columns=leaky)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _detect_feature_cols(df):
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def _csv_params_path(csv_path):
    path = Path(csv_path)
    return path.with_suffix(".params.json")


def _resolve_base_contract(args, old_df):
    from stable_baselines3 import PPO

    base_path = find_model_path(args.base_model, "final")
    if base_path is None:
        raise SystemExit(f"ERROR: base model not found: {args.base_model}")

    base_meta_path = train_meta_path(args.base_model)
    base_meta = _load_json(base_meta_path)
    hparams = base_meta.get("hyperparameters", {})

    probe_model = PPO.load(str(base_path))
    model_actions = int(probe_model.action_space.n)
    obs_dim = int(probe_model.observation_space.shape[0])

    feature_cols = list(base_meta.get("features") or [])
    if not feature_cols:
        base_norm = find_norm_path(args.base_model)
        if base_norm is None:
            raise SystemExit("ERROR: original normalization is required for fine-tune")
        feature_cols = list(pd.read_csv(base_norm, index_col=0).index)

    if not feature_cols:
        raise SystemExit("ERROR: no numeric feature columns found.")

    window = args.window or int(hparams.get("window") or 0)
    if not window:
        raw = obs_dim - 3
        if raw > 0 and raw % len(feature_cols) == 0:
            window = raw // len(feature_cols)
    if not window:
        raise SystemExit("ERROR: could not resolve window size from metadata/model.")

    expected_obs = window * len(feature_cols) + 3
    if expected_obs != obs_dim:
        raise SystemExit(
            f"ERROR: observation shape mismatch. base model obs={obs_dim}, "
            f"resolved window={window}, features={len(feature_cols)} -> {expected_obs}. "
            "Use the original feature set/window for fine-tune."
        )

    action_profile_value = (
        hparams.get("action_profile_config")
        or hparams.get("action_profile")
        or None
    )
    action_params = hparams.get("action_profile_params") or {}
    if action_profile_value is None:
        action_key, action_profile_cfg = profile_for_action_count(model_actions)
    else:
        action_key, action_profile_cfg = get_action_profile(action_profile_value, action_params)

    if len(action_profile_cfg["actions"]) != model_actions:
        raise SystemExit(
            f"ERROR: action profile has {len(action_profile_cfg['actions'])} actions "
            f"but base model outputs {model_actions}."
        )

    max_hold = args.max_hold if args.max_hold is not None else int(hparams.get("max_hold", 30))
    ep_len = args.ep_len if args.ep_len is not None else int(hparams.get("ep_len", 2000))
    reward_mode = hparams.get("reward_mode", "realized")
    reward_profile = hparams.get("reward_profile", "balanced")
    reward_profile_config = hparams.get("reward_profile_config") or {}
    reward_overrides = {k: v for k, v in reward_profile_config.items() if k in REWARD_PARAM_SPEC_BY_KEY}
    if not reward_overrides:
        reward_overrides = hparams.get("reward_profile_overrides") or {}
    _, reward_profile_config = get_reward_profile(reward_profile, reward_overrides)
    reward_formula = hparams.get("reward_formula", "")

    return {
        "base_path": base_path,
        "base_meta_path": base_meta_path if base_meta_path.exists() else None,
        "base_meta": base_meta,
        "feature_cols": feature_cols,
        "window": int(window),
        "max_hold": int(max_hold),
        "ep_len": int(ep_len),
        "reward_mode": reward_mode,
        "reward_profile": reward_profile,
        "reward_profile_config": reward_profile_config,
        "reward_overrides": reward_overrides,
        "reward_formula": reward_formula,
        "action_key": action_key,
        "action_profile": action_profile_cfg,
        "model_actions": model_actions,
        "obs_dim": obs_dim,
    }


def _require_features(df, feature_cols, label):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise SystemExit(
            f"ERROR: {label} is missing {len(missing)} base-model features: "
            f"{preview}{suffix}"
        )


def _copy_params_sidecar(base_model, out_name, old_csv, new_csv):
    candidates = [
        find_params_path(base_model),
        _csv_params_path(new_csv) if new_csv else None,
        _csv_params_path(old_csv),
    ]
    src = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if not src:
        print("[params] no .params.json sidecar found to forward")
        return ""
    dst = artifact_params_path(out_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[params] forwarded -> {dst}")
    return str(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_model", help="base model name (without .zip)")
    ap.add_argument("--old_csv", required=True, help="older CSV used for replay/mixing")
    ap.add_argument("--new_csv", help="new CSV for fine-tuning")
    ap.add_argument("--steps", type=int, default=50_000, help="fine-tune steps")
    ap.add_argument("--mix_ratio", type=float, default=0.3, help="old-data ratio in mixed mode")
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tune learning rate")
    ap.add_argument("--mode", default="mixed", choices=["pure", "mixed", "replay"])
    ap.add_argument("--window", type=int, default=0, help="0 = use base model metadata")
    ap.add_argument("--max_hold", type=int, default=None, help="default = base model metadata")
    ap.add_argument("--ep_len", type=int, default=None, help="default = base model metadata")
    ap.add_argument("--name", default=None, help="output model name")
    args = ap.parse_args()
    if args.steps <= 0 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Steps and learning rate must be positive")

    if not (0 <= args.mix_ratio < 1):
        raise SystemExit("ERROR: --mix_ratio must be >= 0 and < 1")

    out_name = args.name or f"{args.base_model}_ft"
    if out_name.casefold() == args.base_model.casefold():
        raise SystemExit(
            "ERROR: --name must differ from the base model "
            "(would overwrite its weights and train metadata)"
        )
    with ArtifactRun(out_name) as run:
        return finetune(args, out_name, run)


def finetune(args, out_name, run):
    from artifact_paths import pin_model_generation
    pin_model_generation(args.base_model)
    model_root = ensure_model_dirs(out_name)
    meta_path = train_meta_path(out_name)

    print("=" * 60)
    print("  RL Fine-tuning")
    print("=" * 60)
    print(f"  Base model    : {args.base_model}")
    print(f"  Mode          : {args.mode}")
    print(f"  Steps         : {args.steps:,}")
    print(f"  Mix ratio     : {args.mix_ratio:.0%} old / {1 - args.mix_ratio:.0%} new")
    print(f"  Learning rate : {args.lr}")
    print(f"  Output        : {out_name}")
    print("=" * 60)

    print(f"\n[load] old_csv: {args.old_csv}")
    old_df = _load_csv(args.old_csv)
    print(f"  rows: {len(old_df):,}")

    new_df = None
    if args.new_csv:
        print(f"\n[load] new_csv: {args.new_csv}")
        new_df = _load_csv(args.new_csv)
        print(f"  rows: {len(new_df):,}")
    elif args.mode in ("pure", "mixed"):
        raise SystemExit("ERROR: --new_csv is required for pure/mixed fine-tune modes.")

    contract = _resolve_base_contract(args, old_df)
    feature_cols = contract["feature_cols"]
    _require_features(old_df, feature_cols, "old_csv")
    if new_df is not None:
        _require_features(new_df, feature_cols, "new_csv")

    print("\n[base contract]")
    print(f"  model path    : {contract['base_path']}")
    print(f"  metadata      : {contract['base_meta_path'] or '(none; inferred)'}")
    print(f"  features      : {len(feature_cols)}")
    print(f"  window        : {contract['window']}")
    print(f"  action profile: {contract['action_profile']['label']} ({contract['model_actions']} actions)")
    print(f"  max_hold      : {contract['max_hold']}")
    print(f"  ep_len        : {contract['ep_len']}")

    if new_df is not None and new_df.timestamp.min() <= old_df.timestamp.max():
        raise ValueError("New data must be strictly later than old data; remove overlapping rows")
    if args.mode == "pure":
        segments = [new_df.copy()]
    elif args.mode == "mixed":
        old_sample_n = min(len(old_df), int(len(new_df) * args.mix_ratio / (1 - args.mix_ratio)))
        segments = []
        if old_sample_n:
            if old_sample_n < contract["window"] + 3:
                raise ValueError("Old replay segment too short; increase mix ratio or supply more new data")
            start = int(np.random.default_rng(42).integers(len(old_df) - old_sample_n + 1))
            segments.append(old_df.iloc[start:start + old_sample_n].copy())
        segments.append(new_df.copy())
    else:
        segments = [old_df.copy()] + ([new_df.copy()] if new_df is not None else [])
    if any(len(frame) < contract["window"] + 3 for frame in segments):
        raise ValueError("Each replay segment needs at least window + 3 rows")
    segment_lengths = [len(frame) for frame in segments]
    train_df = pd.concat(segments, ignore_index=True)
    print(f"[segments] contiguous rows per segment: {segment_lengths}; episodes never cross boundaries")

    base_norm = find_norm_path(args.base_model)
    out_norm = artifact_norm_path(out_name)
    norm = None
    if base_norm and base_norm.exists():
        print(f"\n[norm] using base stats -> {base_norm}")
        norm = pd.read_csv(base_norm, index_col=0)
        train_df = apply_normalization(train_df, feature_cols, norm)
        out_norm.parent.mkdir(parents=True, exist_ok=True)
        norm.to_csv(out_norm)
    else:
        raise ValueError("Base normalization is required for fine-tune; restore the original artifact set")
    print(f"  saved -> {out_norm}")

    params_out = _copy_params_sidecar(args.base_model, out_name, args.old_csv, args.new_csv)
    train_df = train_df.fillna(0).reset_index(drop=True)

    meta = {
        "environment_version": "2.0-episode-liquidation",
        "segment_lengths": segment_lengths,
        "evaluation_role": "in-sample diagnostic on new training data",
        "model": out_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fine_tune": True,
        "base_model": args.base_model,
        "base_model_path": str(contract["base_path"]),
        "base_train_meta": str(contract["base_meta_path"] or ""),
        "old_csv": args.old_csv,
        "new_csv": args.new_csv or "",
        "mode": args.mode,
        "mix_ratio": args.mix_ratio,
        "source_rows": len(train_df),
        "train_rows": len(train_df),
        "train_period": _period(train_df),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "artifacts_dir": str(model_root),
        "artifacts": {
            "model": str(final_model_path(out_name)),
            "norm": str(out_norm),
            "params": params_out,
            "logs": str(logs_dir(out_name)),
            "train_meta": str(meta_path),
        },
        "hyperparameters": {
            "steps": args.steps,
            "window": contract["window"],
            "ep_len": contract["ep_len"],
            "algo": "ppo",
            "reward_mode": contract["reward_mode"],
            "reward_profile": (
                contract["base_meta"].get("hyperparameters", {}).get("reward_profile")
                or "balanced"
            ),
            "reward_profile_config": contract["reward_profile_config"],
            "reward_formula": contract["reward_formula"],
            "action_profile": contract["action_key"],
            "action_profile_label": contract["action_profile"]["label"],
            "action_profile_json": contract["base_meta"].get("hyperparameters", {}).get("action_profile_json"),
            "action_profile_params": contract["action_profile"]["params"],
            "action_profile_config": contract["action_profile"],
            "max_hold": contract["max_hold"],
            "learning_rate": args.lr,
            "fine_tune_mode": args.mode,
            "mix_ratio": args.mix_ratio,
        },
    }
    _write_json(meta_path, meta)
    print(f"[meta] -> {meta_path}")

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        boundaries = np.cumsum([0] + segment_lengths)
        normalized_segments = [train_df.iloc[a:b].reset_index(drop=True)
                               for a, b in zip(boundaries[:-1], boundaries[1:])]
        return Monitor(SegmentedTradingEnv(
            normalized_segments,
            feature_cols,
            window_size=contract["window"],
            max_steps=contract["ep_len"],
            reward_mode=contract["reward_mode"],
            reward_profile=contract["reward_profile"],
            reward_overrides=contract["reward_overrides"],
            reward_formula=contract["reward_formula"],
            action_profile=contract["action_profile"],
            max_hold_bars=contract["max_hold"],
        ))

    train_env = DummyVecEnv([make_env])

    print(f"\n[load] base model: {contract['base_path']}")
    model = PPO.load(str(contract["base_path"]), env=train_env)
    base_timesteps = model.num_timesteps
    model.tensorboard_log = str(logs_dir(out_name))

    print(f"[adjust] learning_rate -> {args.lr}")
    model.learning_rate = args.lr
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:
        group["lr"] = args.lr

    print(f"\n[finetune] {args.steps:,} steps ...")
    model.learn(
        total_timesteps=args.steps,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    save_path = final_model_path(out_name)
    model.save(str(save_path))
    print(f"\n[save] {save_path}")
    print(f"[save] {out_norm}")

    stats = None
    if new_df is not None:
        print("\n[eval] IN-SAMPLE diagnostic on new training data (not unseen Test) ...")
        eval_df = new_df.copy()
        for c in feature_cols:
            if c in norm.index:
                eval_df[c] = (eval_df[c] - norm.at[c, "mean"]) / norm.at[c, "std"]
        eval_df = eval_df.fillna(0).reset_index(drop=True)
        eval_env = TradingEnv(
            eval_df,
            feature_cols,
            window_size=contract["window"],
            max_steps=max(1, len(eval_df) - contract["window"] - 2),
            reward_mode=contract["reward_mode"],
            reward_profile=contract["reward_profile"],
            reward_overrides=contract["reward_overrides"],
            reward_formula=contract["reward_formula"],
            action_profile=contract["action_profile"],
            max_hold_bars=contract["max_hold"],
        )
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(int(action))
            done = terminated or truncated
        stats = eval_env.get_stats()
        print("\n" + "=" * 50)
        print("  Quick eval on NEW data")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:<15}: {value:.4f}")
            else:
                print(f"  {key:<15}: {value}")
        print("=" * 50)

    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    meta["quick_eval_stats"] = stats
    meta["actual_timesteps"] = model.num_timesteps - base_timesteps
    meta["segment_training_steps"] = train_env.envs[0].unwrapped.segment_steps
    _write_json(meta_path, meta)
    run.publish(model)
    train_env.close()
    print(f"[meta] updated -> {meta_path}")
    print("\nFine-tuning complete. Try:")
    print(f"  python backtest_live.py {out_name} <csv> --conf 0 --mode pure_agent")


if __name__ == "__main__":
    main()
