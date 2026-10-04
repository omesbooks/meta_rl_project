"""Shared chronological data validation and train-only preprocessing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NON_FEATURES = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
LEAKY = ("future_", "forward_", "next_", "target")


def feature_columns(df):
    return [c for c in df if c not in NON_FEATURES
            and not any(k in c.lower() for k in LEAKY)
            and pd.api.types.is_numeric_dtype(df[c])]


def validate_frame(df, label="dataset", features=None):
    df = df.copy()
    if not {"timestamp", "close"}.issubset(df.columns):
        raise ValueError(f"{label}: timestamp and close are required")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any() or df["timestamp"].duplicated().any():
        raise ValueError(f"{label}: invalid or duplicate timestamps")
    for col in ("open", "high", "low", "close"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if not np.isfinite(df[col]).all() or (df[col] <= 0).any():
                raise ValueError(f"{label}: {col} must contain finite positive prices")
    if {"open", "high", "low", "close"}.issubset(df):
        if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
                (df.low > df[["open", "close", "high"]].min(axis=1))).any():
            raise ValueError(f"{label}: inconsistent OHLC bounds")
    cols = ([c for c in df if c not in NON_FEATURES and not any(k in c.lower() for k in LEAKY)]
            if features is None else list(features))
    if not cols:
        raise ValueError(f"{label}: no numeric features")
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing features: {sorted(missing)}")
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(df[cols].to_numpy()).all():
        raise ValueError(f"{label}: features contain NaN/Inf or nonnumeric values")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_dataset(path, features=None):
    return validate_frame(pd.read_csv(path), str(path), features)


def prune_features(df, features, threshold):
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Correlation threshold must be > 0 and <= 1")
    kept = [c for c in features if df[c].nunique() > 1]
    dropped = [c for c in features if c not in kept]
    reasons = {c: (None, None) for c in dropped}
    corr = df[kept].corr().abs().fillna(0)
    while len(kept) > 1:
        values = corr.loc[kept, kept].to_numpy(copy=True)
        np.fill_diagonal(values, 0)
        i, j = np.unravel_index(values.argmax(), values.shape)
        if values[i, j] <= threshold:
            break
        remove, keep = (i, j) if values[i].mean() > values[j].mean() else (j, i)
        name = kept[remove]
        reasons[name] = (kept[keep], float(values[i, j]))
        dropped.append(name)
        kept.pop(remove)
    return kept, dropped, reasons


def fit_preprocessing(train, features, threshold=1.0):
    kept, dropped, reasons = prune_features(train, features, threshold)
    if not kept:
        raise ValueError("All training features are constant; collect informative features")
    norm = pd.DataFrame({"mean": train[kept].mean(), "std": train[kept].std()})
    norm["std"] = norm["std"].mask(norm["std"] < 1e-6, 1.0)
    if not np.isfinite(norm.to_numpy()).all():
        raise ValueError("Not enough valid training rows for normalization")
    info = dict(features=kept, dropped=dropped, reasons=reasons, threshold=threshold,
                fit_start=train.timestamp.iloc[0].isoformat(),
                fit_end=train.timestamp.iloc[-1].isoformat(), fit_rows=len(train))
    return kept, norm, info


def apply_normalization(df, features, norm):
    out = validate_frame(df, features=features)
    if list(norm.index) != list(features) or not np.isfinite(norm.to_numpy()).all() or (norm["std"] <= 0).any():
        raise ValueError("Normalization feature order or values are invalid")
    out[features] = (out[features] - norm["mean"]) / norm["std"]
    if not np.isfinite(out[features].to_numpy(dtype=np.float32)).all():
        raise ValueError("Normalization produced nonfinite observations")
    return out


def split_training_data(df, train_pct, window, eval_df=None):
    if not np.isfinite(train_pct) or not 0 < train_pct <= 1:
        raise ValueError("Train fraction must be > 0 and <= 1")
    split = int(len(df) * train_pct)
    train = df.iloc[:split].copy()
    minimum = window + 3
    if len(train) < minimum:
        raise ValueError(f"Train needs at least {minimum} rows")
    holdout = df.iloc[split:].copy() if eval_df is None else eval_df.copy()
    if eval_df is not None and (holdout.empty or holdout.timestamp.min() <= train.timestamp.max()):
        raise ValueError("Separate evaluation data must be strictly later than Train")
    if holdout.empty:
        return train, None, None, split
    middle = len(holdout) // 2
    validation, test = holdout.iloc[:middle].copy(), holdout.iloc[middle:].copy()
    if min(len(validation), len(test)) < minimum:
        raise ValueError(f"Holdout needs at least {2 * minimum} rows for separate Validation and Test")
    return train, validation, test, split


def check_selection_scope(path, train):
    sidecar = Path(path).with_suffix(".features.json")
    if sidecar.exists():
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if pd.Timestamp(info["fit_end"]) > train.timestamp.max():
            raise ValueError("Feature selection used data later than this Train cutoff; use raw CSV")
    elif "_clean" in Path(path).stem:
        raise ValueError("Clean CSV has no feature-selection provenance; select the original CSV and clean again")


def walkforward_source(path):
    path = Path(path).resolve()
    sidecar = path.with_suffix(".features.json")
    if sidecar.exists():
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        source = Path(info["source_csv"])
        if not source.is_file() or source == path:
            raise ValueError("Walk-Forward requires the original unselected feature CSV")
        raw = load_dataset(source)
        selected = load_dataset(path)
        raw = raw[raw.timestamp.isin(selected.timestamp)].reset_index(drop=True)
        if len(raw) != len(selected):
            raise ValueError("Original CSV no longer matches the cleaned dataset timestamps")
        return raw, float(info["threshold"])
    if "_clean" in path.stem:
        raise ValueError("Walk-Forward requires raw CSV; old Clean CSV has no selection provenance")
    return load_dataset(path), 1.0


def validate_hyperparameters(args):
    for key in ("steps", "window", "max_hold", "ep_len", "n_epochs"):
        if hasattr(args, key) and (not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
            raise ValueError(f"{key} must be positive")
    for key in ("n_steps", "batch_size"):
        if getattr(args, key, 2) < 2:
            raise ValueError(f"{key} must be >= 2")
    if getattr(args, "batch_size", 2) > getattr(args, "n_steps", 2):
        raise ValueError("batch_size must be <= n_steps")
    for key in ("learning_rate", "clip_range", "gamma", "gae_lambda", "ent_coef", "vf_coef"):
        value = getattr(args, key, None)
        if value is not None and (not np.isfinite(value) or value < 0 or
                                  (key in ("learning_rate", "clip_range") and value == 0) or
                                  (key in ("gamma", "gae_lambda", "clip_range") and value > 1)):
            raise ValueError(f"Invalid {key}")
    for key, allow_zero, allow_one in (("train_pct", False, True), ("mc_skip_frac", True, False)):
        value = getattr(args, key, None)
        if value is not None and (not np.isfinite(value) or value < 0 or value > 1 or
                                  (not allow_zero and value == 0) or (not allow_one and value == 1)):
            raise ValueError(f"Invalid {key}")
    if hasattr(args, "mc_eval") and args.mc_eval < 0:
        raise ValueError("mc_eval must be >= 0")
