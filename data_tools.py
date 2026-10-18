"""Data Tools page work, shared by rl_app.py and the web GUI.

Moved verbatim out of RLTradingStudio (_import_from_collector_worker, _split_csv_worker,
_show_correlation_worker, _clean_features_worker, _smart_read_csv, the params-sidecar
helpers and the export-command setup of _export_onnx / _export_onnx_worker). Every
function reports through two callbacks so each front end decides how to show it:

    log(text, tag)                                    tag: info | success | warn | error | metric
    notice(title, message, paths=None, level="success", popup=True)

Nothing here touches a widget; the callers marshal to their UI thread themselves.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _noop_notice(title, message, paths=None, level="success", popup=True):
    pass


# ------------------------------------------------------------------ files
def common_files_dir():
    """MT5 Common\\Files folder (where DataCollector_RL writes)."""
    return Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal" / "Common" / "Files"


def collector_sources(d=None):
    """CSV files in Common\\Files for the import dropdown (_m1.csv companions excluded:
    they are imported automatically alongside their dataset)."""
    d = Path(d) if d else common_files_dir()
    try:
        if d.exists():
            return sorted([p.name for p in d.glob("*.csv") if not p.name.endswith("_m1.csv")])
    except Exception:
        pass
    return []


def collector_out_default(src_name):
    return f"training_data_{Path(src_name).stem}.csv"


def smart_read_csv(path, log, notice=_noop_notice, **kwargs):
    """Read CSV with auto encoding detection.
    Handles raw MT4/MT5 UTF-16 LE files transparently.
    If file is UTF-16, converts in-place and saves as UTF-8."""
    import pandas as pd
    path = Path(path)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        pass
    with open(path, 'rb') as f:
        magic = f.read(2)
    if magic == b'\xff\xfe':
        encoding = 'utf-16-le'
    elif magic == b'\xfe\xff':
        encoding = 'utf-16-be'
    else:
        encoding = 'cp1252'
    log(f"⚠️  {path.name} is {encoding} — converting to UTF-8...", "warn")
    text = path.read_bytes().decode(encoding, errors='replace').lstrip('﻿')
    backup = path.with_suffix(path.suffix + '.utf16_backup')
    if not backup.exists():
        path.rename(backup)
        log(f"  Original backed up as: {backup.name}", "info")
    path.write_text(text, encoding='utf-8')
    log(f"  ✓ Converted to UTF-8 ({path.stat().st_size / 1024 / 1024:.1f} MB)", "success")
    notice("Modified CSV encoding", "Converted the source CSV to UTF-8 and kept the original backup.",
           [path, backup], popup=False)
    return pd.read_csv(path, **kwargs)


def find_params_sidecar(csv_path):
    csv_path = Path(csv_path)
    for candidate in (csv_path.with_suffix(".params.json"), csv_path.parent / (csv_path.stem + ".params.json")):
        if candidate.exists():
            return candidate
    return None


def copy_params_sidecar(src_csv, dst_csv):
    src_params = find_params_sidecar(src_csv)
    if not src_params:
        return None, None
    dst_params = Path(dst_csv).with_suffix(".params.json")
    try:
        shutil.copy(src_params, dst_params)
        return dst_params, None
    except Exception as e:
        return None, e


def get_feature_columns(df):
    """Get numeric feature columns (skip OHLCV + leaky)"""
    import pandas as pd
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    leaky_kw = ("future_", "forward_", "next_", "target")
    return [c for c in df.columns
            if c not in skip and not any(k in c.lower() for k in leaky_kw) and pd.api.types.is_numeric_dtype(df[c])]


def is_prune_too_aggressive(total_features, kept_features):
    """Correlation pruning can delete entire indicator period ladders when the threshold
    is too low. Block writes when the result keeps too little."""
    if total_features <= 0:
        return False, 0, 1.0
    min_keep = min(total_features, max(5, math.ceil(total_features * 0.35)))
    keep_ratio = kept_features / total_features
    return kept_features < min_keep, min_keep, keep_ratio


# ------------------------------------------------------------------ ① import from DataCollector_RL
def import_collector(src_csv, src_params, out_name, want_m1, work_dir, log, notice=_noop_notice):
    """Copy CSV (+ params.json, + <base>_m1.csv) from Common\\Files into the project."""
    work_dir = Path(work_dir)
    dst_csv = work_dir / Path(out_name).with_suffix(".csv").name
    raw_copy = dst_csv
    csv_existed = raw_copy.exists()
    shutil.copy(src_csv, raw_copy)
    log(f"[copy] CSV  -> {raw_copy.name}", "success")
    copied_paths = [raw_copy]
    src_m1 = Path(src_csv).parent / f"{Path(src_csv).stem}_m1.csv"
    if want_m1 and src_m1.exists():
        dst_m1 = work_dir / f"{raw_copy.stem}_m1.csv"
        shutil.copy(src_m1, dst_m1)
        copied_paths.append(dst_m1)
        log(f"[copy] M1   -> {dst_m1.name} (ใช้ในช่อง M1 Data ของหน้า Backtest)", "success")
    if Path(src_params).exists():
        dst_params = raw_copy.with_suffix(".params.json")
        params_existed = dst_params.exists()
        shutil.copy(src_params, dst_params)
        copied_paths.append(dst_params)
        log(f"[copy] params -> {dst_params.name}", "success")
    else:
        params_existed = False
        log("[warn] no params.json sidecar — EA will use RL_Indicators defaults", "warn")
    log("[done] collector files imported", "success")
    action = "Updated existing import" if (csv_existed or params_existed) else "Created import files"
    notice(action, "Collector CSV/params copied into the project folder.", copied_paths)
    return copied_paths


# ------------------------------------------------------------------ ② split train/test by date
def split_csv(settings, work_dir, log, notice=_noop_notice, read_csv=None, prune=None):
    """settings: csv, split_date, train_suffix, test_suffix, clean_after, clean_threshold.
    Returns the written paths ([] when nothing was written)."""
    import pandas as pd
    from training_data import prune_features
    work_dir = Path(work_dir)
    prune = prune or prune_features
    read_csv = read_csv or (lambda p: smart_read_csv(p, log, notice))
    csv = settings["csv"]
    if csv in ("(none)", ""):
        log("Please select a CSV", "error")
        return []
    split_date = settings["split_date"]
    if not split_date:
        log("Please enter split date", "error")
        return []
    train_suffix = settings["train_suffix"]
    test_suffix = settings["test_suffix"]

    log(f"Loading {csv}...", "info")
    df = read_csv(work_dir / csv)
    if 'timestamp' not in df.columns:
        log("No 'timestamp' column found", "error")
        return []
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if df.empty:
        log("No valid timestamp rows after parsing", "error")
        return []
    split_dt = pd.to_datetime(split_date)
    train_df = df[df['timestamp'] < split_dt]
    test_df = df[df['timestamp'] >= split_dt]
    if train_df.empty or test_df.empty:
        first_date = df['timestamp'].min().date()
        last_date = df['timestamp'].max().date()
        log("Split aborted: train/test would be empty.", "error")
        log(f"  Data range : {first_date} -> {last_date}", "metric")
        log(f"  Train rows : {len(train_df):,}", "error" if train_df.empty else "metric")
        log(f"  Test rows  : {len(test_df):,}", "error" if test_df.empty else "metric")
        log("Choose a split date inside the data range.", "warn")
        return []

    log(f"Total: {len(df):,} rows | Split at {split_date}", "info")
    log(f"  Train: {len(train_df):,} rows ({df['timestamp'].min().date()} → {(split_dt - pd.Timedelta(days=1)).date()})", "metric")
    log(f"  Test : {len(test_df):,} rows ({split_dt.date()} → {df['timestamp'].max().date()})", "metric")

    base = Path(csv).stem
    train_path = work_dir / f"{base}{train_suffix}.csv"
    test_path = work_dir / f"{base}{test_suffix}.csv"
    train_existed = train_path.exists()
    test_existed = test_path.exists()
    auto_cleaned = False
    auto_clean_dropped = 0
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    copied_sidecars = []
    for output_path in (train_path, test_path):
        copied, err = copy_params_sidecar(work_dir / csv, output_path)
        if copied:
            copied_sidecars.append(copied)
            log(f"+ forwarded {copied.name}", "success")
        elif err:
            log(f"[warn] could not forward params for {output_path.name}: {err}", "warn")
        else:
            log(f"[warn] no params.json sidecar found for {csv}", "warn")
    log(f"✓ Saved: {train_path.name}", "success")
    log(f"✓ Saved: {test_path.name}", "success")

    if settings["clean_after"]:
        try:
            threshold = float(settings["clean_threshold"])
        except ValueError:
            threshold = 0.99
        log(f"\n=== Auto-cleaning redundant features (corr > {threshold}) ===", "info")
        train_features = get_feature_columns(train_df)
        kept, dropped, reasons = prune(train_df, train_features, threshold)
        if dropped:
            blocked, min_keep, keep_ratio = is_prune_too_aggressive(len(train_features), len(kept))
            if blocked:
                log(f"  ⚠ Auto-clean skipped: would keep only {len(kept)}/{len(train_features)} "
                    f"features ({keep_ratio:.0%}); minimum safe keep is {min_keep}. "
                    "Raise threshold closer to 0.99.", "error")
                notice("Auto-clean skipped",
                       "Redundant-feature cleanup was too aggressive, so split CSV files were left unchanged.",
                       level="warn", popup=False)
                dropped = []
            else:
                log(f"  Dropped {len(dropped)} features  (kept {len(kept)})", "warn")
                for col in dropped[:10]:
                    kept_col, r = reasons[col]
                    if kept_col is None:
                        log(f"    - {col:25s}  (constant / no variance)", "warn")
                    else:
                        log(f"    - {col:25s}  (r={r:.3f} with {kept_col})", "warn")
                if len(dropped) > 10:
                    log(f"    ... +{len(dropped) - 10} more", "warn")
                non_feat_train = [c for c in train_df.columns if c not in train_features]
                out_cols_train = non_feat_train + kept
                non_feat_test = [c for c in test_df.columns if c not in train_features]
                kept_in_test = [c for c in kept if c in test_df.columns]
                out_cols_test = non_feat_test + kept_in_test
                train_df[out_cols_train].to_csv(train_path, index=False)
                test_df[out_cols_test].to_csv(test_path, index=False)
                auto_cleaned = True
                auto_clean_dropped = len(dropped)
                log(f"  ✓ Re-saved: {train_path.name} ({len(out_cols_train)} cols)", "success")
                log(f"  ✓ Re-saved: {test_path.name} ({len(out_cols_test)} cols)", "success")
        else:
            log("  No redundant features found — files unchanged", "info")

    title = "Updated split files" if (train_existed or test_existed) else "Created split files"
    detail = f"Train/test split saved: {len(train_df):,} train rows, {len(test_df):,} test rows."
    if auto_cleaned:
        detail += f" Auto-clean re-saved both files after dropping {auto_clean_dropped} features."
    notice(title, detail, [train_path, test_path, *copied_sidecars])
    return [train_path, test_path, *copied_sidecars]


# ------------------------------------------------------------------ ③ feature analysis
def show_correlation(csv, threshold_raw, train_pct, work_dir, log, notice=_noop_notice, read_csv=None, open_file=None):
    """Correlation pairs on TRAIN rows + heatmap PNG; returns the PNG path (None on early exit)."""
    import numpy as np
    from training_data import feature_columns, validate_frame
    from ui_values import number as input_number
    work_dir = Path(work_dir)
    read_csv = read_csv or (lambda p: smart_read_csv(p, log, notice))
    if csv in ("(none)", ""):
        log("Please select a CSV", "error")
        return None
    threshold = input_number(threshold_raw, "Correlation threshold", 0, 1)
    if threshold == 0:
        raise ValueError("Correlation threshold must be > 0")
    log(f"Loading {csv}...", "info")
    df = validate_frame(read_csv(work_dir / csv), csv)
    if not 0 < train_pct <= 1:
        raise ValueError("Train fraction must be > 0 and <= 1")
    df = df.iloc[:int(len(df) * train_pct)]
    if len(df) < 3:
        raise ValueError("Not enough Train rows for correlation")
    features = feature_columns(df)
    log(f"TRAIN ONLY: {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]} ({len(df):,} rows)", "info")
    log(f"Found {len(features)} numeric features (after dropping OHLCV + leaky)", "info")
    if len(features) < 2:
        log("Not enough features to analyze", "error")
        return None
    constant_features = [c for c in features if df[c].nunique(dropna=True) <= 1]
    if constant_features:
        log(f"Found {len(constant_features)} constant/no-variance feature(s); "
            "their correlations are shown as 0.", "warn")
    log("Computing correlation matrix...", "info")
    corr = df[features].corr().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pairs = []
    for i, c1 in enumerate(features):
        for j, c2 in enumerate(features):
            if i >= j:
                continue
            r = corr.loc[c1, c2]
            if r > threshold:
                pairs.append((r, c1, c2))
    pairs.sort(reverse=True)
    log(f"\n[Top correlated pairs] (threshold > {threshold}):", "metric")
    log(f"  Found {len(pairs)} pairs above threshold", "metric")
    for r, c1, c2 in pairs[:30]:
        log(f"  r={r:.3f}  {c1:25s} <-> {c2}", "info")
    if len(pairs) > 30:
        log(f"  ... +{len(pairs) - 30} more", "info")

    log("\nGenerating heatmap PNG...", "info")
    # Figure objects (not pyplot): same image, safe from any thread and any GUI backend
    from matplotlib.figure import Figure
    n = len(features)
    fig_size = max(10, min(24, n * 0.3))
    fig = Figure(figsize=(fig_size, fig_size))
    ax = fig.subplots()
    im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(features, rotation=90, fontsize=8)
    ax.set_yticklabels(features, fontsize=8)
    for i in range(n):
        for j in range(n):
            if i != j and corr.values[i, j] > threshold:
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center',
                        color='white' if corr.values[i, j] > 0.6 else 'black', fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Correlation Matrix — {csv} ({n} features, threshold={threshold})", fontsize=14)
    fig.tight_layout()
    png_path = work_dir / f"{Path(csv).stem}_correlation.png"
    png_existed = png_path.exists()
    fig.savefig(png_path, dpi=110, bbox_inches='tight')
    log(f"✓ Saved: {png_path.name}", "success")
    title = "Updated correlation image" if png_existed else "Created correlation image"
    notice(title, f"Correlation heatmap saved. Found {len(pairs)} pairs above threshold {threshold}.", [png_path])
    if open_file is not None:
        try:
            open_file(png_path)
            log("Opened in default image viewer", "info")
        except Exception:
            log(f"Open manually: {png_path}", "info")
    return png_path


def clean_features(csv, threshold, train_pct, work_dir, log, notice=_noop_notice, read_csv=None, prune=None):
    """Drop redundant features (fit on TRAIN rows) -> <stem>_clean.csv + .features.json; returns paths."""
    from training_data import feature_columns, prune_features, validate_frame
    work_dir = Path(work_dir)
    prune = prune or prune_features
    read_csv = read_csv or (lambda p: smart_read_csv(p, log, notice))
    log(f"\n=== Cleaning redundant features (corr > {threshold}) ===", "info")
    log(f"Loading {csv}...", "info")
    df = validate_frame(read_csv(work_dir / csv), csv)
    if not 0 < train_pct <= 1:
        raise ValueError("Train fraction must be > 0 and <= 1")
    fit_df = df.iloc[:int(len(df) * train_pct)]
    if len(fit_df) < 3:
        raise ValueError("Not enough Train rows for feature selection")
    features = feature_columns(df)
    log(f"Original: {len(features)} features; fit TRAIN ONLY through {fit_df.timestamp.iloc[-1]}", "info")
    kept, dropped, reasons = prune(fit_df, features, threshold)
    if dropped:
        blocked, min_keep, keep_ratio = is_prune_too_aggressive(len(features), len(kept))
        if blocked:
            log(f"⚠ Clean skipped: would keep only {len(kept)}/{len(features)} "
                f"features ({keep_ratio:.0%}); minimum safe keep is {min_keep}.", "error")
            log("Raise threshold closer to 0.99, then run Show Correlation Matrix first.", "warn")
            notice("Clean skipped", "Redundant-feature cleanup was too aggressive, so no cleaned CSV was written.",
                   level="warn")
            return []
    if not dropped:
        log("✓ No redundant features found — already clean!", "success")
        notice("No file changed", "No redundant features were found, so no cleaned CSV was created.",
               level="warn", popup=False)
        return []
    log(f"\nDropped {len(dropped)} features:", "warn")
    for col in dropped:
        kept_col, r = reasons[col]
        if kept_col is None:
            log(f"  - {col:25s}  (constant / no variance)", "warn")
        else:
            log(f"  - {col:25s}  (r={r:.3f} with {kept_col})", "warn")
    log(f"\nKept {len(kept)} features:", "metric")
    for col in kept:
        log(f"  + {col}", "info")
    non_feature = [c for c in df.columns if c not in features]
    out_cols = non_feature + kept
    cleaned_df = df[out_cols]
    out_name = f"{Path(csv).stem}_clean.csv"
    out_path = work_dir / out_name
    out_existed = out_path.exists()
    cleaned_df.to_csv(out_path, index=False)
    out_path.with_suffix(".features.json").write_text(json.dumps({
        "source_csv": str((work_dir / csv).resolve()),
        "fit_start": fit_df.timestamp.iloc[0].isoformat(),
        "fit_end": fit_df.timestamp.iloc[-1].isoformat(),
        "fit_rows": len(fit_df), "threshold": threshold,
        "features": kept, "dropped": dropped,
    }, indent=2), encoding="utf-8")
    notice_paths = [out_path]
    params_path, params_error = copy_params_sidecar(work_dir / csv, out_path)
    if params_path:
        notice_paths.append(params_path)
        log(f"  ✓ Copied params sidecar: {params_path.name}", "success")
    elif params_error:
        log(f"  ⚠ params sidecar copy failed: {params_error}", "warn")
    else:
        log("  ⚠ no params sidecar found for source CSV", "warn")
    size_mb = out_path.stat().st_size / 1024 / 1024
    log(f"\n✓ Saved: {out_name} ({size_mb:.1f} MB · {len(out_cols)} cols)", "success")
    log(f"  Reduction: {len(features)} → {len(kept)} features ({(len(kept) / len(features)) * 100:.0f}% kept)", "metric")
    title = "Updated cleaned CSV" if out_existed else "Created cleaned CSV"
    notice(title, f"Cleaned dataset saved after dropping {len(dropped)} redundant features.", notice_paths)
    return notice_paths


# ------------------------------------------------------------------ ④ export to MT5 (ONNX)
EXPORT_SOURCES = ["Final .zip (matches Backtest)", "Best checkpoint (_best/best_model.zip)", "Auto (final, then best)"]


def clean_deploy_name(deploy_name):
    """ASCII MQL identifier: non [A-Za-z0-9_] -> _, a leading digit gets an m_ prefix ('' = invalid)."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", deploy_name).strip("_")
    if name and name[0].isdigit():
        name = "m_" + name
    return name


def export_plan(model_name, deploy_name, outdir, source_text, work_dir, python=None):
    """-> {cmd, source_key, outdir, display_outdir, expected_paths, existed} for export_to_onnx.py."""
    python = python or sys.executable
    work_dir = Path(work_dir)
    if source_text.startswith("Best"):
        source_key = "best"
    elif source_text.startswith("Auto"):
        source_key = "auto"
    else:
        source_key = "final"
    legacy_outdir = str(work_dir / "mt5_files" / "MQL5")
    use_auto_package = not outdir
    if outdir:
        try:
            use_auto_package = Path(outdir).resolve() == Path(legacy_outdir).resolve()
        except Exception:
            use_auto_package = outdir.rstrip("\\/") == legacy_outdir.rstrip("\\/")
    if use_auto_package:
        outdir = ""
    display_outdir = Path(outdir) if outdir else work_dir / "mt5_files" / "packages" / deploy_name / "MQL5"
    expected = [
        display_outdir / "Files" / f"{deploy_name}.onnx",
        display_outdir / "Include" / f"{deploy_name}_config.mqh",
        display_outdir / "Include" / "RL_Indicators.mqh",
        display_outdir / "Indicators" / "CandlePatterns.mq5",
        display_outdir / "Experts" / f"{deploy_name}_EA.mq5",
    ]
    cmd = [python, "export_to_onnx.py", model_name, "--name", deploy_name, "--source", source_key]
    if outdir:
        cmd += ["--output_dir", outdir]
    return {"cmd": cmd, "source_key": source_key, "outdir": outdir, "display_outdir": display_outdir,
            "expected_paths": expected, "existed": any(p.exists() for p in expected)}


def export_line_tag(line):
    if "✅" in line or "[ok]" in line.lower():
        return "success"
    if "❌" in line or "error" in line.lower():
        return "error"
    if "⚠️" in line or "warn" in line.lower():
        return "warn"
    if "[" in line and "]" in line:
        return "metric"
    return "info"


def export_done_lines(deploy_name, out_path):
    return [(f"\n✅ Export complete! Files in {out_path}", "success"),
            (f"  Files/{deploy_name}.onnx", "metric"), (f"  Include/{deploy_name}_config.mqh", "metric"),
            ("  Include/RL_Indicators.mqh", "metric"), ("  Indicators/CandlePatterns.mq5", "metric"),
            (f"  Experts/{deploy_name}_EA.mq5", "metric"),
            (f"\n📋 Next: copy {out_path}/* to MT5's MQL5/ folder", "info")]
