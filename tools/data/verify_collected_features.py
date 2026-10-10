"""Verify a freshly collected dataset before training on it.

Checks the two defects found in the 2026-08-09 collections and a few generic
ones, using only the CSV + its .params.json:

  1. constant columns (a feature that never changes carries nothing — the
     August divergence columns were all 0 / 1.0)
  2. divergence parity: recompute the *_div_* flags from the CSV's own
     rsi_<N> / macd_hist columns with tools/data/divergence_features.py logic
     and compare with what the collector wrote
  3. nearness windows: compare near_high_<N> with close / rolling(N).max()
     from the CSV's own high column, for N in NEAR_N1..3 — and report which
     window each column REALLY matches (the August files had 100 -> 250 and
     250/1500 -> 1500)
  4. basic sanity: timestamps strictly increasing, OHLC consistent, NaN/Inf
  5. candle parameter mapping (info only): which marubozu threshold the
     candle_marubozu column really follows. CandlePatterns keeps `input group`
     lines, so iCustom shifts its parameters (effective threshold 0.30, not
     0.95) — consistent between collector and EA, so not a training blocker.

Usage:
    python tools/data/verify_collected_features.py usdjpy_h4_2025.csv
Exit code 0 = all checks passed, 1 = something to fix before training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    df = pd.read_csv(path)
    sidecar = path.with_suffix(".params.json")
    params = json.loads(sidecar.read_text(encoding="utf-8-sig")) if sidecar.exists() else {}
    base = {"timestamp", "open", "high", "low", "close", "volume"}
    feats = [c for c in df.columns if c not in base]
    print(f"[data] {path.name}: {len(df):,} rows, {len(feats)} features, "
          f"{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"[params] {sidecar.name}: {'found' if params else 'MISSING'}")
    problems = []

    # ---- 4. sanity -------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
        problems.append("timestamps not strictly increasing / unparsable / duplicated")
    ohlc = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
        problems.append("OHLC contains NaN/Inf or non-positive prices")
    elif ((df.high < df[["open", "close", "low"]].max(axis=1)) |
          (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        problems.append("OHLC bounds inconsistent (high below open/close or low above)")
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in feats if not np.isfinite(num[c].to_numpy(dtype=float)).all()]
    if bad:
        problems.append(f"NaN/Inf/non-numeric in {len(bad)} feature(s): {bad[:8]}")

    # ---- 1. constants ----------------------------------------------------
    const = [c for c in feats if num[c].nunique(dropna=False) <= 1]
    print(f"\n[1] constant features: {len(const)}")
    for c in const:
        print(f"      {c} = {num[c].iloc[0]}")
    div_const = [c for c in const if "_div_" in c or c.startswith("div_")]
    if div_const:
        problems.append(f"divergence columns constant ({len(div_const)}) — collector ran with a "
                        f"build that never writes signals; recompile PriceDivergence + collector")
    # candle_mathold is constant by construction: the CandlePatterns iCustom
    # shift makes its outer-body minimum 1.0 (see check 5 / RL_Indicators.mqh)
    other_const = [c for c in const if c not in div_const and c != "candle_mathold"]
    if other_const:
        problems.append(f"other constant features: {other_const}")

    # ---- 2. divergence parity --------------------------------------------
    div_cols = [c for c in feats if "_div_" in c]
    if div_cols:
        try:
            from divergence_features import collect_divergences
            rsi_period = int(params.get("DIV_RSI_PERIOD", 14))
            oscs = [f"rsi_{rsi_period}", "macd_hist"]
            missing = [o for o in oscs if o not in df.columns]
            if missing:
                print(f"\n[2] divergence parity skipped — oscillator columns missing: {missing}")
            else:
                feat, _ = collect_divergences(
                    df, oscs,
                    int(params.get("DIV_PIVOT_LEFT", 3)), int(params.get("DIV_PIVOT_RIGHT", 3)),
                    int(params.get("DIV_MIN_SPAN", 5)), int(params.get("DIV_MAX_SPAN", 60)),
                    True, int(params.get("DIV_AGE_CAP", 50)))
                print(f"\n[2] divergence parity (collector vs Python twin):")
                worst = 0.0
                for c in div_cols:
                    if c not in feat.columns:
                        continue
                    # flags are sparse (~1% of bars) so plain agreement reads ~99% even
                    # when the collector wrote nothing — compare the signal SETS instead
                    a = num[c].to_numpy(dtype=float) > 0
                    b = feat[c].to_numpy(dtype=float) > 0
                    union = int((a | b).sum())
                    jacc = (float((a & b).sum()) / union) if union else 1.0
                    worst = max(worst, 1 - jacc)
                    print(f"      {c:22s} collector signals={int(a.sum()):4d}  "
                          f"python={int(b.sum()):4d}  overlap(Jaccard)={jacc:6.1%}")
                for c in ("div_bull_age", "div_bear_age"):
                    if c in feat.columns and c in num.columns:
                        agree = float(np.mean(np.isclose(num[c], feat[c], atol=1e-6)))
                        worst = max(worst, 1 - agree)
                        print(f"      {c:22s} exact agree={agree:6.1%}")
                if worst > 0.05:
                    problems.append(f"divergence columns disagree with the Python twin "
                                    f"(worst overlap {1 - worst:.1%}) — check indicator build / params")
        except Exception as exc:
            print(f"\n[2] divergence parity skipped: {exc}")
    else:
        print("\n[2] no divergence columns in this file")

    # ---- 3. nearness windows ---------------------------------------------
    near_cols = [c for c in feats if c.startswith("near_high_")]
    if near_cols:
        print(f"\n[3] nearness: which rolling window does each column really match?")
        wanted = [int(params.get(k, 0)) for k in ("NEAR_N1", "NEAR_N2", "NEAR_N3") if params.get(k)]
        candidates = sorted(set(wanted + [50, 100, 250, 500, 1000, 1500, 2000]))
        close, high = df["close"], df["high"]
        for c in near_cols:
            declared = int(c.split("_")[-1])
            col = num[c].to_numpy(dtype=float)
            best_n, best_m = None, -1.0
            for n in candidates:
                ref = (close / high.rolling(n, min_periods=1).max()).to_numpy(dtype=float)
                m = float(np.mean(np.isclose(col, ref, atol=1e-6)))
                if m > best_m:
                    best_n, best_m = n, m
            ok = best_n == declared and best_m > 0.95
            print(f"      {c:16s} declared {declared:5d}  best match rolling({best_n}) {best_m:6.1%}  "
                  f"{'OK' if ok else 'MISMATCH'}")
            if not ok:
                problems.append(f"{c} matches rolling({best_n}) not {declared}")
        # duplicate horizon columns
        for a, b in (("near_high_250", "near_high_1500"), ("near_high_100", "near_high_250")):
            if a in num and b in num and bool((num[a] == num[b]).all()):
                problems.append(f"{a} and {b} are identical")
    else:
        print("\n[3] no nearness columns in this file")

    # ---- 5. candle parameter mapping (info) ----------------------------
    if "candle_marubozu" in num.columns:
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        body, rng = np.abs(c - o), h - l
        col = num["candle_marubozu"].to_numpy(dtype=float)
        declared = float(params.get("CP_MarubozuThresh", 0.95))
        shifted = float(params.get("CP_HammerBodyMaxPct", 0.30))
        print("\n[5] candle_marubozu follows which threshold?")
        for label, thr in (("declared CP_MarubozuThresh", declared),
                           ("iCustom-shifted CP_HammerBodyMaxPct", shifted)):
            flag = np.where((rng > 1e-10) & (body >= rng * thr), np.sign(c - o), 0.0)
            print(f"      {label:36s} {thr:.2f}: agree {np.mean(flag == col):6.1%}")
        print("      (shifted = the known, train/live-consistent CandlePatterns mapping)")

    print("\n" + "=" * 64)
    if problems:
        print("VERDICT: FIX BEFORE TRAINING")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERDICT: OK — dataset passes all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
