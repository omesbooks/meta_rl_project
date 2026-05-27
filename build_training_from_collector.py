"""
build_training_from_collector.py
=================================
Turn the DataCollector_RL.mq5 output (features computed by the SAME code the EA
uses) into a training CSV: add future_return + target, keep the SAME feature set
the model used, drop indicator-warmup rows.

Result trains a model whose features MATCH live MT5 inference -> no parity gap.

Usage:
    python build_training_from_collector.py
    python build_training_from_collector.py --future 5 --mode quantile
    python build_training_from_collector.py --in <path> --out training_data_gbpusd_rl.csv
"""
import sys, io, argparse, os
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_IN = (
    r"C:\Users\omesb\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
    r"\rl_gbpusd_dataset.csv"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN,
                    help="collector CSV (Common\\Files\\rl_gbpusd_dataset.csv)")
    ap.add_argument("--out", default="training_data_gbpusd_rl.csv")
    ap.add_argument("--future", type=int, default=5, help="bars ahead for future_return")
    ap.add_argument("--mode", default="quantile", choices=["quantile", "fixed"])
    ap.add_argument("--threshold", type=float, default=0.001, help="for --mode fixed")
    ap.add_argument("--warmup", type=int, default=250,
                    help="drop first N rows (indicator warmup)")
    ap.add_argument("--features", default="rl_gbpusd_norm.csv",
                    help="file whose index = feature set to keep; 'all' = keep all")
    args = ap.parse_args()

    if not Path(args.inp).exists():
        print(f"ERROR: collector CSV not found: {args.inp}")
        print("  Run DataCollector_RL in the Strategy Tester first.")
        return 1

    df = pd.read_csv(args.inp)
    print(f"[in]   {args.inp}  ({len(df):,} rows, {df.shape[1]} cols)")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- future_return + target ----
    fr = (df["close"].shift(-args.future) - df["close"]) / df["close"]
    df["future_return"] = fr
    if args.mode == "quantile":
        q1, q2 = fr.quantile([1/3, 2/3])
        df["target"] = "FLAT"
        df.loc[fr >= q2, "target"] = "UP"
        df.loc[fr <= q1, "target"] = "DOWN"
        print(f"[label] quantile  DOWN<= {q1:.6f}  UP>= {q2:.6f}")
    else:
        t = args.threshold
        df["target"] = "FLAT"
        df.loc[fr >= t, "target"] = "UP"
        df.loc[fr <= -t, "target"] = "DOWN"
        print(f"[label] fixed  +/- {t}")

    # ---- feature selection ----
    if args.features != "all" and Path(args.features).exists():
        keep_feats = list(pd.read_csv(args.features, index_col=0).index)
        missing = [c for c in keep_feats if c not in df.columns]
        if missing:
            print(f"[warn] {len(missing)} requested features missing in collector: {missing[:6]}...")
            keep_feats = [c for c in keep_feats if c in df.columns]
        print(f"[feat] keeping {len(keep_feats)} features (match old model)")
    else:
        ohlcv = {"timestamp", "open", "high", "low", "close", "volume"}
        keep_feats = [c for c in df.columns
                      if c not in ohlcv and c not in ("future_return", "target")]
        print(f"[feat] keeping ALL {len(keep_feats)} features")

    cols = ["timestamp", "open", "high", "low", "close", "volume"] + keep_feats + \
           ["future_return", "target"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    # ---- drop warmup + tail NaN future ----
    before = len(out)
    out = out.iloc[args.warmup:].copy()
    out = out.dropna(subset=["future_return"]).reset_index(drop=True)
    print(f"[trim] dropped {before - len(out):,} rows (warmup {args.warmup} + tail NaN)")

    out.to_csv(args.out, index=False)
    print(f"\n[out]  {args.out}  ({len(out):,} rows, {len(keep_feats)} features)")
    print(f"  date range: {out['timestamp'].min()} -> {out['timestamp'].max()}")
    print(f"  target dist: {dict(out['target'].value_counts())}")
    print(f"\nNext:")
    print(f"  python rl_train.py {args.out} --window 30 --name rl_gbpusd_v2 \\")
    print(f"      --steps 600000 --train_pct 0.85")
    print(f"  python export_to_onnx.py rl_gbpusd_v2 --name rl_gbpusd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
