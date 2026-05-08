"""
เพิ่ม Lag Features ให้ CSV เพื่อให้ Supervised Model มี "ความจำ"
-----------------------------------------------------------------
รับ CSV ที่มี features ปัจจุบัน → output CSV ที่มี features + lag

เช่น เดิม:
    rsi, ema_fast, atr, ...

หลัง:
    rsi, rsi_lag1, rsi_lag5, rsi_lag10, rsi_lag20,
    ema_fast, ema_fast_lag1, ema_fast_lag5, ...,
    + rolling stats (mean, std)

Usage:
    python add_lag_features.py training_data_v2.csv
    python add_lag_features.py training_data_v2.csv --lags 1,3,5,10,20,50
"""
import sys
import io
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--lags", default="1,3,5,10,20",
                    help="lag periods, comma-separated")
    ap.add_argument("--rolling", default="10,20,50",
                    help="rolling window sizes for mean/std")
    ap.add_argument("--diffs", default="1,5",
                    help="diff periods (rate of change)")
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(",")]
    windows = [int(x) for x in args.rolling.split(",")]
    diffs = [int(x) for x in args.diffs.split(",")]

    df = pd.read_csv(args.csv)
    print(f"[in] {args.csv} — {len(df):,} rows × {len(df.columns)} cols")

    # detect feature columns (exclude meta/label)
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low",
            "close", "volume", "future_return", "target",
            "hour", "dow", "session_london", "session_ny", "session_asia",
            "trend_h4", "trend_d1"}  # binary/time features ไม่ต้อง lag
    feat = [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    print(f"[features] {len(feat)} numeric columns to lag")

    # ----- 1) Lag features -----
    new_cols = {}
    for c in feat:
        for lag in lags:
            new_cols[f"{c}_lag{lag}"] = df[c].shift(lag)
    print(f"[lag] adding {len(feat) * len(lags)} lag columns")

    # ----- 2) Rolling stats -----
    for c in feat:
        for w in windows:
            new_cols[f"{c}_mean{w}"] = df[c].shift(1).rolling(w).mean()
            new_cols[f"{c}_std{w}"]  = df[c].shift(1).rolling(w).std()
    print(f"[rolling] adding {len(feat) * len(windows) * 2} rolling stats")

    # ----- 3) Differences (rate of change) -----
    for c in feat:
        for d in diffs:
            new_cols[f"{c}_diff{d}"] = df[c].diff(d)
    print(f"[diff] adding {len(feat) * len(diffs)} diff columns")

    # combine
    df_new = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # drop initial NaN rows (from lag/rolling)
    max_lookback = max(lags + windows + diffs)
    df_new = df_new.iloc[max_lookback:].reset_index(drop=True)
    df_new = df_new.fillna(0)

    out = Path(args.csv).with_stem(Path(args.csv).stem + "_lagged")
    df_new.to_csv(out, index=False)

    print(f"\n[out] {out}")
    print(f"  rows: {len(df_new):,} (dropped first {max_lookback} for NaN)")
    print(f"  cols: {len(df_new.columns)} (เดิม {len(df.columns)})")
    print(f"\nไปเทรนต่อด้วย:")
    print(f"  python app.py  -> เปิดไฟล์ {out.name}")


if __name__ == "__main__":
    main()
