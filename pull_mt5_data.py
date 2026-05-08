"""
pull_mt5_data.py — Pull Historical Data from MT5 + Calculate Features
======================================================================
ดึงข้อมูลจาก MT5 → คำนวณ features → label → save CSV
ใช้สำหรับ train + fine-tune ใน production cycle

Usage:
    # Pull 10 years for training
    python pull_mt5_data.py --start 2010-01-01 --end 2020-12-31 --name train_data

    # Pull last 3 months for fine-tune
    python pull_mt5_data.py --days 90 --name latest_data

    # Pull test set
    python pull_mt5_data.py --start 2021-01-01 --end 2026-04-30 --name test_data
"""
import sys, io, argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from mt5_connector import MT5Connector
from features import calc_features, get_feature_columns


def label_data(df: pd.DataFrame, forward_bars: int = 5,
               method: str = "quantile") -> pd.DataFrame:
    """Label rows with UP/DOWN/FLAT based on future_return"""
    # future return
    future_close = df['close'].shift(-forward_bars)
    df['future_return'] = (future_close - df['close']) / df['close']

    if method == "quantile":
        q33 = df['future_return'].quantile(0.33)
        q67 = df['future_return'].quantile(0.67)
        df['target'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, q33, q67, np.inf],
            labels=['DOWN', 'FLAT', 'UP']
        )
    else:
        thr = 0.001
        df['target'] = 'FLAT'
        df.loc[df['future_return'] >= thr, 'target'] = 'UP'
        df.loc[df['future_return'] <= -thr, 'target'] = 'DOWN'

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--start", help="start date YYYY-MM-DD")
    ap.add_argument("--end", help="end date YYYY-MM-DD")
    ap.add_argument("--days", type=int,
                    help="ใช้ N วันล่าสุด (alternative to start/end)")
    ap.add_argument("--name", default="data",
                    help="output filename (without .csv)")
    ap.add_argument("--forward_bars", type=int, default=5)
    ap.add_argument("--label_method", default="quantile",
                    choices=["quantile", "threshold"])
    args = ap.parse_args()

    # Parse dates
    if args.days:
        end = datetime.now()
        start = end - timedelta(days=args.days)
    elif args.start and args.end:
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
    else:
        print("ERROR: ต้องระบุ --days หรือ --start + --end")
        return 1

    print("=" * 60)
    print(f"  PULL MT5 DATA")
    print("=" * 60)
    print(f"  Symbol     : {args.symbol}")
    print(f"  Timeframe  : {args.timeframe}")
    print(f"  Period     : {start:%Y-%m-%d} → {end:%Y-%m-%d}")
    print(f"  Output     : {args.name}.csv")
    print("=" * 60)

    # Connect MT5
    mt5 = MT5Connector(symbol=args.symbol)
    if not mt5.connect():
        print("MT5 connection failed")
        return 1

    # Pull H1 data
    print(f"\n[pull] H1 bars ...")
    df_h1 = mt5.get_rates_range(args.timeframe, start, end)
    if len(df_h1) == 0:
        print("No data returned")
        return 1
    print(f"  H1: {len(df_h1):,} bars")

    # Pull H4 + D1 for multi-TF features
    print(f"[pull] H4 bars ...")
    df_h4 = mt5.get_rates_range('H4', start - timedelta(days=30), end)
    print(f"  H4: {len(df_h4):,} bars")

    print(f"[pull] D1 bars ...")
    df_d1 = mt5.get_rates_range('D1', start - timedelta(days=300), end)
    print(f"  D1: {len(df_d1):,} bars")

    # Calc features
    print(f"\n[features] calculating 35 features ...")
    # set timestamp index for alignment
    df_h4 = df_h4.set_index('timestamp', drop=False)
    df_d1 = df_d1.set_index('timestamp', drop=False)
    df = calc_features(df_h1, df_h4=df_h4, df_d1=df_d1)

    # Drop initial NaN
    feature_cols = get_feature_columns()
    initial = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"  dropped {initial - len(df)} rows (NaN from indicators warmup)")
    print(f"  remaining: {len(df):,} rows")

    # Label
    print(f"\n[label] forward_bars={args.forward_bars}, method={args.label_method}")
    df = label_data(df, args.forward_bars, args.label_method)

    # Class balance
    if 'target' in df.columns:
        print(f"  Class distribution:")
        for cls, count in df['target'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {cls:<6}: {count:>7,} ({pct:5.1f}%)")

    # Save
    df['symbol'] = args.symbol
    out = f"{args.name}.csv"
    # Reorder columns for readability
    meta = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    meta = [c for c in meta if c in df.columns]
    final_cols = meta + feature_cols + ['future_return', 'target']
    final_cols = [c for c in final_cols if c in df.columns]
    df[final_cols].to_csv(out, index=False)

    print(f"\n[save] -> {out}")
    print(f"  size: {Path(out).stat().st_size / 1024:.0f} KB")

    mt5.disconnect()

    print("\n[next steps]")
    print(f"  1. Inspect: head {out}")
    print(f"  2. Relabel (if needed): python relabel.py {out}")
    print(f"  3. Train: python rl_train.py {out} --steps 200000 --name rl_v3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
