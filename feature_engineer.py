"""
Feature Engineering Tool — Phase A
=====================================
เพิ่ม features ที่ standard indicators ไม่มี:

1. Multi-Timeframe (MTF):
   - Aggregate ขึ้น TF สูงกว่า → compute indicators → broadcast back
   - H4 → D1 features
   - H1 → H4 features

2. Volatility Regime:
   - ATR percentile / z-score (relative volatility)
   - ADX trend strength flag
   - BB width squeeze
   - Volatility state (low/medium/high)

3. Range vs Trend:
   - Range position (0-1 where price is in recent range)
   - Trend strength (using ADX + EMA alignment)

Usage:
    python feature_engineer.py training_data_v3_h4.csv

    # Custom MTF
    python feature_engineer.py training_data_v3_h4.csv --target_tf D1

Output: <input>_enriched.csv
"""
import sys, io, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def detect_timeframe(df: pd.DataFrame) -> str:
    """Detect timeframe from timestamp diffs"""
    if 'timestamp' not in df.columns:
        return 'unknown'
    ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
    if len(ts) < 2:
        return 'unknown'
    # Mode of first 100 diffs
    diffs = ts.diff().dropna().head(100)
    mode_diff = diffs.mode().iloc[0] if len(diffs) > 0 else None
    if mode_diff is None:
        return 'unknown'
    seconds = mode_diff.total_seconds()
    if seconds <= 60: return 'M1'
    if seconds <= 5*60: return 'M5'
    if seconds <= 15*60: return 'M15'
    if seconds <= 30*60: return 'M30'
    if seconds <= 60*60: return 'H1'
    if seconds <= 4*60*60: return 'H4'
    if seconds <= 24*60*60: return 'D1'
    return 'unknown'


def aggregate_to_higher_tf(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """Aggregate OHLC bars to higher timeframe.
    Returns dataframe with timestamp + OHLCV at target_tf."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    rule_map = {
        'M5': '5min', 'M15': '15min', 'M30': '30min',
        'H1': '1H', 'H4': '4H', 'D1': '1D', 'W1': '1W',
    }
    rule = rule_map.get(target_tf)
    if rule is None:
        raise ValueError(f"Unsupported target TF: {target_tf}")

    agg = df.resample(rule).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna(subset=['open']).reset_index()

    return agg


def compute_basic_indicators(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """Compute essential indicators on a dataframe (assumes OHLCV)"""
    df = df.copy()
    p = prefix

    # RSI(14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df[f'{p}rsi'] = 100 - 100 / (1 + rs)

    # EMA fast & slow
    df[f'{p}ema_fast'] = df['close'].ewm(span=10, adjust=False).mean()
    df[f'{p}ema_slow'] = df['close'].ewm(span=30, adjust=False).mean()

    # Trend direction (1 if ema_fast > ema_slow, else -1)
    df[f'{p}trend_dir'] = np.where(df[f'{p}ema_fast'] > df[f'{p}ema_slow'], 1, -1)

    # ATR(14)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f'{p}atr'] = tr.rolling(14).mean()

    # ATR percentile (relative volatility) over last 100 bars
    df[f'{p}atr_pct'] = df[f'{p}atr'].rolling(100).rank(pct=True)

    # ADX(14) — simplified
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_smooth + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df[f'{p}adx'] = dx.rolling(14).mean()

    return df


def add_multi_tf_features(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """Add features from a higher timeframe by:
    1. Aggregating OHLC to target_tf
    2. Computing indicators on aggregated data
    3. Forward-filling back to source_tf rows (broadcast)"""
    print(f"  [MTF] Aggregating {source_tf} -> {target_tf}...")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Aggregate
    agg = aggregate_to_higher_tf(df, source_tf, target_tf)
    print(f"    {len(df):,} {source_tf} bars -> {len(agg):,} {target_tf} bars")

    # Compute indicators on aggregated
    prefix = f'{target_tf.lower()}_'
    agg = compute_basic_indicators(agg, prefix=prefix)

    # Select only the new feature columns to merge back
    new_features = [c for c in agg.columns
                    if c.startswith(prefix) and c not in ('open', 'high', 'low', 'close', 'volume')]
    print(f"    New features: {new_features}")

    # Set index for merge_asof (forward-fill back to source bars)
    agg_minimal = agg[['timestamp'] + new_features].sort_values('timestamp')
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # merge_asof: each source bar gets the most recent target_tf bar's features
    # (simulates "what info was available at this source bar")
    merged = pd.merge_asof(
        df_sorted, agg_minimal,
        on='timestamp', direction='backward', allow_exact_matches=True
    )

    # Add alignment feature (does H4 trend agree with current trend?)
    if 'ema_20' in df.columns and 'ema_50' in df.columns:
        df_trend = np.where(df_sorted['ema_20'] > df_sorted['ema_50'], 1, -1)
        merged[f'{prefix}alignment'] = (merged[f'{prefix}trend_dir'] == df_trend).astype(int)

    return merged


def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility regime features"""
    print("  [VolRegime] Adding volatility regime features...")
    df = df.copy()

    # Use existing atr_mean if available, else compute
    if 'atr_mean' in df.columns:
        atr = df['atr_mean']
    elif 'atr' in df.columns:
        atr = df['atr']
    else:
        # Compute basic ATR
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['atr_basic'] = atr

    # ATR percentile over last 100 bars (range volatility)
    df['atr_percentile_100'] = atr.rolling(100).rank(pct=True)

    # ATR z-score
    atr_mean_100 = atr.rolling(100).mean()
    atr_std_100 = atr.rolling(100).std()
    df['atr_zscore_100'] = (atr - atr_mean_100) / (atr_std_100 + 1e-9)

    # Volatility state (categorical)
    df['vol_state'] = pd.cut(
        df['atr_percentile_100'],
        bins=[-0.01, 0.33, 0.67, 1.01],
        labels=[0, 1, 2]  # low, medium, high
    ).astype(float)

    # ADX-based trend strength (if ADX exists)
    adx_col = None
    for cand in ('adx_mean', 'adx', 'adx_h4'):
        if cand in df.columns:
            adx_col = cand
            break
    if adx_col:
        df['adx_trending'] = (df[adx_col] > 25).astype(int)
        df['adx_strong'] = (df[adx_col] > 40).astype(int)

    return df


def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add range/breakout features"""
    print("  [Range] Adding range/breakout features...")
    df = df.copy()

    # Range position: where is current close in last 50-bar range?
    high_50 = df['high'].rolling(50).max()
    low_50 = df['low'].rolling(50).min()
    df['range_pos_50'] = (df['close'] - low_50) / (high_50 - low_50 + 1e-9)

    # Range position: 20-bar
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    df['range_pos_20'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-9)

    # Distance from EMA (% terms) — useful for mean reversion
    if 'ema_50' in df.columns:
        df['dist_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']
    if 'ema_200' in df.columns:
        df['dist_ema200'] = (df['close'] - df['ema_200']) / df['ema_200']

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="input CSV file")
    ap.add_argument("--target_tf", default=None,
                    help="Higher TF for MTF features (auto: H1->H4, H4->D1)")
    ap.add_argument("--no_mtf", action="store_true",
                    help="Skip multi-timeframe features")
    ap.add_argument("--no_volreg", action="store_true",
                    help="Skip volatility regime features")
    ap.add_argument("--no_range", action="store_true",
                    help="Skip range features")
    ap.add_argument("--output", help="output CSV (default: <input>_enriched.csv)")
    args = ap.parse_args()

    print("=" * 60)
    print("  Feature Engineering Pipeline (Phase A)")
    print("=" * 60)

    in_path = Path(args.csv)
    if not in_path.exists():
        print(f"❌ File not found: {in_path}")
        return 1

    print(f"\n[load] {in_path.name}")
    df = pd.read_csv(in_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Original columns: {len(df.columns)}")

    source_tf = detect_timeframe(df)
    print(f"  Detected TF: {source_tf}")

    # Auto-pick target TF
    if args.target_tf:
        target_tf = args.target_tf
    else:
        auto_map = {'M5': 'M30', 'M15': 'H1', 'M30': 'H4',
                    'H1': 'H4', 'H4': 'D1', 'D1': 'W1'}
        target_tf = auto_map.get(source_tf)
        if target_tf:
            print(f"  Auto target TF: {target_tf}")

    cols_before = list(df.columns)

    # === Multi-TF features ===
    if not args.no_mtf and target_tf and source_tf != 'unknown':
        df = add_multi_tf_features(df, source_tf, target_tf)
    else:
        print("  [MTF] skipped")

    # === Volatility regime ===
    if not args.no_volreg:
        df = add_volatility_regime(df)
    else:
        print("  [VolRegime] skipped")

    # === Range features ===
    if not args.no_range:
        df = add_range_features(df)
    else:
        print("  [Range] skipped")

    # Drop rows with NaN (from rolling windows)
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    print(f"\n[clean] Dropped {n_before - n_after} NaN rows ({(n_before-n_after)/n_before*100:.1f}%)")

    # Summary
    cols_after = list(df.columns)
    new_cols = [c for c in cols_after if c not in cols_before]
    print(f"\n[summary]")
    print(f"  Original: {len(cols_before)} cols")
    print(f"  Added:    {len(new_cols)} new features")
    for c in new_cols:
        print(f"    + {c}")
    print(f"  Total:    {len(cols_after)} cols")
    print(f"  Final rows: {len(df):,}")

    # Save
    out_path = Path(args.output) if args.output else \
        in_path.with_name(in_path.stem + '_enriched.csv')
    df.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[save] -> {out_path.name} ({size_mb:.1f} MB)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
