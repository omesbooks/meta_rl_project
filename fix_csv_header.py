"""
Fix DataCollector_v3 CSV — เพิ่ม header + clean BOM
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CSV_PATH = "training_data_v3.csv"

# Reconstruct header based on default v3 settings:
# RSI=AGGREGATE, EMA=MULTI(4), ATR/Stoch/CCI/WPR/ADX=AGGREGATE
HEADER = [
    # Meta (7)
    "timestamp", "symbol", "open", "high", "low", "close", "volume",
    # RSI AGGREGATE (4)
    "rsi_min", "rsi_max", "rsi_mean", "rsi_std",
    # EMA MULTI 20/50/100/200 (4)
    "ema_20", "ema_50", "ema_100", "ema_200",
    # ATR AGGREGATE (4)
    "atr_min", "atr_max", "atr_mean", "atr_std",
    # Stoch AGGREGATE (4)
    "stoch_min", "stoch_max", "stoch_mean", "stoch_std",
    # CCI AGGREGATE (4)
    "cci_min", "cci_max", "cci_mean", "cci_std",
    # WPR AGGREGATE (4)
    "wpr_min", "wpr_max", "wpr_mean", "wpr_std",
    # ADX AGGREGATE (4)
    "adx_min", "adx_max", "adx_mean", "adx_std",
    # Static others (5)
    "ema_long", "macd", "macd_signal", "macd_hist", "bb_position",
    # Time (5)
    "hour", "dow", "session_london", "session_ny", "session_asia",
    # Lagged returns (5)
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    # Statistical (5)
    "close_zscore", "pct_rank", "sharpe_20", "hl_range", "body_size",
    # Label (2)
    "future_return", "target",
]
print(f"Total columns in header: {len(HEADER)}")

# Read existing CSV
with open(CSV_PATH, 'r', encoding='utf-8-sig', newline='') as f:
    content = f.read()

# Prepend header
header_line = ",".join(HEADER) + "\n"
fixed_content = header_line + content

# Write back
with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
    f.write(fixed_content)

print(f"[OK] Header prepended to {CSV_PATH}")

# Verify
import pandas as pd
df = pd.read_csv(CSV_PATH)
print(f"\n[verify] Loaded {len(df):,} rows × {len(df.columns)} cols")
print(f"\nColumn names: {list(df.columns)[:5]} ... {list(df.columns)[-3:]}")
print(f"\nFirst row:")
print(df.iloc[0].head(10))
print(f"\nClass distribution:")
print(df['target'].value_counts())
