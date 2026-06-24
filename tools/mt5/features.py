"""
features.py — Feature Calculation Engine (Python version)
==========================================================
Reuse logic จาก DataCollector_v2.mq4 มาเป็น Python
ใช้ใน live_trader.py สำหรับคำนวณ features จาก live MT5 data

ต้องคำนวณให้ตรงกับ EA ทุกประการ — ไม่งั้น model ใช้ไม่ได้

ติดตั้ง:
    pip install pandas numpy pandas_ta

Usage:
    from features import calc_features

    df = mt5.copy_rates_from_pos(...)
    df = calc_features(df, h4_df, d1_df)
    # df จะมี 35 features พร้อมใช้
"""
import numpy as np
import pandas as pd
from typing import Optional


# =============================================================
# Indicator helpers (manual implementations — ไม่ต้อง import lib)
# =============================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI calculation (Wilder's smoothing)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return close.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd, signal, histogram)"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    """Returns (stoch_k, stoch_d)"""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R"""
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest + 1e-10)


def bollinger(close: pd.Series, period: int = 20, std: float = 2):
    """Returns (upper, middle, lower)"""
    middle = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = middle + std * sd
    lower = middle - std * sd
    return upper, middle, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average Directional Index"""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_val = atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_val + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_val + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def rolling_sharpe(close: pd.Series, period: int = 50) -> pd.Series:
    """Rolling Sharpe ratio of returns"""
    returns = close.pct_change()
    mean = returns.rolling(period).mean()
    std = returns.rolling(period).std()
    return mean / (std + 1e-10)


def rolling_zscore(close: pd.Series, period: int = 50) -> pd.Series:
    """Z-score of close in rolling window"""
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - mean) / (std + 1e-10)


def rolling_pct_rank(close: pd.Series, period: int = 100) -> pd.Series:
    """Percentile rank in rolling window (0-1)"""
    return close.rolling(period).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
    )


# =============================================================
# Main calc function
# =============================================================
def calc_features(
    df: pd.DataFrame,
    df_h4: Optional[pd.DataFrame] = None,
    df_d1: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculate 35 features เหมือน DataCollector_v2.mq4

    Args:
        df: DataFrame ของ H1 (มี columns: open, high, low, close, volume, time)
        df_h4: H4 dataframe (สำหรับ multi-TF features)
        df_d1: D1 dataframe (optional, ถ้าไม่มีจะ compute จาก df)

    Returns:
        df พร้อม features เพิ่มเข้ามา (35 columns)
    """
    df = df.copy()

    # Ensure datetime
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
        df = df.set_index('timestamp', drop=False)

    o, h, l, c = df['open'], df['high'], df['low'], df['close']

    # ============= GROUP 1: Base indicators (5) =============
    df['rsi'] = rsi(c, 14)
    df['ema_fast'] = ema(c, 20)
    df['ema_slow'] = ema(c, 50)
    df['ema_long'] = ema(c, 200)
    df['atr'] = atr(h, l, c, 14)

    # ============= GROUP 2: Multi-TF (5) =============
    # H4
    if df_h4 is not None and len(df_h4) > 50:
        h4_rsi_series = rsi(df_h4['close'], 14)
        h4_ema = ema(df_h4['close'], 50)
        df['rsi_h4'] = _align_to_h1(h4_rsi_series, df.index)
        h4_close = _align_to_h1(df_h4['close'], df.index)
        h4_ema_aligned = _align_to_h1(h4_ema, df.index)
        df['trend_h4'] = (h4_close > h4_ema_aligned).astype(int) * 2 - 1
    else:
        # Fallback: use H1 with longer period
        df['rsi_h4'] = rsi(c, 56)  # ~14 H4 periods
        df['trend_h4'] = (c > ema(c, 200)).astype(int) * 2 - 1

    # D1
    if df_d1 is not None and len(df_d1) > 50:
        d1_rsi_series = rsi(df_d1['close'], 14)
        d1_ema = ema(df_d1['close'], 50)
        df['rsi_d1'] = _align_to_h1(d1_rsi_series, df.index)
        d1_close = _align_to_h1(df_d1['close'], df.index)
        d1_ema_aligned = _align_to_h1(d1_ema, df.index)
        df['trend_d1'] = (d1_close > d1_ema_aligned).astype(int) * 2 - 1
    else:
        df['rsi_d1'] = rsi(c, 336)  # ~14 D1 periods
        df['trend_d1'] = (c > ema(c, 1000)).astype(int) * 2 - 1

    df['close_vs_ema200'] = (c - df['ema_long']) / (df['ema_long'] + 1e-10)

    # ============= GROUP 3: Momentum (7) =============
    macd_line, signal_line, hist = macd(c)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist

    stoch_k, stoch_d = stochastic(h, l, c)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    df['cci'] = cci(h, l, c, 14)
    df['wpr'] = williams_r(h, l, c, 14)

    # ============= GROUP 4: Volatility (3) =============
    bb_up, bb_mid, bb_lo = bollinger(c, 20, 2)
    df['bb_position'] = (c - bb_lo) / (bb_up - bb_lo + 1e-10)

    atr_long = atr(h, l, c, 50)
    df['atr_ratio'] = df['atr'] / (atr_long + 1e-10)

    df['adx'] = adx(h, l, c, 14)

    # ============= GROUP 5: Time (5) =============
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        df['hour'] = ts.dt.hour
        df['dow'] = ts.dt.dayofweek
        df['session_london'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
        df['session_ny']     = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
        df['session_asia']   = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
    else:
        df['hour'] = 12
        df['dow'] = 2
        df['session_london'] = 1
        df['session_ny'] = 1
        df['session_asia'] = 0

    # ============= GROUP 6: Lagged returns (5) =============
    df['ret_1']  = c.pct_change(1)
    df['ret_3']  = c.pct_change(3)
    df['ret_5']  = c.pct_change(5)
    df['ret_10'] = c.pct_change(10)
    df['ret_20'] = c.pct_change(20)

    # ============= GROUP 7: Statistical (5) =============
    df['close_zscore'] = rolling_zscore(c, 50)
    df['pct_rank'] = rolling_pct_rank(c, 100)
    df['sharpe_20'] = rolling_sharpe(c, 50)
    df['hl_range'] = (h - l) / (c + 1e-10)
    df['body_size'] = (c - o).abs() / (c + 1e-10)

    return df


def _align_to_h1(higher_tf_series: pd.Series, h1_index) -> pd.Series:
    """Align higher TF series ไปยัง H1 index (forward fill)"""
    return higher_tf_series.reindex(h1_index, method='ffill')


def get_feature_columns():
    """Return list of 35 feature column names"""
    return [
        # Base (5)
        'rsi', 'ema_fast', 'ema_slow', 'ema_long', 'atr',
        # Multi-TF (5)
        'rsi_h4', 'rsi_d1', 'trend_h4', 'trend_d1', 'close_vs_ema200',
        # Momentum (7)
        'macd', 'macd_signal', 'macd_hist',
        'stoch_k', 'stoch_d', 'cci', 'wpr',
        # Volatility (3)
        'bb_position', 'atr_ratio', 'adx',
        # Time (5)
        'hour', 'dow', 'session_london', 'session_ny', 'session_asia',
        # Lagged (5)
        'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
        # Statistical (5)
        'close_zscore', 'pct_rank', 'sharpe_20', 'hl_range', 'body_size',
    ]


if __name__ == "__main__":
    # Quick test
    import sys
    print("[features.py] Quick test")
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        # Convert timestamp if needed
        if 'timestamp' in df.columns and df['timestamp'].dtype == object:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        result = calc_features(df)
        print(f"Input rows : {len(df)}")
        print(f"Features   : {len(get_feature_columns())}")
        print(f"Output cols: {list(result.columns)}")
        print(f"\nLast row sample:")
        print(result[get_feature_columns()].tail(1).T)
    else:
        print("Usage: python features.py <csv_file>")
