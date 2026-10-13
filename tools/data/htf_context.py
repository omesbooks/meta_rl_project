"""Higher-timeframe context features for a lower-TF bar table, lag-safe.

    ctx = htf_context(h1, rule="4h", prefix="h4")      # or rule="1D", prefix="d1"

Bars are resampled (label=left, closed=left - the same alignment MT5 uses, verified
against the terminal's own H4 bars), the indicators are computed on the higher TF, and
each lower-TF bar receives the values of the LAST higher-TF bar that had CLOSED by the
time the lower-TF bar opened (merge_asof on the higher bar's close time). The higher-TF
bar that contains the current bar is still forming and is never used.

Columns (all ATR-normalised on the higher TF unless noted):
  <p>_ema50_dist, <p>_ema200_dist, <p>_ema50_slope, <p>_rsi14 (0-100), <p>_adx14 (0-100),
  <p>_atr_pct (ATR/close*100), <p>_atr_ratio (ATR / 50-bar mean), <p>_range_pos20 (0-1),
  <p>_ret1, <p>_ret5, <p>_trend (+1/-1 ema50 vs ema200)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {"4h": pd.Timedelta("4h"), "1D": pd.Timedelta("1D"), "1W": pd.Timedelta("7D")}


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _rma(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru, rd = _rma(up, n), _rma(dn, n)
    return np.where(rd > 0, 100 - 100 / (1 + ru / np.where(rd > 0, rd, 1)), 100.0)


def _adx(h, l, c, n=14):
    """Wilder ADX: +DM/-DM, TR smoothed with RMA, DX, ADX = RMA(DX)."""
    up = np.diff(h, prepend=h[0]); dn = -np.diff(l, prepend=l[0])
    plus = np.where((up > dn) & (up > 0), up, 0.0); minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = _rma(tr, n); atr = np.where(atr > 0, atr, np.nan)
    pdi = 100 * _rma(plus, n) / atr; mdi = 100 * _rma(minus, n) / atr
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi > 0, pdi + mdi, np.nan)
    return _rma(np.nan_to_num(dx), n)


def _atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


def htf_context(h1: pd.DataFrame, rule: str = "4h", prefix: str = "h4") -> pd.DataFrame:
    if "t" not in h1:
        raise ValueError("h1 needs a datetime column 't'")
    bars = (h1.set_index("t")[["open", "high", "low", "close"]]
            .resample(rule, label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    o = bars["open"].to_numpy(float); h = bars["high"].to_numpy(float); l = bars["low"].to_numpy(float); c = bars["close"].to_numpy(float)
    a = _atr(h, l, c, 14); a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
    e50, e200 = _ema(c, 50), _ema(c, 200)
    lc = np.log(c)
    F = pd.DataFrame({
        f"{prefix}_ema50_dist": (c - e50) / a,
        f"{prefix}_ema200_dist": (c - e200) / a,
        f"{prefix}_ema50_slope": np.r_[np.full(5, np.nan), e50[5:] - e50[:-5]] / a,
        f"{prefix}_rsi14": _rsi(c, 14),
        f"{prefix}_adx14": _adx(h, l, c, 14),
        f"{prefix}_atr_pct": a / c * 100,
        f"{prefix}_atr_ratio": a / pd.Series(a).rolling(50).mean().to_numpy(),
        f"{prefix}_range_pos20": (c - pd.Series(l).rolling(20).min().to_numpy()) /
                                  np.maximum(1e-9, pd.Series(h).rolling(20).max().to_numpy() - pd.Series(l).rolling(20).min().to_numpy()),
        f"{prefix}_ret1": np.r_[np.nan, lc[1:] - lc[:-1]] / (a / c),
        f"{prefix}_ret5": np.r_[np.full(5, np.nan), lc[5:] - lc[:-5]] / (a / c),
        f"{prefix}_trend": np.where(e50 > e200, 1.0, -1.0),
    }, index=bars.index)
    # a higher bar is usable only after it CLOSED: open + rule duration
    F["available_at"] = F.index + RULES[rule]
    F = F.sort_values("available_at")
    out = pd.merge_asof(h1[["t"]].sort_values("t"), F, left_on="t", right_on="available_at", direction="backward")
    out.index = h1.index
    return out.drop(columns=["t", "available_at"])
