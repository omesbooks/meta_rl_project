"""tools/data/htf_context.py: higher-TF context on lower-TF bars must be lag-safe — an H1
bar sees only higher-TF bars that CLOSED before it opened — and must line up with the
higher-TF bar boundaries MT5 uses (label=left)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "data"))
from htf_context import htf_context  # noqa: E402


def _h1(n=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=n, freq="h"), "open": c, "high": c + 0.2, "low": c - 0.2, "close": c})
    return df


def test_h4_value_comes_from_the_last_closed_h4_bar():
    df = _h1()
    ctx = htf_context(df, "4h", "h4")
    # H1 bars 04:00..07:00 belong to the H4 bar opening 04:00 (still forming) -> they must all carry
    # the values of the H4 bar 00:00-04:00, i.e. identical to each other and equal to the 08:00 bar? no:
    # 08:00 sees the 04:00 bar. So: rows 4..7 equal each other; row 8 differs from row 7.
    v = ctx["h4_rsi14"].to_numpy()
    i = 24 * 10 + 4                                                   # a 04:00 bar well after warm-up
    assert np.allclose(v[i:i + 4], v[i]) and not np.isclose(v[i + 4], v[i])
    # and the value equals the RSI computed on H4 bars up to the one that closed at 04:00
    h4 = df.set_index("t").resample("4h", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    from htf_context import _rsi
    rsi_h4 = _rsi(h4["close"].to_numpy(), 14)
    k = h4.index.get_loc(df["t"].iloc[i].floor("4h") - pd.Timedelta("4h"))   # the bar that closed at 04:00
    assert np.isclose(v[i], rsi_h4[k])


def test_future_bars_do_not_change_past_context():
    df = _h1()
    a = htf_context(df, "1D", "d1")
    df2 = df.copy(); df2.loc[df2.index >= len(df2) - 48, ["open", "high", "low", "close"]] *= 1.05
    b = htf_context(df2, "1D", "d1")
    cut = len(df) - 48
    safe = df["t"] < df["t"].iloc[cut].floor("1D")      # bars whose D1 source closed before the edit
    pd.testing.assert_frame_equal(a[safe].reset_index(drop=True), b[safe].reset_index(drop=True))
    assert a.columns.tolist()[0] == "d1_ema50_dist" and a.shape[1] == 11


def test_no_nan_after_warmup_and_sane_ranges():
    df = _h1(24 * 120)
    ctx = pd.concat([htf_context(df, "4h", "h4"), htf_context(df, "1D", "d1")], axis=1)
    tail = ctx.iloc[24 * 80:]      # d1_atr_ratio needs 14 + 50 daily bars
    assert np.isfinite(tail.to_numpy(float)).all()
    assert tail["h4_rsi14"].between(0, 100).all() and tail["d1_adx14"].between(0, 100).all()
    assert tail["h4_range_pos20"].between(0, 1).all() and set(tail["d1_trend"].unique()) <= {-1.0, 1.0}
