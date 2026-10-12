"""tools/data/srb_features.py — the Python twin of the SRB EA must use MT5's ATR
definition, find a multi-touch zone, fire on a buffered close above it, and never
look ahead. (Parity against the tester report itself is a data run, see
docs/srb_event_edge_2026-10-12.md: 1107/1107 entries reproduced.)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "data"))
from srb_features import SRBParams, apply_state, atr_mt5, srb_features  # noqa: E402


def test_atr_is_mt5_simple_average_of_true_range():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 0.1, 200)); h = c + rng.uniform(0, 0.2, 200); l = c - rng.uniform(0, 0.2, 200)
    a = atr_mt5(h, l, c, 14)
    tr = np.r_[0.0, np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])]
    assert np.isnan(a[13]) and np.isfinite(a[14])
    assert np.allclose(a[14:], pd.Series(tr).rolling(14).mean().to_numpy()[14:])


def _bars(h1_closes, start="2024-01-01"):
    """H1 bars from a close path (small bodies), plus the matching H4 bars."""
    t = pd.date_range(start, periods=len(h1_closes), freq="h")
    c = np.asarray(h1_closes, float); o = np.r_[c[0], c[:-1]]
    h1 = pd.DataFrame({"t": t, "open": o, "high": np.maximum(o, c) + 0.05, "low": np.minimum(o, c) - 0.05, "close": c})
    h4 = h1.set_index("t").resample("4h", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    return h1, h4


def test_zone_then_buffered_break_fires_buy_and_not_before():
    # H4 bars oscillating 99..100.4 with a distinct swing high at 102.0 every 16 bars (4 touches
    # in the 63-bar scan, > 2*tol away from the 100.4 cluster so it is its own zone), then an
    # H1 close through it. Swings closer than 2*tol are merged into one zone, as in the EA.
    n4 = 68 * 4
    base = []
    for k in range(n4 // 4):
        base += [99.3, 100.0, 102.0 if k % 4 == 3 else 100.4, 99.8]
    h1 = np.repeat(base, 4) + np.tile([0.0, 0.03, -0.03, 0.0], len(base))
    # breakout: a strong H1 bar closing well above 102 + buffer, preceded by a quiet bar below the zone
    h1 = np.r_[h1, [100.6, 100.6, 103.6]]
    H1, H4 = _bars(h1)
    H1.loc[len(H1) - 1, ["open", "low"]] = [100.6, 100.5]               # range 3.1 >> 0.5 ATR
    p = SRBParams()
    F = srb_features(H1, H4, p)
    assert F["n_res"].iloc[-1] >= 1 and F["res_touches"].iloc[-1] == 0  # the zones are now below the close
    assert F["sig_buy"].iloc[-1] == 1 and F["brk_res"].iloc[-1] == 1
    assert abs(F["brk_level"].iloc[-1] - 102.05) < 0.1                   # the 102 swing-high zone, not the 100.4 cluster
    early = F.iloc[:-1]
    assert (early.loc[early["sig_buy"] == 1, "close"] > 101.5).all()     # earlier fires are the 102 spikes breaking the 100.4 zone, never bars inside it
    assert early["sig_sell"].sum() == 0                                  # nothing closes below the 99.2 support by a buffer
    fired = apply_state(F, p)
    last = fired.iloc[-1]
    assert fired["dir"].eq("buy").all() and last["t"] == F["t"].iloc[-1]
    # one buy per calendar day: the final break is gated only when an earlier spike fired the same day
    same_day = (early["t"].dt.date == F["t"].iloc[-1].date()) & (early["sig_buy"] == 1)
    assert last["blocked"] == ("day_flag" if same_day.any() else "")


def test_no_lookahead_when_future_bars_change():
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 0.08, 68 * 4 * 3))
    H1, H4 = _bars(closes)
    A = srb_features(H1, H4)
    H1b = H1.copy(); cut = len(H1) - 40
    H1b.loc[cut:, ["open", "high", "low", "close"]] *= 1.03               # rewrite the last 40 bars
    H4b = H1b.set_index("t").resample("4h", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    B = srb_features(H1b, H4b)
    cols = ["dist_res_atr", "dist_sup_atr", "res_touches", "sig_buy", "sig_sell", "atr", "ema", "brk_level"]
    # every bar before the cut (and its H4 block) must be identical: the twin only reads the past
    safe = H1["t"] < H1["t"].iloc[cut].floor("4h")
    pd.testing.assert_frame_equal(A.loc[safe, cols].reset_index(drop=True), B.loc[safe, cols].reset_index(drop=True))
