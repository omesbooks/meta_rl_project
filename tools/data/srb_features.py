"""Python twin of the Metafxclub SRB v02 signal (swing support/resistance breakout).

Reproduces, bar for bar, what the MT5 EA computes before it trades:
  * zones  - swing highs/lows on the level TF (H4), merged within ATR(H4)*tol,
             kept when touched >= min_touches times (InpTouchMode 0: bars inside the zone)
  * signal - trading-TF (H1) bar whose close crosses a zone by ATR(H1)*break_buf,
             bar range >= ATR*strong, close on the right side of the EMA
The EA (v02 defaults: FreezeSnapshot on) tests every closed H1 bar against the zone set
that existed when the bar opened, i.e. the zones built at the open of the H4 bar that
contains it, from COMPLETE H4 bars only - so nothing here looks ahead.

State that depends on fills (one position at a time, one buy + one sell per day,
84-bar cooldown per traded zone) is applied separately in `apply_state`, driven by
the position intervals of a tester report, so the pure market signal and the
EA's gating can be compared on their own.

Features (per H1 bar, all ATR(H1)-normalised, from the frozen zone set):
  dist_res_atr, dist_sup_atr   distance from close to the nearest zone above / below
  res_touches, sup_touches     touch count of those zones (0 when none)
  n_res, n_sup                 zones in the set
  zone_tol_atr                 zone half-width / ATR(H1)
  room_up_atr, room_dn_atr     distance to the NEXT zone beyond the nearest (measured move)
  brk_res, brk_sup             breakout flag with the EA's buffer (+1 / 0), level in brk_level
  candle_range_atr, close_loc  (h-l)/ATR, (c-l)/(h-l)
  ema_dist_atr                 (close - EMA)/ATR
  sig_buy, sig_sell            full entry condition (break + strong + EMA bias), no state
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SRBParams:
    lookback: int = 60          # InpLookbackBars (level TF)
    swing_left: int = 2         # InpSwingLeft  (older bars)
    swing_right: int = 3        # InpSwingRight (newer bars)
    level_atr: int = 14         # InpLevelAtrPeriod
    touch_tol_atr: float = 0.60 # InpTouchTolATR
    min_touches: int = 3        # InpMinTouches
    touch_mode: int = 0         # InpTouchMode (0 = bars inside the zone)
    break_buf_atr: float = 0.50 # InpBreakBufferATR
    atr_period: int = 14        # InpAtrPeriod (trading TF)
    strong_atr: float = 0.50    # InpStrongAtrMult
    ema_period: int = 50        # InpEmaPeriod
    use_ema_bias: bool = True
    cooldown_bars: int = 84     # InpLevelCooldownBars
    sl_atr: float = 1.5         # InpSlAtrMult (mode 2)
    tp_measured: float = 0.50   # InpTpMeasuredMult (mode 2)
    tp_atr: float = 5.0         # InpTpAtrMult fallback
    order1_rr: float = 1.25     # InpOrder1RR


def atr_mt5(h, l, c, period):
    """MT5 built-in iATR (Indicators/Examples/ATR.mq5): a SIMPLE moving average of the
    true range max(high, prev close) - min(low, prev close) - NOT Wilder smoothing.
    TR of the first bar is 0 and the first value sits at index `period`."""
    h = np.asarray(h, float); l = np.asarray(l, float); c = np.asarray(c, float)
    tr = np.zeros(len(h))
    pc = c[:-1]
    tr[1:] = np.maximum(h[1:], pc) - np.minimum(l[1:], pc)
    out = np.full(len(h), np.nan)
    if len(h) <= period:
        return out
    cs = np.cumsum(tr)
    out[period:] = (cs[period:] - cs[:-period]) / period
    return out


def ema(x, period):
    """MT5 iMA(MODE_EMA): seeded with the first price."""
    x = np.asarray(x, float); out = np.empty(len(x)); a = 2.0 / (period + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _count_touches(px, scan, level, tol, mode):
    if mode == 1:
        inside = False; n = 0
        for j in range(scan, 0, -1):
            inn = abs(px[j] - level) <= tol
            if inn and not inside:
                n += 1
            inside = inn
        return n
    return int(np.sum(np.abs(px[1:scan + 1] - level) <= tol))


def _zones(pts, px, scan, tol, p: SRBParams):
    """BuildZones: sorted swing prices merged while within 2*tol of the group's first point."""
    out = []
    if len(pts) == 0:
        return out
    pts = np.sort(np.asarray(pts, float))
    s = pts[0]; cnt = 1; z0 = pts[0]
    for i in range(1, len(pts) + 1):
        flush = i == len(pts) or pts[i] - z0 > 2.0 * tol
        if not flush:
            s += pts[i]; cnt += 1; continue
        level = s / cnt
        t = _count_touches(px, scan, level, tol, p.touch_mode)
        if t >= p.min_touches:
            out.append((level, t))
        if i < len(pts):
            s = pts[i]; cnt = 1; z0 = pts[i]
    return out


def build_zone_sets(h4: pd.DataFrame, p: SRBParams):
    """Zone set at the OPEN of every H4 bar k, from complete bars k-1 .. k-need+1 (MQL
    series index j = bar k-j; index 0 is the forming bar and is never read).
    Returns dict: h4 open time -> (res [(level, touches)], sup [...], tol)."""
    hi = h4["high"].to_numpy(float); lo = h4["low"].to_numpy(float)
    atr4 = atr_mt5(hi, lo, h4["close"].to_numpy(float), p.level_atr)
    need = p.lookback + p.swing_left + p.swing_right + 3
    last = p.lookback + p.swing_right
    scan = p.lookback + p.swing_right
    times = h4["t"].to_numpy()
    sets = {}
    for k in range(need, len(h4)):
        a = atr4[k - 1]                                   # ATR(H4) at shift 1
        if not np.isfinite(a) or a <= 0:
            continue
        tol = a * p.touch_tol_atr
        H = hi[k - need + 1:k + 1][::-1]                  # H[j] = bar k-j
        L = lo[k - need + 1:k + 1][::-1]
        sh = []; sl = []
        for i in range(p.swing_right + 1, last + 1):
            is_h = True; is_l = True
            for kk in range(1, p.swing_right + 1):        # newer bars: strictly lower / higher
                if H[i - kk] >= H[i]: is_h = False
                if L[i - kk] <= L[i]: is_l = False
                if not (is_h or is_l): break
            for k2 in range(1, p.swing_left + 1):         # older bars: not beyond
                if not (is_h or is_l): break
                if i + k2 >= need: is_h = is_l = False; break
                if H[i + k2] > H[i]: is_h = False
                if L[i + k2] < L[i]: is_l = False
            if is_h: sh.append(H[i])
            if is_l: sl.append(L[i])
        sets[times[k]] = (_zones(sh, H, scan, tol, p), _zones(sl, L, scan, tol, p), tol)
    return sets


def srb_features(h1: pd.DataFrame, h4: pd.DataFrame, p: SRBParams = SRBParams()) -> pd.DataFrame:
    """Per-H1-bar SRB features + raw entry conditions (see module docstring)."""
    h1 = h1.copy(); h4 = h4.copy()
    for d in (h1, h4):
        if "t" not in d:
            d["t"] = pd.to_datetime(d["timestamp"], format="%Y.%m.%d %H:%M")
    sets = build_zone_sets(h4, p)
    o = h1["open"].to_numpy(float); h = h1["high"].to_numpy(float); l = h1["low"].to_numpy(float); c = h1["close"].to_numpy(float)
    atr = atr_mt5(h, l, c, p.atr_period)
    em = ema(c, p.ema_period)
    n = len(h1)
    cols = {k: np.full(n, np.nan) for k in ("dist_res_atr", "dist_sup_atr", "zone_tol_atr", "room_up_atr", "room_dn_atr",
                                            "candle_range_atr", "close_loc", "ema_dist_atr", "brk_level", "res_level", "sup_level")}
    ints = {k: np.zeros(n, int) for k in ("res_touches", "sup_touches", "n_res", "n_sup", "brk_res", "brk_sup", "sig_buy", "sig_sell")}
    t4 = h1["t"].dt.floor("4h").to_numpy()
    for i in range(1, n):
        zs = sets.get(t4[i])
        a = atr[i]
        if zs is None or not np.isfinite(a) or a <= 0:
            continue
        res, sup, tol = zs
        ci, cp = c[i], c[i - 1]
        rng = h[i] - l[i]
        cols["candle_range_atr"][i] = rng / a
        cols["close_loc"][i] = (ci - l[i]) / rng if rng > 0 else 0.5
        cols["ema_dist_atr"][i] = (ci - em[i]) / a
        cols["zone_tol_atr"][i] = tol / a
        ints["n_res"][i] = len(res); ints["n_sup"][i] = len(sup)
        above = [z for z in res if z[0] >= ci]; below = [z for z in sup if z[0] <= ci]
        if above:
            lv, tc = min(above, key=lambda z: z[0]); cols["dist_res_atr"][i] = (lv - ci) / a; ints["res_touches"][i] = tc; cols["res_level"][i] = lv
            nxt = [z[0] for z in res if z[0] > lv + tol]
            cols["room_up_atr"][i] = (min(nxt) - lv) / a if nxt else np.nan
        if below:
            lv, tc = max(below, key=lambda z: z[0]); cols["dist_sup_atr"][i] = (ci - lv) / a; ints["sup_touches"][i] = tc; cols["sup_level"][i] = lv
            nxt = [z[0] for z in sup if z[0] < lv - tol]
            cols["room_dn_atr"][i] = (lv - max(nxt)) / a if nxt else np.nan
        buf = a * p.break_buf_atr
        strong = p.strong_atr <= 0 or rng >= a * p.strong_atr
        # BrokenResistance: highest zone with prev close below and close above level + buffer
        rb = [z[0] for z in res if cp < z[0] and ci > z[0] + buf]
        sb = [z[0] for z in sup if cp > z[0] and ci < z[0] - buf]
        if rb:
            ints["brk_res"][i] = 1; cols["brk_level"][i] = max(rb)
            if strong and (not p.use_ema_bias or ci > em[i]):
                ints["sig_buy"][i] = 1
        if sb:
            ints["brk_sup"][i] = 1
            if not rb: cols["brk_level"][i] = min(sb)
            if strong and (not p.use_ema_bias or ci < em[i]):
                ints["sig_sell"][i] = 1
    out = pd.DataFrame({"t": h1["t"], "open": o, "high": h, "low": l, "close": c, "atr": atr, "ema": em, **cols, **ints})
    return out


def apply_state(feat: pd.DataFrame, p: SRBParams, positions: pd.DataFrame | None = None):
    """EA gating on top of the raw signals. `positions`: DataFrame(entry, exit) of the
    report's positions (one row per signal: first fill -> last leg closed); when given,
    the one-at-a-time gate follows the report, otherwise no position gate is applied.
    Returns a DataFrame of fired signals with the blocked reason for raw signals."""
    t = feat["t"].to_numpy(); n = len(feat)
    ev = t + np.timedelta64(1, "h")                        # evaluation = open of the next bar
    pos_entry = pos_exit = None
    if positions is not None and len(positions):
        pos_entry = positions["entry"].to_numpy(); pos_exit = positions["exit"].to_numpy()
    day = pd.DatetimeIndex(feat["t"]).dayofyear.to_numpy() + pd.DatetimeIndex(feat["t"]).year.to_numpy() * 1000
    recs = {True: [], False: []}
    traded = {True: False, False: False}; cur_day = -1
    rows = []
    for i in range(n):
        if day[i] != cur_day:
            cur_day = day[i]; traded[True] = traded[False] = False
        for is_buy, sig in ((True, feat["sig_buy"].iat[i]), (False, feat["sig_sell"].iat[i])):
            if not sig:
                continue
            level = feat["brk_level"].iat[i]; tol = feat["zone_tol_atr"].iat[i] * feat["atr"].iat[i]
            reason = ""
            if traded[is_buy]:
                reason = "day_flag"
            elif pos_entry is not None and np.any((pos_entry < ev[i]) & (pos_exit >= ev[i])):
                reason = "position_open"
            else:
                for lv, tl, b in recs[is_buy]:
                    if i - b < p.cooldown_bars and abs(level - lv) <= max(tol, tl):
                        reason = "cooldown"; break
            rows.append({"t": t[i], "eval": ev[i], "dir": "buy" if is_buy else "sell", "level": level,
                         "atr": feat["atr"].iat[i], "blocked": reason})
            if reason == "":
                traded[is_buy] = True
                recs[is_buy] = [r for r in recs[is_buy] if i - r[2] < p.cooldown_bars] + [(level, tol, i)]
            if is_buy and reason == "":
                break      # the EA returns after an attempted buy; a cooled / day-flagged buy falls through to the sell test
    return pd.DataFrame(rows)
