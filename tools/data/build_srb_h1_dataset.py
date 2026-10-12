"""Python-first H1 training set for the SRB study (no MT5 collector involved).

    python tools/data/build_srb_h1_dataset.py usdjpy_tds_h1.csv usdjpy_tds_h4.csv --out usdjpy_tds_h1

Writes two rl_train.py-ready CSVs with identical rows:
  <out>_base.csv   OHLC + ~30 plain per-bar features computed here (returns, ATR,
                   candle shape, EMA distance/slope, RSI, rolling-extreme nearness,
                   lag-safe fractals, hour/day-of-week, sessions)
  <out>_srb.csv    the same + the SRB block from tools/data/srb_features.py:
                   srb_sig (-1/0/+1), srb_age (bars since the last signal, decayed),
                   dist/room/touches of the frozen H4 zone set, break flags
so an A/B with tools/analysis/obs_ablation.py isolates the SRB block.
All features use closed bars only (the SRB block uses the zone set that existed when
the bar opened). Every column is finite; the first `--warmup` bars are dropped.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from srb_features import SRBParams, atr_mt5, ema, srb_features  # noqa: E402

CAP = 10.0   # ATR-distance cap for "no zone in sight"


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean(); rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + ru / rd.replace(0, np.nan))).fillna(50).to_numpy()


def base_features(h1):
    o = h1["open"].to_numpy(float); h = h1["high"].to_numpy(float); l = h1["low"].to_numpy(float); c = h1["close"].to_numpy(float)
    a = atr_mt5(h, l, c, 14); a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
    lc = np.log(c); F = {}
    for k in (1, 4, 24):
        F[f"ret_{k}"] = np.r_[np.full(k, np.nan), (lc[k:] - lc[:-k])] / (a / c)   # log return in ATR units
    F["atr_pct"] = a / c * 100
    F["atr_ratio_96"] = a / pd.Series(a).rolling(96).mean().to_numpy()
    rng = h - l
    F["range_atr"] = rng / a; F["body_atr"] = (c - o) / a; F["close_loc"] = np.where(rng > 0, (c - l) / rng, 0.5)
    for n in (50, 200):
        e = ema(c, n); F[f"ema{n}_dist_atr"] = (c - e) / a
        F[f"ema{n}_slope_atr"] = np.r_[np.full(10, np.nan), (e[10:] - e[:-10])] / a
    F["rsi_14"] = rsi(c, 14)
    for n in (24, 120, 240):   # rolling-extreme nearness: the existing collector's closest analogue of a level
        F[f"near_high_{n}"] = (pd.Series(h).rolling(n).max().to_numpy() - c) / a
        F[f"near_low_{n}"] = (c - pd.Series(l).rolling(n).min().to_numpy()) / a
    # lag-safe Bill Williams fractals: flag at bar i says bar i-2 was a 5-bar swing (known only at i)
    fu = np.zeros(len(c)); fd = np.zeros(len(c))
    for i in range(4, len(c)):
        m = i - 2
        if h[m] > max(h[m - 2], h[m - 1], h[m + 1], h[m + 2]): fu[i] = 1
        if l[m] < min(l[m - 2], l[m - 1], l[m + 1], l[m + 2]): fd[i] = 1
    F["fractal_up"] = fu; F["fractal_down"] = fd
    t = pd.DatetimeIndex(h1["t"]); hr = t.hour.to_numpy(); dow = t.dayofweek.to_numpy()
    F["hour_sin"] = np.sin(2 * np.pi * hr / 24); F["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    F["dow_sin"] = np.sin(2 * np.pi * dow / 5); F["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    F["session_asia"] = ((hr >= 0) & (hr < 9)).astype(float); F["session_london"] = ((hr >= 9) & (hr < 17)).astype(float)
    F["session_ny"] = ((hr >= 14) & (hr < 23)).astype(float)
    return pd.DataFrame(F), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h1"); ap.add_argument("h4"); ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=300, help="bars dropped at the start (rolling windows + zone history)")
    ap.add_argument("--start", default="", help="keep bars from this date (after warm-up)")
    args = ap.parse_args()
    h1 = pd.read_csv(args.h1); h4 = pd.read_csv(args.h4)
    for d in (h1, h4): d["t"] = pd.to_datetime(d["timestamp"], format="%Y.%m.%d %H:%M")
    B, atr = base_features(h1)
    S = srb_features(h1, h4, SRBParams())
    sig = (S["sig_buy"] - S["sig_sell"]).to_numpy(float)
    age = np.full(len(sig), 24.0); last = -10 ** 9
    for i in range(len(sig)):
        if sig[i] != 0: last = i
        age[i] = min(24.0, float(i - last))
    srb = pd.DataFrame({
        "srb_sig": sig, "srb_age": age / 24.0,
        "srb_last_dir": pd.Series(np.where(sig != 0, sig, np.nan)).ffill().fillna(0).to_numpy() * (age < 24),
        "srb_dist_res": S["dist_res_atr"].fillna(CAP).clip(upper=CAP), "srb_dist_sup": S["dist_sup_atr"].fillna(CAP).clip(upper=CAP),
        "srb_res_touches": S["res_touches"].astype(float), "srb_sup_touches": S["sup_touches"].astype(float),
        "srb_room_up": S["room_up_atr"].fillna(CAP).clip(upper=CAP), "srb_room_dn": S["room_dn_atr"].fillna(CAP).clip(upper=CAP),
        "srb_brk_res": S["brk_res"].astype(float), "srb_brk_sup": S["brk_sup"].astype(float),
        "srb_n_res": S["n_res"].astype(float), "srb_n_sup": S["n_sup"].astype(float),
    })
    head = pd.DataFrame({"timestamp": h1["t"].dt.strftime("%Y.%m.%d %H:%M"), "open": h1["open"], "high": h1["high"],
                         "low": h1["low"], "close": h1["close"], "volume": h1.get("tick_volume", 0)})
    base = pd.concat([head, B], axis=1); full = pd.concat([head, B, srb], axis=1)
    keep = np.ones(len(full), bool); keep[:args.warmup] = False
    keep &= np.isfinite(full.drop(columns=["timestamp"]).to_numpy(float)).all(axis=1)
    if args.start: keep &= (h1["t"] >= args.start).to_numpy()
    base = base[keep].reset_index(drop=True); full = full[keep].reset_index(drop=True)
    pb = Path(f"{args.out}_base.csv"); pf = Path(f"{args.out}_srb.csv")
    base.to_csv(pb, index=False); full.to_csv(pf, index=False)
    ns = int((full["srb_sig"] != 0).sum())
    print(f"{pb}: {len(base)} rows x {base.shape[1] - 6} features | {pf}: +{srb.shape[1]} SRB columns, "
          f"{ns} signals ({ns / len(full):.2%} of bars), {full['timestamp'].iloc[0]} -> {full['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
