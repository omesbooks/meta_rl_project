"""Step B of the SRB study: does the SRB breakout EVENT carry forward-return edge on its
own, or does the EA's money come from its exit structure?

    python tools/analysis/srb_event_edge.py usdjpy_tds_h1.csv usdjpy_tds_h4.csv \
        [--report_signals <report_signals.csv>] [--out docs/srb_event_edge_<date>.md]

Everything is in ATR(H1) or R units (R = 1.5 * ATR, the EA's stop) so years with
different volatility are comparable and compounding does not inflate late years.
Entry = open of the bar after the signal bar; the bar's spread (MT5 bars are bid prices) plus
--cost_pips is charged per trade. Intrabar SL-vs-TP ambiguity is resolved pessimistically (SL first).
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "data"))
from srb_features import SRBParams, apply_state, srb_features  # noqa: E402

HORIZONS = (1, 2, 4, 8, 24, 48, 96)
COST = 0.0          # extra round-trip cost in price units (set from --cost_pips)
RNG = np.random.default_rng(0)


def fwd_table(F, idx, sgn):
    """Signed forward return in ATR from the next bar's open (spread-adjusted) at each horizon."""
    o = F["open"].to_numpy(); c = F["close"].to_numpy(); a = F["atr"].to_numpy(); sp = F["spread_px"].to_numpy()
    n = len(F); out = {}
    entry = o[np.minimum(idx + 1, n - 1)] + sgn * (sp[np.minimum(idx + 1, n - 1)] + COST)
    for h in HORIZONS:
        j = idx + h
        ok = j < n
        r = np.full(len(idx), np.nan)
        r[ok] = sgn[ok] * (c[j[ok]] - entry[ok]) / a[idx[ok]]
        out[h] = r
    return pd.DataFrame(out)


def exit_sim(F, idx, sgn, sl_r=1.0, tp_r=None, max_bars=None, be_at=None):
    """Bar-level trade from the next bar's open: SL at sl_r*R, optional TP, optional
    time stop, optional breakeven once be_at*R in profit. R = 1.5*ATR(signal bar).
    Returns the result in R per event (pessimistic: SL checked before TP in a bar)."""
    o = F["open"].to_numpy(); h = F["high"].to_numpy(); l = F["low"].to_numpy(); c = F["close"].to_numpy()
    a = F["atr"].to_numpy(); sp = F["spread_px"].to_numpy(); n = len(F)
    res = np.full(len(idx), np.nan)
    for q, (i, s) in enumerate(zip(idx, sgn)):
        if i + 1 >= n: continue
        R = 1.5 * a[i]; e = o[i + 1] + s * (sp[i + 1] + COST)
        sl = e - s * sl_r * R; tp = e + s * tp_r * R if tp_r else None
        last = min(n - 1, i + (max_bars or 10 ** 9))
        for j in range(i + 1, last + 1):
            if (s > 0 and l[j] <= sl) or (s < 0 and h[j] >= sl):
                res[q] = s * (sl - e) / R; break
            if tp is not None and ((s > 0 and h[j] >= tp) or (s < 0 and l[j] <= tp)):
                res[q] = tp_r; break
            if be_at and sl != e and ((s > 0 and h[j] >= e + be_at * R) or (s < 0 and l[j] <= e - be_at * R)):
                sl = e                                      # move the stop to entry (next bars)
            if j == last:
                res[q] = s * (c[j] - e) / R
    return res


def mstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan, np.nan, 0
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)


def fmt(m, se):
    if not np.isfinite(m): return "–"
    z = m / se if se > 0 else np.nan
    return f"{m:+.3f} ± {se:.3f} (z {z:+.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h1"); ap.add_argument("h4")
    ap.add_argument("--report_signals", default=None, help="per-signal table parsed from the tester report")
    ap.add_argument("--start", default="2015-01-01"); ap.add_argument("--oos_from", default="2026-01-01")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cost_pips", type=float, default=0.0, help="extra round-trip cost per trade in pips (0.01) on top of the bar spread")
    args = ap.parse_args()
    global COST
    COST = args.cost_pips * 0.01
    h1 = pd.read_csv(args.h1); h4 = pd.read_csv(args.h4)
    for d in (h1, h4): d["t"] = pd.to_datetime(d["timestamp"], format="%Y.%m.%d %H:%M")
    p = SRBParams()
    t0 = time.time()
    F = srb_features(h1, h4, p)
    F["spread_px"] = h1["spread"].to_numpy(float) * 0.001 if "spread" in h1 else 0.0
    F["dir_bar"] = np.sign(F["close"] - F["open"])
    print(f"twin: {len(F)} bars, {time.time() - t0:.1f}s")
    valid = (F["t"] >= args.start) & F["atr"].notna() & (F["atr"] > 0)
    oos = F["t"] >= args.oos_from

    # ---- event sets --------------------------------------------------------------
    sets = {}
    sb = valid & (F["sig_buy"] == 1); ss = valid & (F["sig_sell"] == 1)
    sets["SRB signal (all filters, no state)"] = (np.r_[np.where(sb)[0], np.where(ss)[0]], np.r_[np.ones(sb.sum()), -np.ones(ss.sum())])
    fired = apply_state(F, p, None)
    fired = fired[(fired["blocked"] == "") & (fired["t"] >= args.start)]
    fi = F.index[F["t"].isin(fired["t"])].to_numpy()
    fs = np.where(F.loc[fi, "sig_buy"] == 1, 1.0, -1.0)
    sets["SRB fired (after day flag + cooldown gates)"] = (fi, fs)
    strong = valid & (F["candle_range_atr"] >= p.strong_atr)
    emab = (F["dir_bar"] > 0) & (F["ema_dist_atr"] > 0) | (F["dir_bar"] < 0) & (F["ema_dist_atr"] < 0)
    nb = strong & emab & (F["brk_res"] == 0) & (F["brk_sup"] == 0)
    sets["null: strong candle + EMA side, NO zone break"] = (np.where(nb)[0], F.loc[nb, "dir_bar"].to_numpy(float))
    bo = valid & ((F["brk_res"] == 1) | (F["brk_sup"] == 1)) & (F["sig_buy"] == 0) & (F["sig_sell"] == 0)
    bd = np.where(F.loc[bo, "brk_res"] == 1, 1.0, -1.0)
    sets["null: zone break that FAILED strong/EMA filter"] = (np.where(bo)[0], bd)
    anyb = valid & (F["dir_bar"] != 0)
    ai = RNG.choice(np.where(anyb)[0], size=min(20000, anyb.sum()), replace=False)
    sets["null: random bar, direction of that bar"] = (ai, F.loc[ai, "dir_bar"].to_numpy(float))

    L = []
    L.append(f"# SRB breakout — event edge test ({pd.Timestamp.today():%Y-%m-%d})\n")
    L.append(f"Data: `{Path(args.h1).name}` (RoboForex MT5, custom symbol USDJPY_TDS, {F['t'].iloc[0]:%Y-%m-%d} → {F['t'].iloc[-1]:%Y-%m-%d}); "
             f"test window from {args.start}; OOS = from {args.oos_from}. Twin: `tools/data/srb_features.py` (v02 rf1 defaults).\n")
    L.append("Units: forward return in **ATR(H1,14)** from the next bar's open, direction-signed, spread deducted; "
             "trade results in **R = 1.5 ATR** (the EA's stop). `z` = mean / standard error over events (events treated as independent; "
             "SRB fires ~1 per 70 bars so overlap at h ≤ 48 is small).\n")

    # ---- 1. forward returns ----------------------------------------------------------
    L.append("## 1. Forward return after the event (ATR units)\n")
    L.append("| event set | n | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
    L.append("|---|---:|" + "---|" * len(HORIZONS))
    fw = {}
    for name, (idx, sgn) in sets.items():
        T = fwd_table(F, idx, sgn); fw[name] = (idx, sgn, T)
        L.append(f"| {name} | {len(idx)} | " + " | ".join(fmt(*mstat(T[h])[:2]) for h in HORIZONS) + " |")
    L.append("")
    # sub-periods for the signal set
    idx, sgn, T = fw["SRB signal (all filters, no state)"]
    yrs = F.loc[idx, "t"].dt.year.to_numpy()
    L.append("Stability of the SRB signal by period (h=4 / h=24 / h=48):\n")
    L.append("| period | n | h=4 | h=24 | h=48 |\n|---|---:|---|---|---|")
    for lo, hi in ((2015, 2018), (2019, 2022), (2023, 2025), (2026, 2026)):
        m = (yrs >= lo) & (yrs <= hi)
        L.append(f"| {lo}–{hi}{' (OOS)' if lo >= 2026 else ''} | {m.sum()} | " + " | ".join(fmt(*mstat(T[h][m])[:2]) for h in (4, 24, 48)) + " |")
    L.append("")

    # ---- 2. exits the RL env can express ---------------------------------------------
    L.append("## 2. Same entries, simple exits (result in R per event; SL always 1R = 1.5 ATR)\n")
    L.append("| exit rule | all n | mean R | z | win% | PF | 2015–2025 mean R | 2026 (OOS) mean R |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    rules = [("market exit after 4 bars, SL 1R", dict(max_bars=4)), ("market exit after 24 bars, SL 1R", dict(max_bars=24)),
             ("TP 1.25R (= o1 leg), no time stop", dict(tp_r=1.25)), ("TP 2R", dict(tp_r=2.0)), ("TP 3R", dict(tp_r=3.0)),
             ("TP 2R + breakeven at 0.5R (≈ runner)", dict(tp_r=2.0, be_at=0.5)), ("TP 3R + breakeven at 0.5R", dict(tp_r=3.0, be_at=0.5))]
    oos_mask = F.loc[idx, "t"].to_numpy() >= np.datetime64(args.oos_from)
    for name, kw in rules:
        r = exit_sim(F, idx, sgn, **kw)
        m, se, n = mstat(r); ok = np.isfinite(r)
        pf = r[ok & (r > 0)].sum() / max(1e-9, -r[ok & (r < 0)].sum())
        L.append(f"| {name} | {n} | {m:+.3f} | {m / se:+.1f} | {np.mean(r[ok] > 0):.0%} | {pf:.2f} | "
                 f"{np.nanmean(r[~oos_mask]):+.3f} | {np.nanmean(r[oos_mask]):+.3f} |")
    L.append("")
    sp_mean = float(np.nanmean(F.loc[idx, "spread_px"])) * 100
    L.append(f"Cost sensitivity (the TDS bars carry a {sp_mean:.2f}-pip spread and the tester charged no commission; "
             f"a live raw-spread account pays about 0.5–1.5 pips round-trip incl. commission):\n")
    L.append("| extra round-trip cost | 4-bar exit mean R | TP 1.25R mean R | TP 2R + BE mean R |\n|---|---:|---:|---:|")
    for pips in (0.0, 0.5, 1.0, 1.5, 2.0):
        COST_SAVE = COST; globals()["COST"] = pips * 0.01
        r4 = exit_sim(F, idx, sgn, max_bars=4); r1 = exit_sim(F, idx, sgn, tp_r=1.25); r2 = exit_sim(F, idx, sgn, tp_r=2.0, be_at=0.5)
        globals()["COST"] = COST_SAVE
        L.append(f"| {pips:.1f} pips | {np.nanmean(r4):+.3f} | {np.nanmean(r1):+.3f} | {np.nanmean(r2):+.3f} |")
    L.append("")

    # ---- 3. what the EA actually got (report) -----------------------------------------
    if args.report_signals:
        rs = pd.read_csv(args.report_signals, parse_dates=["t_sig", "entry_time", "exit_last"])
        rs["sig_r"] = 0.2 * rs["o1_r"].fillna(0) + 0.8 * rs["o2_r"].fillna(0)      # risk-weighted result per signal
        L.append("## 3. What the EA actually did with these entries (tester report, per signal, in R)\n")
        L.append(f"Signals {len(rs)}, o1 = 20% of risk at TP 1.25R, o2 = 80% runner (TP measured move, breakeven at 0.5R). "
                 f"Median hold {((rs['exit_last'] - rs['entry_time']).dt.total_seconds() / 3600).median():.1f} h.\n")
        for leg in ("o1", "o2"):
            r = rs[f"{leg}_r"]; m, se, n = mstat(r)
            L.append(f"- **{leg}**: mean {m:+.3f} R (z {m / se:+.1f}), win {np.mean(r > 0.05):.0%}, breakeven {np.mean(r.abs() <= 0.05):.0%}, "
                     f"loss {np.mean(r < -0.05):.0%}, PF {r[r > 0].sum() / -r[r < 0].sum():.2f}")
        m, se, n = mstat(rs["sig_r"])
        L.append(f"- **risk-weighted per signal**: mean {m:+.3f} R (z {m / se:+.1f}), PF {rs['sig_r'][rs['sig_r'] > 0].sum() / -rs['sig_r'][rs['sig_r'] < 0].sum():.2f}")
        L.append("\n| period | n | mean R / signal | z |\n|---|---:|---:|---:|")
        yr = rs["t_sig"].dt.year
        for lo, hi in ((2015, 2018), (2019, 2022), (2023, 2025), (2026, 2026)):
            mm = (yr >= lo) & (yr <= hi); m, se, n = mstat(rs.loc[mm, "sig_r"])
            L.append(f"| {lo}–{hi}{' (OOS)' if lo >= 2026 else ''} | {n} | {m:+.3f} | {m / se:+.1f} |")
        L.append("")

    # ---- 4. do the zone features rank the events? --------------------------------------
    L.append("## 4. Inside the events: do the zone features predict the outcome? (Spearman, event bars only)\n")
    feats = ["res_touches", "sup_touches", "room_up_atr", "room_dn_atr", "dist_res_atr", "dist_sup_atr",
             "candle_range_atr", "close_loc", "ema_dist_atr", "zone_tol_atr", "n_res", "n_sup"]
    E = F.loc[idx].copy(); E["sgn"] = sgn
    # direction-aware: for a buy the relevant zone is the broken resistance (room_up), for a sell the support
    E["touches_broken"] = np.where(sgn > 0, E["res_touches"], E["sup_touches"])
    E["room_ahead_atr"] = np.where(sgn > 0, E["room_up_atr"], E["room_dn_atr"])
    E["ema_dist_signed"] = E["ema_dist_atr"] * sgn
    E["close_loc_signed"] = np.where(sgn > 0, E["close_loc"], 1 - E["close_loc"])
    r24 = exit_sim(F, idx, sgn, max_bars=24); r_tp = exit_sim(F, idx, sgn, tp_r=2.0, be_at=0.5)
    L.append("| feature at the signal bar | n | ρ vs fwd h=8 | ρ vs fwd h=24 | ρ vs R (TP2R+BE) |\n|---|---:|---:|---:|---:|")
    T = fw["SRB signal (all filters, no state)"][2]
    for f in ("touches_broken", "room_ahead_atr", "candle_range_atr", "close_loc_signed", "ema_dist_signed", "zone_tol_atr", "n_res", "n_sup"):
        x = E[f].to_numpy(float)
        cells = []
        for y in (T[8].to_numpy(), T[24].to_numpy(), r_tp):
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 30: cells.append("–"); continue
            rho = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"); z = rho * np.sqrt(ok.sum() - 2)
            cells.append(f"{rho:+.3f} (z {z:+.1f})")
        L.append(f"| {f} | {int(np.isfinite(x).sum())} | " + " | ".join(cells) + " |")
    L.append("\nρ is the rank correlation between the feature and the outcome across events; |z| ≥ 3 is the bar used in `feature_screen.py`.\n")

    # ---- 5. how rare the event is (why unconditional IC cannot see it) ----------------
    nbars = int(valid.sum()); nev = len(idx)
    L.append("## 5. Why the 183-feature screen could not have found this\n")
    L.append(f"The SRB condition is true on {nev} of {nbars} H1 bars ({nev / nbars:.2%}). A flag that is 0 on {1 - nev / nbars:.1%} of bars has "
             "a whole-slice rank correlation with forward return of at most a few hundredths even when the conditional mean is large, "
             "so `feature_screen.py` (unconditional IC) is the wrong instrument for event features; this file is the conditional test.\n")
    out = Path(args.out) if args.out else ROOT / "docs" / f"srb_event_edge_{pd.Timestamp.today():%Y-%m-%d}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    E.assign(fwd8=T[8].to_numpy(), fwd24=T[24].to_numpy(), r24=r24, r_tp2_be=r_tp).to_csv(out.with_suffix(".events.csv"), index=False)
    print("\n".join(L))
    print(f"\nwrote {out} and {out.with_suffix('.events.csv')}")


if __name__ == "__main__":
    main()
