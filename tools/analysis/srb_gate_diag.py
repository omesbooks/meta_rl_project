"""Gated arms: what did the agent accept, how did it exit, and does accepting have skill
(accepted vs skipped signals under the same fixed 4-bar rule)?"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\omesb\Documents\claude code\pycaret_trainer")
sys.path.insert(0, str(ROOT / "tools" / "analysis")); sys.path.insert(0, str(ROOT / "tools" / "data"))
import srb_event_edge as E
from srb_features import atr_mt5

prefixes = sys.argv[1:] or ["h1gate"]
f = pd.read_csv(ROOT / "usdjpy_tds_h1_srb.csv"); f["t"] = pd.to_datetime(f["timestamp"], format="%Y.%m.%d %H:%M")
n = len(f); te = int(n * 0.85)
F = f[["t", "open", "high", "low", "close"]].copy(); F["atr"] = atr_mt5(F["high"], F["low"], F["close"], 14); F["spread_px"] = 0.0
E.COST = 0.01
sig_idx = np.where((f.index >= te) & (f["srb_sig"] != 0))[0]; sgn_all = f.loc[sig_idx, "srb_sig"].to_numpy(float)
r4_all = E.exit_sim(F, sig_idx, sgn_all, max_bars=4, sl_r=1e9)          # fixed rule, no stop, 1 pip
Rpx = 1.5 * F["atr"].to_numpy()[sig_idx]
t_sig = f.loc[sig_idx, "t"].to_numpy()
pf = lambda x: x[x > 0].sum() / max(1e-9, -x[x < 0].sum())
print(f"Test slice: {len(sig_idx)} signals | every signal, 4-bar exit: PF {pf(r4_all):.2f}, net {np.nansum(r4_all * Rpx) * 100:+.0f} pips")
for pre in prefixes:
    for sd in (0, 1, 2):
        fs = glob.glob(str(ROOT / "artifacts" / "models" / f"{pre}_C_w15_t100_s{sd}" / "runs" / "*" / "backtests" / "*trades.csv"))
        if not fs: continue
        tr = pd.read_csv(fs[0]); ot = pd.to_datetime(tr["open_time"]).dt.floor("h").to_numpy()
        # the agent acts at the close of the signal bar -> open_time == signal bar time
        acc = np.isin(t_sig, ot)
        side = np.where(tr["side"].str.lower().str.startswith("l"), 1, -1)
        px = (tr["exit"] - tr["entry"]).to_numpy() * side * 100          # pips before spread (spread already in prices)
        line = (f"{pre} s{sd}: trades {len(tr):>3} ({acc.mean():.0%} of signals) | PF {pf(tr['pnl_dollars'].to_numpy()):.2f} | "
                f"net {px.sum():+.0f} pips | hold {tr['bars_held'].mean():.1f} bars | exits {tr['reason'].value_counts().to_dict()}")
        if "sl_atr" in tr and tr["sl_atr"].notna().any():
            line += f" | SL mean {tr['sl_atr'].mean():.2f} sd {tr['sl_atr'].std():.2f}"
        print(line)
        a, s = r4_all[acc], r4_all[~acc]
        print(f"      fixed 4-bar rule on ACCEPTED signals: PF {pf(a):.2f} mean R {np.nanmean(a):+.3f} (n={acc.sum()}) | on SKIPPED: PF {pf(s):.2f} mean R {np.nanmean(s):+.3f} (n={(~acc).sum()})")
