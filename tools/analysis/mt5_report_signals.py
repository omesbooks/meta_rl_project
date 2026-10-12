"""MT5 Strategy Tester HTML report (UTF-16) -> deals / orders CSV + a per-signal table.

    python tools/analysis/mt5_report_signals.py "<report>.html" --out docs/<name>

Writes <out>_deals.csv, <out>_orders.csv and <out>_signals.csv. The signal table pairs
every entry ("in" deals at one bar) with the exits that follow it, so each signal gets
its entry price, original SL/TP (from the Orders table), R = |entry - SL| and the result
of every leg in R. It assumes one position at a time (the SRB EA's InpOneAtATime) and
leg comments like `SRB_o1` / `SRB_o2`; other EAs get the entry/exit pairing only.
"""
from __future__ import annotations

import argparse
import html
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DEAL_COLS = ["time", "deal", "symbol", "type", "direction", "volume", "price", "order", "commission", "swap", "profit", "balance", "comment"]
ORDER_COLS = ["open_time", "order", "symbol", "type", "volume", "price", "sl", "tp", "time", "state", "comment"]


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace("\xa0", "").str.replace(" ", "").str.split("/").str[0], errors="coerce")


def parse(path):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
    text = text.replace("\r", "")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I)
    cells = lambda r: [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                       for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S | re.I)]
    section = None; deals = []; orders = []; settings = {}
    for c in map(cells, rows):
        if not c: continue
        if c[0] == "Orders": section = "orders"; continue
        if c[0] == "Deals": section = "deals"; continue
        dated = re.match(r"\d{4}\.\d\d\.\d\d", c[0]) is not None
        if section == "orders" and dated and len(c) >= 10: orders.append(c[:11] + [""] * (11 - len(c[:11])))
        elif section == "deals" and dated and len(c) >= 10: deals.append(c[:13] + [""] * (13 - len(c[:13])))
        elif section is None:
            for i in range(0, len(c) - 1, 2):
                if c[i].endswith(":") and c[i + 1]: settings[c[i][:-1]] = c[i + 1]
    d = pd.DataFrame(deals, columns=DEAL_COLS); o = pd.DataFrame(orders, columns=ORDER_COLS)
    for col in ("price", "volume", "profit", "commission", "swap"): d[col] = _num(d[col])
    for col in ("sl", "tp", "price"): o[col] = _num(o[col])
    d["time"] = pd.to_datetime(d["time"], format="%Y.%m.%d %H:%M:%S"); o["order"] = o["order"].astype(str)
    return settings, d, o


def signals(deals, orders, bar="h"):
    ins = deals[deals["direction"] == "in"].copy()
    ins["leg"] = ins["comment"].str.extract(r"_(o\d)")[0].fillna("o1")
    outs = deals[deals["direction"] == "out"]
    times = sorted(ins["time"].dt.floor(bar).unique()); rows = []
    for k, t in enumerate(times):
        nxt = times[k + 1] if k + 1 < len(times) else pd.Timestamp("2100-01-01")
        legs = ins[ins["time"].dt.floor(bar) == t]; ex = outs[(outs["time"] >= t) & (outs["time"] < nxt)]
        dr = legs["type"].iloc[0]; sgn = 1 if dr == "buy" else -1; entry = float(legs["price"].iloc[0])
        orow = orders[orders["order"] == str(legs["order"].iloc[0])]
        sl = float(orow["sl"].iloc[0]) if len(orow) else np.nan; R = abs(entry - sl)
        r = {"t_sig": t - pd.Timedelta(1, unit=bar), "entry_time": t, "dir": dr, "entry": entry, "sl": sl, "R": R,
             "n_in": len(legs), "exit_last": ex["time"].max() if len(ex) else pd.NaT, "net": float(ex["profit"].sum())}
        for leg, lg in legs.groupby("leg"):
            v = float(lg["volume"].iloc[0]); e = ex[np.isclose(ex["volume"], v)]
            lrow = orders[orders["order"] == str(lg["order"].iloc[0])]
            r[f"{leg}_tp"] = float(lrow["tp"].iloc[0]) if len(lrow) else np.nan
            if len(e):
                e = e.iloc[0]
                r[f"{leg}_exit"] = e["time"]; r[f"{leg}_how"] = str(e["comment"]).split(" ")[0]
                r[f"{leg}_r"] = sgn * (e["price"] - entry) / R if R > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report"); ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--bar", default="h", help="trading-TF bar for grouping legs (pandas offset, default h)")
    a = ap.parse_args()
    settings, d, o = parse(a.report)
    for k in ("Expert", "Symbol", "Period", "Company", "Initial Deposit", "Total Net Profit", "Profit Factor", "Total Trades"):
        if k in settings: print(f"{k}: {settings[k]}")
    out = Path(a.out)
    d.to_csv(f"{out}_deals.csv", index=False); o.to_csv(f"{out}_orders.csv", index=False)
    s = signals(d, o, a.bar); s.to_csv(f"{out}_signals.csv", index=False)
    print(f"deals={len(d)} orders={len(o)} signals={len(s)} -> {out}_signals.csv")


if __name__ == "__main__":
    main()
