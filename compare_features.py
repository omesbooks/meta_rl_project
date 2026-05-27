"""
compare_features.py — MT5 live features vs training CSV (parity check)

Parses [FEATROW] lines dumped by rl_gbpusd_EA.mq5 (InpDebugFeatures=true) from the
MT5 tester agent log, matches each timestamp against the training CSV, and reports
per-feature differences. Big diffs = feature-computation mismatch (root cause of
"trades only during warmup, then Hold forever" in MT5).

Usage:
    python compare_features.py
    python compare_features.py --csv training_data_v4_GBPUSD.csv --norm rl_gbpusd_norm.csv
    python compare_features.py --log "C:\\path\\to\\agent\\20260527.log"
"""
import sys, io, re, glob, os, argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_AGENT_GLOB = (
    r"C:\Users\omesb\AppData\Roaming\MetaQuotes\Tester"
    r"\*\Agent-*\logs\*.log"
)


def find_latest_log():
    files = glob.glob(DEFAULT_AGENT_GLOB)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def read_log_text(path):
    raw = Path(path).read_bytes()
    # MT5 logs are UTF-16LE; strip nulls as a robust fallback
    for enc in ("utf-16-le", "utf-16", "utf-8"):
        try:
            t = raw.decode(enc, errors="ignore")
            if "FEATROW" in t or "FEATDUMP" in t:
                return t
        except Exception:
            pass
    return raw.replace(b"\x00", b"").decode("utf-8", errors="ignore")


def parse_dumps(text):
    """Return list of dicts: {ts, raw[list], act, conf, probs[list]}"""
    rows = {}
    # [FEATROW] 2004.01.15 12:00 | v0,v1,...
    for m in re.finditer(r"\[FEATROW\]\s+([\d.]+\s+[\d:]+)\s*\|\s*([0-9eE.,\-+]+)", text):
        ts = m.group(1).strip()
        vals = [float(x) for x in m.group(2).split(",") if x != ""]
        rows.setdefault(ts, {})["raw"] = vals
    # [FEATDUMP] 2004.01.15 12:00 act=0 conf=0.71 probs=...
    for m in re.finditer(
        r"\[FEATDUMP\]\s+([\d.]+\s+[\d:]+)\s+act=(\d+)\s+conf=([\d.]+)\s+probs=([0-9.,]+)",
        text,
    ):
        ts = m.group(1).strip()
        d = rows.setdefault(ts, {})
        d["act"] = int(m.group(2))
        d["conf"] = float(m.group(3))
        d["probs"] = [float(x) for x in m.group(4).split(",") if x != ""]
    return rows


def mt5_ts_to_pd(ts):
    # "2004.01.15 12:00" -> Timestamp
    return pd.to_datetime(ts.replace(".", "-"), errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_data_v4_GBPUSD.csv")
    ap.add_argument("--norm", default="rl_gbpusd_norm.csv",
                    help="defines feature order (index)")
    ap.add_argument("--log", default=None, help="MT5 agent log (auto-detect if omitted)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="relative-diff threshold to flag a feature (default 5%)")
    args = ap.parse_args()

    log = args.log or find_latest_log()
    if not log or not Path(log).exists():
        print("ERROR: agent log not found. Pass --log <path>.")
        return 1
    print(f"[log]  {log}")

    feat_cols = list(pd.read_csv(args.norm, index_col=0).index)
    print(f"[norm] {len(feat_cols)} features (model order)")

    text = read_log_text(log)
    dumps = parse_dumps(text)
    dumps = {ts: d for ts, d in dumps.items() if "raw" in d}
    if not dumps:
        print("ERROR: no [FEATROW] lines found. Did you run with InpDebugFeatures=true?")
        return 1
    print(f"[dump] {len(dumps)} bars dumped from MT5\n")

    csv = pd.read_csv(args.csv)
    csv["timestamp"] = pd.to_datetime(csv["timestamp"], errors="coerce")
    csv = csv.set_index("timestamp")

    # Aggregate per-feature diff stats across all matched bars
    feat_abs = {c: [] for c in feat_cols}
    feat_rel = {c: [] for c in feat_cols}
    matched = 0

    for ts, d in sorted(dumps.items()):
        pts = mt5_ts_to_pd(ts)
        if pts not in csv.index:
            continue
        matched += 1
        csv_row = csv.loc[pts]
        if isinstance(csv_row, pd.DataFrame):
            csv_row = csv_row.iloc[0]
        mt5_vals = d["raw"]
        if len(mt5_vals) != len(feat_cols):
            print(f"[warn] {ts}: dumped {len(mt5_vals)} vals != {len(feat_cols)} features")
            continue
        for i, c in enumerate(feat_cols):
            cval = float(csv_row.get(c, np.nan))
            mval = mt5_vals[i]
            if np.isnan(cval):
                continue
            feat_abs[c].append(abs(mval - cval))
            denom = max(abs(cval), 1e-9)
            feat_rel[c].append(abs(mval - cval) / denom)

    if matched == 0:
        print("ERROR: no dumped timestamps matched CSV rows.")
        print("  Dumped sample:", list(dumps.keys())[:3])
        print("  CSV range:", csv.index.min(), "->", csv.index.max())
        return 1

    print(f"[match] {matched} bars matched CSV\n")
    print(f"{'feature':<22} {'mean_abs':>12} {'mean_rel%':>10}  flag")
    print("-" * 52)
    flagged = []
    for c in feat_cols:
        if not feat_rel[c]:
            continue
        ma = np.mean(feat_abs[c])
        mr = np.mean(feat_rel[c]) * 100
        flag = "  <-- MISMATCH" if mr > args.tol * 100 else ""
        if flag:
            flagged.append((c, mr))
        print(f"{c:<22} {ma:>12.6f} {mr:>9.1f}% {flag}")

    print("\n" + "=" * 52)
    if flagged:
        print(f"⚠️  {len(flagged)}/{len(feat_cols)} features differ > {args.tol*100:.0f}%:")
        for c, mr in sorted(flagged, key=lambda x: -x[1]):
            print(f"   {c}: {mr:.1f}%")
        print("\n→ Feature MISMATCH confirmed. Fix RL_Indicators.mqh to match")
        print("  feature_engineer.py / DataCollector_v4 for these features.")
    else:
        print("✅ All features within tolerance — features MATCH.")
        print("→ No-trade cause is elsewhere (normalization / session / logic).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
