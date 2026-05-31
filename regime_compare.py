"""
Regime Detection — Method Comparison

Runs 6 different mathematical methods to detect regime breaks in GBP/USD H4
data, then prints a side-by-side comparison table.

Methods:
  1. Naive year grouping (baseline)
  2. Rolling t-test mean-shift
  3. PELT changepoint detection (ruptures)
  4. Binary Segmentation (ruptures)
  5. HMM 3-state regime model (hmmlearn)
  6. K-Means clustering on rolling features

Outputs:
  - Breakpoints found by each method
  - Match score vs known events (Lehman, Brexit, Truss)
  - HTML chart with all methods overlaid
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Known regime-changing events (ground truth)
KNOWN_EVENTS = {
    "Lehman crash": pd.Timestamp("2008-09-15"),
    "Brexit referendum": pd.Timestamp("2016-06-23"),
    "Truss mini-budget": pd.Timestamp("2022-09-26"),
}
MATCH_TOLERANCE_DAYS = 90  # ภายใน 3 เดือน = match


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Resample to daily for speed
    df["date"] = df.timestamp.dt.date
    daily = df.groupby("date").close.last().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["log_ret"] = np.log(daily.close).diff().fillna(0)
    return daily


# -------------------- METHODS --------------------

def method_naive_yearly(daily: pd.DataFrame) -> list[pd.Timestamp]:
    """Find years where yearly mean changes > 10% from previous year."""
    yr = daily.groupby(daily.date.dt.year).close.mean()
    breaks = []
    for i in range(1, len(yr)):
        change = abs(yr.iloc[i] - yr.iloc[i - 1]) / yr.iloc[i - 1]
        if change > 0.10:
            year = yr.index[i]
            breaks.append(pd.Timestamp(f"{year}-01-01"))
    return breaks


def method_rolling_ttest(daily: pd.DataFrame, win: int = 180,
                         t_thresh: float = 25.0) -> list[pd.Timestamp]:
    """Welch t-test for mean shift over rolling window. Pick top peaks."""
    from scipy.signal import find_peaks
    prices = daily.close.values
    n = len(prices)
    ts = np.zeros(n)
    for i in range(win, n - win):
        left = prices[i - win : i]
        right = prices[i : i + win]
        t = (right.mean() - left.mean()) / np.sqrt(
            right.var(ddof=1) / win + left.var(ddof=1) / win
        )
        ts[i] = abs(t)
    peaks, _ = find_peaks(ts, height=t_thresh, distance=365)
    return [daily.date.iloc[p] for p in peaks]


def method_pelt(daily: pd.DataFrame, pen: float = 50.0) -> list[pd.Timestamp]:
    """PELT (Pruned Exact Linear Time) changepoint detection."""
    import ruptures as rpt
    signal = daily.close.values.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    bkps = algo.predict(pen=pen)
    # ruptures returns end indices; last is len(signal)
    return [daily.date.iloc[b - 1] for b in bkps[:-1]]


def method_binseg(daily: pd.DataFrame, n_bkps: int = 3) -> list[pd.Timestamp]:
    """Binary Segmentation — find n_bkps best splits."""
    import ruptures as rpt
    signal = daily.close.values.reshape(-1, 1)
    algo = rpt.Binseg(model="l2").fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)
    return [daily.date.iloc[b - 1] for b in bkps[:-1]]


def method_hmm(daily: pd.DataFrame, n_states: int = 3) -> tuple[list[pd.Timestamp], np.ndarray]:
    """Hidden Markov Model — assign each bar to one of K hidden regimes."""
    from hmmlearn import hmm
    # Use [log_return, rolling_std] as observation
    obs = np.column_stack([
        daily.log_ret.values,
        daily.close.pct_change().rolling(20).std().fillna(0).values,
    ])
    model = hmm.GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=100, random_state=42,
    )
    model.fit(obs)
    states = model.predict(obs)
    # Find regime transitions
    breaks = []
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            breaks.append(daily.date.iloc[i])
    # Filter: keep only transitions that persist >= 30 days (no flapping)
    persistent = []
    for b in breaks:
        idx = daily.date.searchsorted(b)
        if idx + 30 < len(states):
            if (states[idx : idx + 30] == states[idx]).sum() >= 25:
                persistent.append(b)
    return persistent, states


def method_kmeans(daily: pd.DataFrame, k: int = 3, win: int = 60) -> list[pd.Timestamp]:
    """K-Means on rolling [mean, std, slope] features."""
    from sklearn.cluster import KMeans
    rolling = pd.DataFrame({
        "mean": daily.close.rolling(win).mean(),
        "std": daily.close.rolling(win).std(),
        "slope": daily.close.diff(win) / win,
    }).bfill().fillna(0)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rolling.values)
    breaks = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            breaks.append(daily.date.iloc[i])
    # Filter persistent
    persistent = []
    for b in breaks:
        idx = daily.date.searchsorted(b)
        if idx + 30 < len(labels):
            if (labels[idx : idx + 30] == labels[idx]).sum() >= 25:
                persistent.append(b)
    return persistent


# -------------------- EVALUATION --------------------

def match_known(breaks: list[pd.Timestamp]) -> dict:
    """For each known event, find nearest detected break (within tolerance)."""
    matches = {}
    for name, event in KNOWN_EVENTS.items():
        nearest = None
        min_diff = float("inf")
        for b in breaks:
            d = abs((b - event).days)
            if d < min_diff:
                min_diff = d
                nearest = b
        if nearest is not None and min_diff <= MATCH_TOLERANCE_DAYS:
            matches[name] = (nearest, min_diff)
        else:
            matches[name] = (None, None)
    return matches


def run_all(csv_path: str) -> dict:
    daily = load_data(csv_path)
    print(f"[load] {len(daily):,} daily bars · {daily.date.min().date()} → {daily.date.max().date()}")

    results = {}

    print("\n[1/6] Naive yearly grouping ...")
    t0 = time.time()
    results["1. Naive yearly (>10% jump)"] = {
        "breaks": method_naive_yearly(daily),
        "time": time.time() - t0,
    }

    print("[2/6] Rolling t-test ...")
    t0 = time.time()
    results["2. Rolling t-test (win=180)"] = {
        "breaks": method_rolling_ttest(daily),
        "time": time.time() - t0,
    }

    print("[3/6] PELT (ruptures) ...")
    t0 = time.time()
    results["3. PELT (rbf, pen=50)"] = {
        "breaks": method_pelt(daily, pen=50),
        "time": time.time() - t0,
    }

    print("[4/6] Binary Segmentation ...")
    t0 = time.time()
    results["4. BinSeg (n_bkps=3)"] = {
        "breaks": method_binseg(daily, n_bkps=3),
        "time": time.time() - t0,
    }

    print("[5/6] HMM 3-state ...")
    t0 = time.time()
    hmm_breaks, hmm_states = method_hmm(daily, n_states=3)
    results["5. HMM 3-state"] = {
        "breaks": hmm_breaks,
        "time": time.time() - t0,
        "states": hmm_states,
    }

    print("[6/6] K-Means on rolling features ...")
    t0 = time.time()
    results["6. K-Means rolling (k=3)"] = {
        "breaks": method_kmeans(daily, k=3),
        "time": time.time() - t0,
    }

    return daily, results


def print_table(results: dict):
    """Pretty side-by-side comparison."""
    print("\n" + "=" * 100)
    print("REGIME DETECTION — METHOD COMPARISON")
    print("=" * 100)
    header = f"{'Method':<32} {'#Breaks':<9} {'Lehman':<14} {'Brexit':<14} {'Truss':<14} {'Time(s)':<8} {'Score':<6}"
    print(header)
    print("-" * 100)

    for method, info in results.items():
        breaks = info["breaks"]
        matches = match_known(breaks)
        score = sum(1 for v in matches.values() if v[0] is not None)
        lehman = matches["Lehman crash"][0]
        brexit = matches["Brexit referendum"][0]
        truss = matches["Truss mini-budget"][0]

        def fmt(d):
            if d is None:
                return "MISS"
            return d.strftime("%Y-%m-%d")

        print(f"{method:<32} {len(breaks):<9} {fmt(lehman):<14} {fmt(brexit):<14} {fmt(truss):<14} {info['time']:<8.2f} {score}/3")

    print("=" * 100)
    print("(MATCH = detected break within ±90 days of known event)\n")


def print_breakdowns(results: dict):
    """Show full breakpoint list per method."""
    print("\n--- FULL BREAKPOINT LISTS ---\n")
    for method, info in results.items():
        breaks = info["breaks"]
        print(f"  {method}  ({len(breaks)} breaks):")
        if not breaks:
            print("    (none)")
            continue
        for b in breaks:
            tag = ""
            for name, ev in KNOWN_EVENTS.items():
                if abs((b - ev).days) <= MATCH_TOLERANCE_DAYS:
                    tag = f"  <- {name}"
                    break
            print(f"    {b.strftime('%Y-%m-%d')}{tag}")
        print()


def write_html_chart(daily: pd.DataFrame, results: dict, out_path: str):
    """Write an HTML chart overlaying all methods' breakpoints."""
    import json
    labels = [d.strftime("%Y-%m-%d") for d in daily.date]
    closes = [round(float(v), 5) for v in daily.close]

    # Each method's breaks as JS array
    method_breaks = {}
    for method, info in results.items():
        method_breaks[method] = [b.strftime("%Y-%m-%d") for b in info["breaks"]]

    payload = {
        "labels": labels,
        "closes": closes,
        "methods": method_breaks,
        "events": {n: e.strftime("%Y-%m-%d") for n, e in KNOWN_EVENTS.items()},
    }
    Path("regime_compare_data.json").write_text(json.dumps(payload))

    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#ff7b72"]
    method_blocks = []
    for i, (method, info) in enumerate(results.items()):
        color = colors[i % len(colors)]
        n = len(info["breaks"])
        method_blocks.append(
            f'<div class="m"><span class="dot" style="background:{color}"></span>'
            f'<b>{method}</b> <span class="muted">({n} breaks, {info["time"]:.2f}s)</span></div>'
        )

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Regime Detection — Method Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;margin:0}
h1{color:#58a6ff;margin:0 0 8px}
.sub{color:#8b949e;margin-bottom:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin-bottom:16px}
.m{margin:6px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}
.muted{color:#8b949e;font-size:12px}
canvas{max-height:560px}
</style></head><body>
<h1>Regime Detection — 6 Methods Compared</h1>
<p class="sub">GBP/USD daily close · vertical lines = each method's detected breakpoints · known events marked in white</p>
<div class="card">METHODS_HERE</div>
<div class="card"><canvas id="c"></canvas></div>
<script>
fetch('regime_compare_data.json').then(r=>r.json()).then(d=>{
  const colors = ['#58a6ff','#3fb950','#d29922','#f85149','#a371f7','#ff7b72'];
  const ann = {};
  // Known events: white vertical lines
  Object.entries(d.events).forEach(([name, date], i) => {
    ann['ev_'+i] = {type:'line', xMin:date, xMax:date, borderColor:'#ffffff', borderWidth:2,
      label:{content:name, display:true, position:i%2===0?'start':'end', color:'#ffffff', backgroundColor:'rgba(0,0,0,0.7)', font:{size:11, weight:'bold'}}};
  });
  // Each method's breaks
  let yOffset = 0;
  Object.entries(d.methods).forEach(([method, breaks], mi) => {
    const c = colors[mi % colors.length];
    breaks.forEach((b, bi) => {
      ann['m'+mi+'_'+bi] = {type:'line', xMin:b, xMax:b, borderColor:c, borderWidth:1, borderDash:[4,3]};
    });
  });
  new Chart(document.getElementById('c').getContext('2d'), {
    type:'line',
    data:{labels:d.labels, datasets:[{label:'GBP/USD', data:d.closes, borderColor:'#c9d1d9', borderWidth:1, pointRadius:0, fill:false}]},
    options:{responsive:true, maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#c9d1d9'}}, annotation:{annotations:ann}},
      scales:{x:{type:'time', time:{unit:'year'}, ticks:{color:'#8b949e'}, grid:{color:'#30363d'}},
              y:{ticks:{color:'#8b949e'}, grid:{color:'#30363d'}}}}
  });
});
</script></body></html>"""
    html = html.replace("METHODS_HERE", "\n".join(method_blocks))
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"\n[chart] saved -> {out_path}")


# -------------------- SINGLE-METHOD ENTRY (for GUI) --------------------

def run_single(csv_path: str, method: str, **params) -> dict:
    """Run ONE method and return structured result. Used by the GUI."""
    daily = load_data(csv_path)
    t0 = time.time()

    if method == "hmm":
        breaks, states = method_hmm(daily, n_states=params.get("n_states", 3))
        info = {"breaks": breaks, "time": time.time() - t0, "states": states.tolist()}
    elif method == "kmeans":
        breaks = method_kmeans(daily, k=params.get("k", 3),
                                       win=params.get("win", 60))
        info = {"breaks": breaks, "time": time.time() - t0}
    elif method == "pelt":
        breaks = method_pelt(daily, pen=params.get("penalty", 50.0))
        info = {"breaks": breaks, "time": time.time() - t0}
    else:
        raise ValueError(f"Unknown method: {method}")

    matches = match_known(info["breaks"])
    return {
        "method": method,
        "csv": csv_path,
        "n_bars": len(daily),
        "date_range": [str(daily.date.min().date()), str(daily.date.max().date())],
        "n_breaks": len(info["breaks"]),
        "breaks": [b.strftime("%Y-%m-%d") for b in info["breaks"]],
        "matches": {k: (v[0].strftime("%Y-%m-%d") if v[0] else None) for k, v in matches.items()},
        "score": sum(1 for v in matches.values() if v[0] is not None),
        "time_sec": round(info["time"], 3),
        "daily": daily,
        "info": info,
    }


def write_single_chart(result: dict, out_path: str):
    """HTML chart for a single method's result."""
    import json
    daily = result["daily"]
    breaks = result["breaks"]
    method = result["method"]

    labels = [d.strftime("%Y-%m-%d") for d in daily.date]
    closes = [round(float(v), 5) for v in daily.close]

    payload = {
        "labels": labels,
        "closes": closes,
        "breaks": breaks,
        "events": {n: e.strftime("%Y-%m-%d") for n, e in KNOWN_EVENTS.items()},
        "method": method,
        "score": result["score"],
        "n_breaks": result["n_breaks"],
        "time_sec": result["time_sec"],
        "n_bars": result["n_bars"],
    }
    Path("regime_single_data.json").write_text(json.dumps(payload))

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Regime Detection - METHOD_NAME</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;margin:0}
h1{color:#58a6ff;margin:0 0 8px}
.sub{color:#8b949e;margin-bottom:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin-bottom:16px}
.stat{display:inline-block;margin-right:24px}.stat b{display:block;font-size:24px;color:#58a6ff}.stat span{color:#8b949e;font-size:12px}
canvas{max-height:560px}
</style></head><body>
<h1>Regime Detection - METHOD_NAME</h1>
<p class="sub">GBP/USD daily close - red dashed lines = detected breakpoints, white solid = known events</p>
<div class="card">
  <div class="stat"><b>N_BREAKS</b><span>breakpoints detected</span></div>
  <div class="stat"><b>SCORE</b><span>known events matched</span></div>
  <div class="stat"><b>TIME_SECs</b><span>computation time</span></div>
  <div class="stat"><b>N_BARS</b><span>daily bars analyzed</span></div>
</div>
<div class="card"><canvas id="c"></canvas></div>
<script>
fetch('regime_single_data.json').then(r=>r.json()).then(d=>{
  const ann = {};
  Object.entries(d.events).forEach(([name, date], i) => {
    ann['ev_'+i] = {type:'line', xMin:date, xMax:date, borderColor:'#ffffff', borderWidth:2,
      label:{content:name, display:true, position:i%2===0?'start':'end', color:'#ffffff', backgroundColor:'rgba(0,0,0,0.7)', font:{size:11, weight:'bold'}}};
  });
  d.breaks.forEach((b, bi) => {
    ann['b_'+bi] = {type:'line', xMin:b, xMax:b, borderColor:'#f85149', borderWidth:2, borderDash:[5,3]};
  });
  new Chart(document.getElementById('c').getContext('2d'), {
    type:'line',
    data:{labels:d.labels, datasets:[{label:'GBP/USD', data:d.closes, borderColor:'#c9d1d9', borderWidth:1, pointRadius:0, fill:false}]},
    options:{responsive:true, maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#c9d1d9'}}, annotation:{annotations:ann}},
      scales:{x:{type:'time', time:{unit:'year'}, ticks:{color:'#8b949e'}, grid:{color:'#30363d'}},
              y:{ticks:{color:'#8b949e'}, grid:{color:'#30363d'}}}}
  });
});
</script></body></html>"""
    html = (html
        .replace("METHOD_NAME", method.upper())
        .replace("N_BREAKS", str(result["n_breaks"]))
        .replace("SCORE", f"{result['score']}/3")
        .replace("TIME_SEC", f"{result['time_sec']:.2f}")
        .replace("N_BARS", f"{result['n_bars']:,}"))
    Path(out_path).write_text(html, encoding="utf-8")


# -------------------- CLI --------------------

if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(
        description="Regime detection — single method or all methods compared.")
    ap.add_argument("csv", nargs="?", default="gu_h4_dataset.csv",
                    help="CSV with timestamp + close columns")
    ap.add_argument("--method", choices=["all", "hmm", "kmeans", "pelt"], default="all",
                    help="all = run all 6 methods comparison; otherwise run one")
    # Method-specific params
    ap.add_argument("--n-states", type=int, default=3, help="HMM number of states")
    ap.add_argument("--k", type=int, default=3, help="K-Means number of clusters")
    ap.add_argument("--win", type=int, default=60, help="K-Means rolling window (days)")
    ap.add_argument("--penalty", type=float, default=50.0, help="PELT penalty (higher = fewer breaks)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of pretty text")
    ap.add_argument("--chart", default="", help="write HTML chart to this path")
    args = ap.parse_args()

    if args.method == "all":
        # Original behavior: compare all 6 methods
        daily, results = run_all(args.csv)
        print_table(results)
        print_breakdowns(results)
        write_html_chart(daily, results, args.chart or "regime_compare.html")
    else:
        # Single-method mode (used by GUI)
        params = {}
        if args.method == "hmm":      params = {"n_states": args.n_states}
        elif args.method == "kmeans": params = {"k": args.k, "win": args.win}
        elif args.method == "pelt":   params = {"penalty": args.penalty}

        print(f"[load] {args.csv}", flush=True)
        result = run_single(args.csv, args.method, **params)

        chart_path = args.chart or "regime_single.html"
        write_single_chart(result, chart_path)

        # Remove non-serializable bits before printing JSON
        serializable = {k: v for k, v in result.items() if k not in ("daily", "info")}
        if args.json:
            print(json.dumps(serializable, indent=2), flush=True)
        else:
            print(f"\n=== {args.method.upper()} Regime Detection ===")
            print(f"  CSV:        {result['csv']}")
            print(f"  Bars:       {result['n_bars']:,} daily")
            print(f"  Range:      {result['date_range'][0]} -> {result['date_range'][1]}")
            print(f"  Time:       {result['time_sec']:.2f}s")
            print(f"  Breaks:     {result['n_breaks']}")
            print(f"  Match:      {result['score']}/3 known events")
            print()
            print("  Breakpoints:")
            for b in result["breaks"]:
                tag = ""
                bdt = pd.Timestamp(b)
                for name, ev in KNOWN_EVENTS.items():
                    if abs((bdt - ev).days) <= MATCH_TOLERANCE_DAYS:
                        tag = f"  <- {name} ({abs((bdt-ev).days)}d)"
                        break
                print(f"    {b}{tag}")
            print(f"\n  Chart:  {chart_path}")
