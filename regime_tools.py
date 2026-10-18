"""Regime Check page rules shared by rl_app.py and the web GUI.

Moved verbatim out of RLTradingStudio (_run_regime_detection, _refresh_events_with_gemini,
_regime_event_status_text, _load_regime_results, _use_regime_cutoff, _browse_regime_csv).
Errors are raised as RegimeError(field, message) so each front end can point at the field.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ui_values import number as input_number

ROOT = Path(__file__).parent.resolve()
METHODS = [
    ("hmm", "HMM 3-state ⭐", "เห็น regime ผ่าน vol+return joint dist · ดีสุดจากการทดสอบ (3/3 events) · ~3s"),
    ("kmeans", "K-Means rolling", "Cluster บน rolling [mean, std, slope] · เร็วสุด ~60ms · 2/3 events"),
    ("pelt", "PELT (changepoint)", "Optimal segmentation ด้วย RBF kernel · 2/3 events · ช้า ~3 นาที"),
]
DEFAULTS = {"method": "hmm", "hmm_states": "3", "km_k": "3", "km_win": "60", "pelt_pen": "50", "ev_symbol": "GBPUSD", "ev_topk": "15"}


class RegimeError(ValueError):
    def __init__(self, field, message, title="Invalid params"):
        super().__init__(message)
        self.field, self.message, self.title = field, message, title


def csv_meta(path):
    """'N rows · start → end' for the dataset header."""
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=lambda c: c.lower() in ("timestamp", "close"))
        n = len(df)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            return f"{n:,} rows · {ts.min().date()} → {ts.max().date()}"
        return f"{n:,} rows · (no timestamp col?)"
    except Exception as e:
        return f"(error reading: {e})"


def detect_command(csv_path, method, params, python=None):
    python = python or sys.executable
    if not csv_path:
        raise RegimeError("regime_csv", "Please pick a CSV first.")
    cmd = [python, "regime_compare.py", str(csv_path), "--method", method]
    try:
        if method == "hmm":
            states = input_number(params.get("hmm_states", ""), "HMM states", 2, integer=True)
            cmd += ["--n-states", str(states)]
        elif method == "kmeans":
            k = input_number(params.get("km_k", ""), "K-means states", 2, integer=True)
            win = input_number(params.get("km_win", ""), "Regime window", 2, integer=True)
            cmd += ["--k", str(k), "--win", str(win)]
        elif method == "pelt":
            pen = input_number(params.get("pelt_pen", ""), "PELT penalty", 0)
            if pen == 0:
                raise ValueError("PELT penalty must be > 0")
            cmd += ["--penalty", str(pen)]
        else:
            raise ValueError(f"unknown method {method!r}")
    except ValueError as exc:
        field = {"hmm": "regime_hmm_states", "kmeans": "regime_km_k", "pelt": "regime_pelt_pen"}.get(method, "regime_method")
        raise RegimeError(field, str(exc))
    return cmd


def gemini_command(csv_path, symbol, top_k_raw, api_key, python=None):
    """-> (cmd, masked display text). The key rides on the command line like the CTk app,
    but only the masked text is ever shown or logged."""
    python = python or sys.executable
    if not csv_path:
        raise RegimeError("regime_csv", "Please pick a CSV first (Browse above) — Gemini needs price data "
                                        "to detect shock dates before labeling.")
    symbol = str(symbol or "").strip()
    if not symbol:
        raise RegimeError("regime_ev_symbol", "Symbol is required")
    try:
        top_k = input_number(top_k_raw, "Top-K", 1, integer=True)
    except ValueError as exc:
        raise RegimeError("regime_ev_topk", str(exc))
    cmd = [python, "gemini_labeler.py", str(csv_path), "--symbol", symbol, "--top-k", str(top_k)]
    if api_key:
        cmd += ["--api-key", api_key]
    masked = [(c if c != api_key else (api_key[:8] + "..." + api_key[-4:])) for c in cmd] if api_key else list(cmd)
    return cmd, " ".join(masked)


def event_status_text(work_dir=ROOT):
    """Show last_updated + n_events from known_events.json (if any)."""
    p = Path(work_dir) / "known_events.json"
    if not p.exists():
        return "● No known_events.json yet — using 3 hardcoded events (Lehman/Brexit/Truss)"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        n = data.get("n_events", 0)
        ts = data.get("last_updated", "?")
        sym = data.get("symbol", "?")
        return f"✓ {n} events loaded · symbol {sym} · last updated {ts}"
    except Exception as e:
        return f"⚠ known_events.json present but unreadable: {e}"


def load_results(work_dir=ROOT):
    """regime_single_data.json -> {"rows": [[date, match, delta]], "summary", "good", "csv"}; None when absent."""
    json_path = Path(work_dir) / "regime_single_data.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    breaks = data.get("breaks", [])
    events = data.get("events", {})
    method = data.get("method", "?").upper()
    score = data.get("score", 0)
    n_bars = data.get("n_bars", 0)
    time_sec = data.get("time_sec", 0)
    try:
        event_dates = {n: datetime.strptime(d, "%Y-%m-%d") for n, d in events.items()}
    except Exception:
        event_dates = {}
    rows = []
    for b in breaks:
        try:
            bdt = datetime.strptime(b, "%Y-%m-%d")
        except Exception:
            rows.append([b, "—", "—"])
            continue
        best_name, best_delta = None, None
        for name, edt in event_dates.items():
            delta = abs((bdt - edt).days)
            if delta <= 90 and (best_delta is None or delta < best_delta):
                best_name, best_delta = name, delta
        rows.append([b, f"✓ {best_name}", f"{best_delta}d"] if best_name else [b, "—", "—"])
    n_events = len(event_dates) if event_dates else 0
    good = (n_events == 0) or (score / max(n_events, 1) >= 0.4)
    summary = (f"Method: {method}  ·  {len(breaks)} breakpoints  ·  {score}/{n_events} known events matched  ·  "
               f"{n_bars:,} bars analyzed in {time_sec:.2f}s")
    return {"rows": rows, "summary": summary, "good": good, "csv": data.get("csv"), "mtime": json_path.stat().st_mtime}


def same_source(results_csv, csv_path):
    try:
        return results_csv is not None and Path(results_csv).resolve() == Path(csv_path).resolve()
    except Exception:
        return str(results_csv) == str(csv_path)


def cutoff_target(csv_path, date_str):
    src = Path(csv_path)
    return src.parent / f"{src.stem}_from_{date_str}.csv"


def apply_cutoff(csv_path, date_str):
    """Write <stem>_from_<date>.csv (rows with timestamp >= date) + forward the params
    sidecar. -> {"path", "before", "after", "params_msg", "log"}; RegimeError on bad input."""
    import pandas as pd
    src_csv = Path(csv_path)
    dst_csv = cutoff_target(src_csv, date_str)
    try:
        df = pd.read_csv(src_csv)
    except Exception as e:
        raise RegimeError("regime_csv", str(e), title="Filter failed")
    if "timestamp" not in df.columns:
        raise RegimeError("regime_csv", "CSV has no 'timestamp' column — can't filter by date.", title="Bad CSV")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    before = len(df)
    cutoff = pd.Timestamp(date_str)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    after = len(df)
    if after == 0:
        raise RegimeError("regime_table", f"Cutoff {date_str} leaves no rows in {src_csv.name}.", title="Cutoff outside CSV")
    df.to_csv(dst_csv, index=False)
    src_params = src_csv.with_suffix("").with_suffix(".params.json")
    if not src_params.exists():
        src_params = src_csv.parent / (src_csv.stem + ".params.json")
    dst_params = dst_csv.with_suffix("").with_suffix(".params.json")
    if src_params.exists():
        try:
            shutil.copy(src_params, dst_params)
            params_msg = f"\n  + forwarded {src_params.name}"
        except Exception as e:
            params_msg = f"\n  (params sidecar copy failed: {e})"
    else:
        params_msg = ""
    line = f"[cutoff] saved {dst_csv.name} · {after:,} / {before:,} rows kept{params_msg}"
    return {"path": str(dst_csv), "before": before, "after": after, "params_msg": params_msg, "log": line}
