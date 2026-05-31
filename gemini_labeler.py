"""
Gemini-based Auto Event Labeler
================================
Auto-detect price shocks in a CSV, then label each with Gemini AI.
No manual event list needed.

Pipeline:
    CSV ─► detect shocks ─► Gemini labels ─► known_events.json

Setup (one-time):
    1. Get a free Gemini API key: https://aistudio.google.com/app/apikey
    2. Save it via the GUI Settings page (or set GEMINI_API_KEY env var)

Usage:
    python gemini_labeler.py gu_h4_dataset.csv --symbol GBPUSD --top-k 15
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


GEMINI_MODEL = "gemini-2.5-flash"  # free tier, fast, JSON-friendly
EVENTS_FILE = "known_events.json"
RATE_LIMIT_SLEEP = 4.5  # 15 req/min free tier => sleep 4s between calls


# -------------------- Layer B: auto-detect shock dates --------------------

def load_daily(csv_path: str) -> pd.DataFrame:
    """Load CSV and resample to daily close."""
    df = pd.read_csv(csv_path, usecols=lambda c: c.lower() in ("timestamp", "close"))
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("CSV must have 'timestamp' and 'close' columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date").close.last().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ret"] = daily.close.pct_change()
    return daily


def detect_shocks(daily: pd.DataFrame, top_k: int = 15,
                  min_gap_days: int = 30) -> list[dict]:
    """Find the top-K price shocks using volatility-normalized z-score.

    A shock is a day where |return| is unusually large vs a rolling 60-day baseline.
    Selects top-K candidates while enforcing a minimum separation between them
    so we don't get a cluster of 5 days from the same event.
    """
    daily = daily.copy()
    daily["abs_ret"] = daily.ret.abs()
    daily["base_mean"] = daily.abs_ret.rolling(60).mean()
    daily["base_std"] = daily.abs_ret.rolling(60).std()
    daily["z"] = (daily.abs_ret - daily.base_mean) / daily.base_std.replace(0, np.nan)
    daily = daily.dropna(subset=["z"])

    candidates = daily.nlargest(top_k * 5, "z")
    candidates = candidates.sort_values("date")

    selected = []
    last_date = None
    for _, row in candidates.iterrows():
        if last_date is None or (row.date - last_date).days >= min_gap_days:
            selected.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "shock_pct": round(float(row.ret * 100), 2),
                "z_score": round(float(row.z), 2),
            })
            last_date = row.date
        if len(selected) >= top_k:
            break
    return selected


# -------------------- Layer D: Gemini labeler --------------------

def label_one(date: str, symbol: str, shock_pct: float, api_key: str,
              client=None) -> dict:
    """Ask Gemini what event likely caused this market move.

    Returns a dict with keys: event, category, confidence, rationale.
    On any error (network, parse, rate-limit), returns a placeholder so the
    pipeline never aborts on a single bad call.
    """
    try:
        from google import genai
    except ImportError:
        return {"event": f"Shock @ {date[:7]}", "category": "unknown",
                "confidence": 0.0, "rationale": "google-genai not installed"}

    if client is None:
        client = genai.Client(api_key=api_key)

    direction = "fell" if shock_pct < 0 else "rose"
    prompt = (
        f"On {date}, {symbol} {direction} by {abs(shock_pct):.1f}% in one day.\n\n"
        f"What was the most likely macroeconomic, political, or financial event "
        f"that caused this move? Consider central bank actions, political events, "
        f"crises, geopolitical shocks, and major data releases for that exact week.\n\n"
        f"Reply with ONLY a JSON object, no markdown fences, in this format:\n"
        f'{{"event": "short name max 40 chars", '
        f'"category": "monetary|political|crisis|geopolitical|data|tech|unknown", '
        f'"confidence": 0.0_to_1.0, '
        f'"rationale": "one sentence why"}}'
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return {
            "event": str(data.get("event", "Unknown"))[:60],
            "category": str(data.get("category", "unknown")),
            "confidence": float(data.get("confidence", 0.5)),
            "rationale": str(data.get("rationale", ""))[:200],
        }
    except json.JSONDecodeError as e:
        return {"event": f"Shock @ {date[:7]}", "category": "unknown",
                "confidence": 0.0, "rationale": f"JSON parse failed: {e}"}
    except Exception as e:
        return {"event": f"Shock @ {date[:7]}", "category": "unknown",
                "confidence": 0.0, "rationale": f"API error: {str(e)[:100]}"}


# -------------------- Orchestrator --------------------

def refresh_known_events(csv_path: str, symbol: str, api_key: str,
                         top_k: int = 15, min_gap_days: int = 30,
                         out_path: str = EVENTS_FILE,
                         verbose: bool = True) -> dict:
    """Full pipeline: detect shocks -> label with Gemini -> save JSON."""
    if verbose:
        print(f"[detect] scanning {csv_path} for top-{top_k} shocks ...", flush=True)
    daily = load_daily(csv_path)
    shocks = detect_shocks(daily, top_k=top_k, min_gap_days=min_gap_days)
    if verbose:
        print(f"[detect] found {len(shocks)} candidates "
              f"({daily.date.min().date()} -> {daily.date.max().date()})", flush=True)

    if not api_key:
        if verbose:
            print("[gemini] no API key provided; saving shock dates without labels", flush=True)
        labeled = [{**s, "event": f"Shock @ {s['date'][:7]}",
                    "category": "unknown", "confidence": 0.0, "source": "auto-detect"}
                   for s in shocks]
    else:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"[gemini] client init failed: {e}", flush=True)
            client = None

        labeled = []
        for i, s in enumerate(shocks, 1):
            if verbose:
                print(f"[gemini] ({i}/{len(shocks)}) labeling {s['date']} ({s['shock_pct']:+.1f}%) ...",
                      flush=True)
            label = label_one(s["date"], symbol, s["shock_pct"], api_key, client=client)
            labeled.append({**s, **label, "source": "gemini"})
            if verbose:
                print(f"  -> {label['event']} ({label['category']}, conf={label['confidence']:.2f})",
                      flush=True)
            if i < len(shocks):
                time.sleep(RATE_LIMIT_SLEEP)  # respect 15 req/min free tier

    output = {
        "symbol": symbol,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": Path(csv_path).name,
        "n_events": len(labeled),
        "events": labeled,
    }
    Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    if verbose:
        print(f"\n[save] {len(labeled)} events -> {out_path}", flush=True)
    return output


def load_known_events(path: str = EVENTS_FILE) -> dict[str, pd.Timestamp]:
    """Load events back as {name: Timestamp} for use by regime_compare scoring.

    Returns a dict that can be a drop-in replacement for the hardcoded
    KNOWN_EVENTS in regime_compare.py.
    """
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {ev["event"]: pd.Timestamp(ev["date"]) for ev in data.get("events", [])}


# -------------------- CLI --------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-label price shocks with Gemini")
    ap.add_argument("csv", help="CSV with timestamp + close columns")
    ap.add_argument("--symbol", default="GBPUSD", help="symbol name (for Gemini context)")
    ap.add_argument("--top-k", type=int, default=15, help="max number of shocks to label")
    ap.add_argument("--min-gap-days", type=int, default=30,
                    help="minimum days between selected shocks")
    ap.add_argument("--api-key", default="",
                    help="Gemini API key (or set GEMINI_API_KEY env var; "
                         "without one, output is shock dates only)")
    ap.add_argument("--out", default=EVENTS_FILE, help="output JSON file")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    refresh_known_events(
        csv_path=args.csv,
        symbol=args.symbol,
        api_key=api_key,
        top_k=args.top_k,
        min_gap_days=args.min_gap_days,
        out_path=args.out,
    )
