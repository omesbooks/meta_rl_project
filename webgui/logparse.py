"""Subprocess stdout -> structured events (same parsing rules as rl_app.py).

The page never parses log text: Python turns every line into an event dict that the
UI only paints. Rules mirror RLTradingStudio._classify_log / _parse_progress /
_handle_log_line (train branch).
"""
from __future__ import annotations

import re

from metric_health import METRIC_INFO, classify_metric

_METRIC = re.compile(r"\|\s*([a-z_]+)\s*\|\s*([+\-\d\.eE]+)\s*\|")
_TQDM = re.compile(r"(\d[\d,]*)/(\d[\d,]*)\s*\[")
_TOTAL = re.compile(r"\|\s*total_timesteps\s*\|\s*(\d+)\s*\|")
_VIZ = re.compile(r"VIZ_(SNAPSHOT|ROLLOUT)\s+(\d+)\s+(\d+)\s+(.+?)\s*$")


def classify(line):
    m = _METRIC.match(line)
    if m:
        try:
            status = classify_metric(m.group(1), float(m.group(2)))
            return {"good": "success", "warn": "warn", "bad": "error"}.get(status, "metric")
        except (ValueError, KeyError):
            pass
    low = line.lower()
    if low.startswith("wf_progress"):
        return "metric"
    if any(w in low for w in ("error", "fail", "exception", "traceback")):
        return "error"
    if any(w in low for w in ("warn", "skip", "abort")):
        return "warn"
    if any(w in low for w in ("✓", "ok", "success", "[save]", "complete", "done")):
        return "success"
    if re.search(r"step\s+\d+", low) or re.search(r"\d+%", low):
        return "metric"
    return "info"


def progress(line, total=None):
    m = _TQDM.search(line)
    if m:
        try:
            return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        except ValueError:
            pass
    m = _TOTAL.search(line)
    if m and total:
        return int(m.group(1)), int(total)
    return None


def metric(line):
    m = _METRIC.match(line)
    if not m or m.group(1) not in METRIC_INFO:
        return None
    try:
        value = float(m.group(2))
    except ValueError:
        return None
    return {"name": m.group(1), "value": value, "status": classify_metric(m.group(1), value),
            "label": METRIC_INFO[m.group(1)].get("label", m.group(1))}


def viz(line):
    m = _VIZ.search(line)
    if not m:
        return None
    return {"kind": "rollout" if m.group(1) == "ROLLOUT" else "eval", "index": int(m.group(2)),
            "step": int(m.group(3)), "path": m.group(4)}


def walkforward(line):
    """WF_PROGRESS + verdict (rl_walkforward.py stdout) -> {"progress": [cur, total, label]?, "verdict": {...}?}."""
    import pipeline_shared
    out = {}
    prog = pipeline_shared.parse_wf_progress(line)
    if prog:
        out["progress"] = list(prog)
    v = pipeline_shared.wf_verdict(line)
    if v:
        out["verdict"] = {"icon": v[0], "title": v[1], "tone": v[2], "sub": v[3]}
    return out or None


def event(line, page="train", total=None):
    """One stdout line -> event dict for the page (keys only present when they apply)."""
    ev = {"type": "line", "page": page, "line": line, "cls": classify(line)}
    if page == "walkfwd":
        w = walkforward(line)
        if w:
            ev["wf"] = w
    if page in ("train", "pipeline", "finetune"):
        p = progress(line, total if page != "finetune" else None)
        if p:
            ev["progress"] = {"step": p[0], "total": p[1]}
        m = metric(line)
        if m:
            ev["metric"] = m
        v = viz(line) if page == "train" else None
        if v:
            ev["viz"] = v
    return ev
