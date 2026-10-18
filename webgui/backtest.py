"""Backtest-page form -> backtest_live.py / backtest_chart.py command lines.

Same contract as webgui/commands.py for training: for the same inputs
`build_backtest_command` produces exactly the command RLTradingStudio._run_backtest
builds (test_ui_training_contract.py::test_web_backtest_command_equals_ctk_*).
"""
from __future__ import annotations

import json
import sys

from ui_values import fraction as input_fraction
from ui_values import number as input_number
from ui_values import percent_units

from webgui.commands import FormError, _int

MODES = {"pure_agent": "Pure Agent (matches training) ⭐", "agent_sltp": "Agent + SL/TP (live-realistic)"}
INTRABAR = {"pessimistic": "pessimistic (SL first) ⭐", "optimistic": "optimistic (TP first)"}
SOURCES = {"final": "Final ⭐", "best": "Best (pre-overfit)"}
NONE = "(none)"


def form_defaults():
    """The CTk Backtest page's initial field values."""
    return {"model": "", "source": "final", "csv": "", "m1_csv": NONE, "conf": "0", "start": "0",
            "mode": "pure_agent", "risk": "0.01", "max_pos": "1", "sl": "2.0", "tp": "4.0",
            "intrabar": "pessimistic", "stop_slip": "0.01", "swap_long": "0", "swap_short": "0",
            "random_baseline": "20", "mc": "1000", "window": "auto"}


def model_max_hold(model):
    """(max_hold, source, warning) — the trained max hold from the model's train meta, else 30."""
    from artifact_paths import train_meta_path
    meta = train_meta_path(model)
    if meta.exists():
        try:
            hparams = json.loads(meta.read_text(encoding="utf-8-sig")).get("hyperparameters", {})
            if hparams.get("max_hold") is not None:
                return int(hparams["max_hold"]), "model meta", ""
        except Exception as exc:
            return 30, "fallback", f"[warn] cannot read max_hold from meta: {exc}"
    return 30, "fallback", ""


def _num(form, key, fn):
    try:
        return fn(form.get(key, ""))
    except ValueError as exc:
        raise FormError(key, str(exc))


def build_backtest_command(form, python=None):
    """-> {"cmd", "model", "max_hold", "max_hold_source", "warnings"}; raises FormError(field, message)."""
    python = python or sys.executable
    model = str(form.get("model", "")).strip()
    csv = str(form.get("csv", "")).strip()
    if model in (NONE, ""):
        raise FormError("bt_model", "Please select model and dataset")
    if csv in (NONE, ""):
        raise FormError("bt_csv", "Please select model and dataset")

    max_positions = _int(form, "max_pos", "Max Positions")
    window_raw = str(form.get("window", "")).strip()
    window_override = 0 if window_raw.lower() == "auto" else _int(form, "window", "Window", minimum=1)
    random_baseline = _int(form, "random_baseline", "Random Baseline", minimum=0)
    mc_runs = _int(form, "mc", "Monte Carlo Runs", minimum=0)

    max_hold, max_hold_source, warn = model_max_hold(model)
    warnings = [warn] if warn else []

    mode = "pure_agent" if form.get("mode", "pure_agent") == "pure_agent" else "agent_sltp"
    intrabar = "optimistic" if form.get("intrabar") == "optimistic" else "pessimistic"
    stop_slip = _num(form, "stop_slip", lambda v: percent_units(v, "Stop slippage", 0))
    skip = _num(form, "start", lambda v: input_fraction(v, "Backtest start", allow_one=False))
    conf = _num(form, "conf", lambda v: input_number(v, "Confidence", 0, 1))
    risk = _num(form, "risk", lambda v: input_number(v, "Risk", 0, 1))
    atr_sl = _num(form, "sl", lambda v: input_number(v, "ATR SL", 0))
    atr_tp = _num(form, "tp", lambda v: input_number(v, "ATR TP", 0))
    swap_long = _num(form, "swap_long", lambda v: percent_units(v, "Swap long"))
    swap_short = _num(form, "swap_short", lambda v: percent_units(v, "Swap short"))

    model_source = "best" if form.get("source") == "best" else "final"
    cmd = [
        python, "backtest_live.py", model, csv,
        "--source", model_source,
        "--conf", str(conf),
        "--risk", str(risk),
        "--max_positions", str(max_positions),
        "--max_hold", str(max_hold),
        "--atr_sl", str(atr_sl),
        "--atr_tp", str(atr_tp),
        "--window", str(window_override),  # 0 = auto-detect from model
        "--start", str(skip),  # skip frac
        "--mode", mode,
        "--intrabar", intrabar,
        "--stop_slippage", str(stop_slip),
    ]
    m1_choice = str(form.get("m1_csv", "")).strip()
    if m1_choice and m1_choice != NONE:
        cmd += ["--m1_csv", m1_choice]
    cmd += ["--swap_long", str(swap_long), "--swap_short", str(swap_short)]
    cmd += ["--random_baseline", str(random_baseline)]
    cmd += ["--mc", str(mc_runs)]
    return {"cmd": cmd, "model": model, "max_hold": max_hold, "max_hold_source": max_hold_source, "warnings": warnings}


def build_chart_command(form, python=None):
    """'Generate Chart': needs a trades file from a previous run (same check as the CTk page)."""
    from artifact_paths import find_trades_path
    python = python or sys.executable
    model = str(form.get("model", "")).strip()
    csv = str(form.get("csv", "")).strip()
    if model in (NONE, "") or csv in (NONE, ""):
        raise FormError("bt_model" if model in (NONE, "") else "bt_csv", "Please select model and dataset")
    if not find_trades_path(model):
        raise FormError("bt_model", f"ไม่พบไฟล์ {model}_live_bt_trades.csv — กรุณา Run Backtest ก่อน เพื่อสร้างไฟล์ trades")
    return {"cmd": [python, "backtest_chart.py", model, csv, "--limit", "5000"], "model": model}
