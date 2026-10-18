"""Train-page form -> rl_train.py command line, with the same rules as the CTk app.

This is the contract between the web GUI and the training engine: for the same inputs
`build_train_command` must produce exactly the command `RLTradingStudio._start_training`
builds (test_webgui.py checks this against the live CTk app). Everything here is pure
Python - no Tk, no pywebview - so it can be tested without a window.

The form is a plain dict (what the page sends). Reward/action slider values arrive in RAW
units (the page converts from display units with the spec's display_scale).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from action_profiles import ACTION_PARAM_SPECS, ACTION_PROFILES
from reward_formula import DEFAULT_REWARD_FORMULA, validate_reward_formula
from reward_profiles import REWARD_PARAM_SPECS, REWARD_PROFILES
from ui_values import fraction as input_fraction
from ui_values import number as input_number

ALGOS = {"ppo": "PPO (recommended)", "a2c": "A2C", "dqn": "DQN"}
REWARD_MODES = {"realized": "realized (recommended)", "mtm": "mtm"}
PPO_PRESETS = {
    "default": {"lr": "3e-4", "clip": "0.2", "ent": "0.01", "nsteps": "2048", "nepochs": "10", "batch": "64", "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
    "stable": {"lr": "1e-4", "clip": "0.1", "ent": "0.01", "nsteps": "4096", "nepochs": "10", "batch": "128", "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
    "fast": {"lr": "5e-4", "clip": "0.3", "ent": "0.01", "nsteps": "1024", "nepochs": "5", "batch": "64", "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
    "explore": {"lr": "3e-4", "clip": "0.2", "ent": "0.05", "nsteps": "2048", "nepochs": "10", "batch": "64", "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
}
# web-only engine options (not on the CTk page): appended only when they differ from these
EXTRA_DEFAULTS = {"entry_gate": "", "spread_pct": 0.0002, "commission": 0.0001, "seed": ""}


def form_defaults():
    p = PPO_PRESETS["default"]
    return {
        "csv": "", "name": "rl_prod_v1", "algo": "ppo", "reward_mode": "realized",
        "reward_profile": "balanced", "action_profile": "basic_4",
        "steps": "200000", "window": "10", "max_hold": "30", "train_pct": "85",
        "mc_eval": "1000", "mc_skip": "10", "corr_threshold": "1.0",
        "lr": p["lr"], "clip": p["clip"], "ent": p["ent"], "nsteps": p["nsteps"], "nepochs": p["nepochs"],
        "batch": p["batch"], "gamma": p["gamma"], "gae": p["gae"], "vf": p["vf"],
        "reward_values": dict(REWARD_PROFILES["balanced"]["params"]),
        "action_values": dict(ACTION_PROFILES["basic_4"]["params"]),
        "reward_formula_enabled": False, "reward_formula": DEFAULT_REWARD_FORMULA,
        "reward_json": None, "action_json": None,
        "viz": True, "viz_mode": "monitor",
        "entry_gate": "", "spread_pct": "0.0002", "commission": "0.0001", "seed": "",
    }


class FormError(ValueError):
    def __init__(self, field, message):
        super().__init__(message)
        self.field = field
        self.message = message


def _int(form, key, label, minimum=1):
    raw = str(form.get(key, "")).strip()
    try:
        return input_number(raw, label, minimum, integer=True)
    except (TypeError, ValueError, OverflowError):
        raise FormError(key, f"{label} must be a whole number >= {minimum} (got {raw!r}).")


def _fraction(form, key, label, **kw):
    try:
        return input_fraction(form.get(key, ""), label, **kw)
    except ValueError as exc:
        raise FormError(key, str(exc))


def _corr(form):
    try:
        value = input_number(form.get("corr_threshold", ""), "Correlation threshold", 0, 1)
    except ValueError as exc:
        raise FormError("corr_threshold", str(exc))
    if value <= 0:
        raise FormError("corr_threshold", "Correlation threshold: must be > 0 (use 1 to keep every non-constant feature)")
    return value


def normalize_name(raw):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(raw).strip()).rstrip(". ")
    if not cleaned:
        raise FormError("name", "Model name is empty after cleanup.")
    return cleaned


def _overrides(values, defaults, specs, field, integer_aware=False):
    """Values that differ from the baseline by more than the spec's display precision -
    the same rule as RLTradingStudio._collect_train_*_overrides."""
    out = {}
    for spec in specs:
        key = spec["key"]
        if key not in defaults:
            continue
        raw = (values or {}).get(key, defaults[key])
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise FormError(f"{field}.{key}", f"{spec['label']} must be numeric.")
        if not math.isfinite(value) or value < float(spec["min"]) or value > float(spec["max"]):
            raise FormError(f"{field}.{key}", f"{spec['label']} must be between {spec['min']} and {spec['max']}.")
        if integer_aware and int(spec.get("decimals", 3)) == 0:
            if not value.is_integer():
                raise FormError(f"{field}.{key}", f"{spec['label']} must be a whole number.")
            value = int(value)
        if abs(float(value) - float(defaults[key])) > 10 ** (-(int(spec.get("decimals", 3)) + 1)):
            out[key] = value
    return out


def reward_overrides(form, compare_loaded_json=True):
    key = form.get("reward_profile", "balanced")
    if key not in REWARD_PROFILES:
        raise FormError("reward_profile", f"unknown reward profile {key!r}")
    defaults = dict(REWARD_PROFILES[key]["params"])
    rj = form.get("reward_json") or {}
    if compare_loaded_json and rj.get("path"):
        defaults.update(rj.get("overrides") or {})
    return _overrides(form.get("reward_values"), defaults, REWARD_PARAM_SPECS, "reward_values")


def action_params(form, compare_loaded_json=True):
    key = form.get("action_profile", "basic_4")
    if key not in ACTION_PROFILES:
        raise FormError("action_profile", f"unknown action profile {key!r}")
    defaults = dict(ACTION_PROFILES[key]["params"])
    aj = form.get("action_json") or {}
    if compare_loaded_json and aj.get("path"):
        defaults.update(aj.get("params") or {})
    return _overrides(form.get("action_values"), defaults, ACTION_PARAM_SPECS, "action_values", integer_aware=True)


def reward_formula(form):
    if not form.get("reward_formula_enabled"):
        return ""
    text = str(form.get("reward_formula") or "").strip()
    try:
        validate_reward_formula(text)
    except Exception as exc:
        raise FormError("reward_formula", f"Invalid reward formula: {exc}")
    return text


def validate_ppo(values):
    from types import SimpleNamespace
    from training_data import validate_hyperparameters
    try:
        validate_hyperparameters(SimpleNamespace(**{k: float(v) for k, v in values.items()}))
    except (ValueError, TypeError, OverflowError) as exc:
        raise FormError("advanced", str(exc))


def build_train_command(form, python=None):
    """-> {"cmd": [...], "steps": int, "name": str, "warnings": [...]}; raises FormError."""
    python = python or sys.executable
    warnings = []
    csv = str(form.get("csv") or "").strip()
    if not csv:
        raise FormError("csv", "Please select a CSV file first")
    if Path(csv).name.endswith("_m1.csv"):
        raise FormError("csv", f"{Path(csv).name} is raw M1 execution data for backtests, not a feature dataset")
    steps = _int(form, "steps", "Training Steps")
    window = _int(form, "window", "Window")
    max_hold = _int(form, "max_hold", "Max Hold")
    mc_eval = _int(form, "mc_eval", "MC Eval Runs", minimum=0)
    name = normalize_name(form.get("name", ""))
    reward_mode = form.get("reward_mode", "realized")
    if reward_mode not in REWARD_MODES:
        raise FormError("reward_mode", f"unknown reward mode {reward_mode!r}")
    overrides = reward_overrides(form)
    formula = reward_formula(form)
    action_profile = form.get("action_profile", "basic_4")
    aparams = action_params(form)
    train_pct = _fraction(form, "train_pct", "Train %", allow_zero=False)
    mc_skip = _fraction(form, "mc_skip", "MC Skip %", allow_one=False)
    corr = _corr(form)
    algo = form.get("algo", "ppo")
    if algo not in ALGOS:
        raise FormError("algo", f"unknown algorithm {algo!r}")
    nsteps = _int(form, "nsteps", "N Steps")
    nepochs = _int(form, "nepochs", "N Epochs")
    batch = _int(form, "batch", "Batch Size")
    lr, clip, ent = str(form.get("lr")), str(form.get("clip")), str(form.get("ent"))
    gamma, gae, vf = str(form.get("gamma")), str(form.get("gae")), str(form.get("vf"))
    validate_ppo(dict(learning_rate=lr, clip_range=clip, ent_coef=ent, n_steps=nsteps, n_epochs=nepochs,
                      batch_size=batch, gamma=gamma, gae_lambda=gae, vf_coef=vf))
    cmd = [python, "rl_train.py", csv,
           "--steps", str(steps), "--window", str(window), "--max_hold", str(max_hold),
           "--train_pct", str(train_pct), "--mc_eval", str(mc_eval), "--mc_skip_frac", str(mc_skip),
           "--corr_threshold", f"{corr:g}", "--reward_mode", reward_mode,
           "--reward_profile", form.get("reward_profile", "balanced"),
           "--reward_overrides", json.dumps(overrides, separators=(",", ":")),
           "--action_profile", action_profile, "--action_params", json.dumps(aparams, separators=(",", ":")),
           "--algo", algo, "--name", name,
           "--learning_rate", lr, "--clip_range", clip, "--ent_coef", ent,
           "--n_steps", str(nsteps), "--n_epochs", str(nepochs), "--batch_size", str(batch),
           "--gamma", gamma, "--gae_lambda", gae, "--vf_coef", vf]
    rj = form.get("reward_json") or {}
    json_formula_disabled = bool(rj.get("path")) and bool(rj.get("has_formula")) and not formula
    if rj.get("path") and not json_formula_disabled:
        cmd += ["--reward_profile_json", rj["path"]]
    elif json_formula_disabled:
        full = reward_overrides(form, compare_loaded_json=False)
        cmd[cmd.index("--reward_overrides") + 1] = json.dumps(full, separators=(",", ":"))
        warnings.append("reward JSON formula disabled — passing the JSON's numeric overrides explicitly instead")
    if formula:
        cmd += ["--reward_formula", formula]
    aj = form.get("action_json") or {}
    if aj.get("path"):
        cmd += ["--action_profile_json", aj["path"]]
    if form.get("viz"):
        if algo == "dqn":
            warnings.append("Live 3D visual needs PPO/A2C — skipped for DQN")
        else:
            cmd.append("--viz")
    # ---- web-only engine options (CTk page has no fields for these)
    gate = str(form.get("entry_gate") or "").strip()
    if gate:
        cmd += ["--entry_gate", gate]
    for key, flag, label in (("spread_pct", "--spread_pct", "Spread per side"), ("commission", "--commission", "Commission")):
        raw = str(form.get(key, "")).strip()
        if raw == "":
            continue
        try:
            value = input_number(raw, label, 0)
        except ValueError as exc:
            raise FormError(key, str(exc))
        if abs(value - EXTRA_DEFAULTS[key]) > 1e-12:
            cmd += [flag, f"{value:g}"]
    seed = str(form.get("seed") or "").strip()
    if seed:
        cmd += ["--seed", str(_int({"seed": seed}, "seed", "Seed", minimum=0))]
    return {"cmd": cmd, "steps": steps, "name": name, "warnings": warnings}


def suggest_steps(rows, train_pct_raw, window_raw):
    """Same rule as the CTk Train page hint: passes over the effective train bars."""
    if not rows:
        return None
    try:
        train_pct = input_fraction(train_pct_raw, "Train %", allow_zero=False)
        window = input_number(window_raw, "Window", 1, integer=True)
    except ValueError:
        return None
    effective = max(int(rows * train_pct) - max(window, 0), 1)
    if effective < 5000:
        passes, minimum = 30, 100000
    elif effective < 30000:
        passes, minimum = 12, 200000
    elif effective < 100000:
        passes, minimum = 8, 400000
    else:
        passes, minimum = 5, 600000
    step = 50000
    suggested = max(step, int(round(max(effective * passes, minimum) / step) * step))
    return {"steps": suggested, "train_bars": effective, "passes": round(suggested / effective, 1)}
