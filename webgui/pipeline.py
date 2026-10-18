"""Pipeline / Walk-Forward / Fine-tune forms -> commands, with the CTk rules.

Contract (test_ui_training_contract.py::test_web_pipeline_* / test_web_walkforward_* /
test_web_finetune_*): for the same inputs these build exactly what
RLTradingStudio._run_full_pipeline / _pipeline_worker, _run_walkforward and _run_finetune
build. FormError.field is the id of the page field to highlight.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from action_profiles import ACTION_PROFILES, get_action_profile, load_action_profile_json
from reward_formula import validate_reward_formula
from reward_profiles import REWARD_PROFILES
from ui_values import fraction as input_fraction
from ui_values import number as input_number

import pipeline_shared
from webgui import commands
from webgui.commands import FormError

NONE = "(none)"
PIPE_HP = [  # (form key, CTk label, default, hint) - the CTk h_fields order
    ("lr", "learning_rate", "3e-4", "ความแรงในการเรียนรู้"), ("clip", "clip_range", "0.2", "กัน policy เปลี่ยนแรงเกิน"),
    ("ent", "ent_coef", "0.01", "เพิ่ม/ลดการลองทางใหม่"), ("nsteps", "n_steps", "2048", "จำนวน step ก่อน update"),
    ("nepochs", "n_epochs", "10", "รอบเรียนซ้ำต่อ rollout"), ("batch", "batch_size", "64", "ขนาด minibatch"),
    ("gamma", "gamma", "0.99", "น้ำหนักผลลัพธ์อนาคต"), ("gae", "gae_lambda", "0.95", "ทำให้ advantage นิ่งขึ้น"),
    ("vf", "vf_coef", "0.5", "น้ำหนัก critic/value loss"), ("max_hold", "max_hold", "30", "ถือ position ได้กี่แท่ง"),
    ("ep_len", "ep_len", "2000", "ความยาว episode ตอน train"), ("net_arch", "net_arch", "auto", "ขนาด neural network"),
]
WF_PPO = [("lr", "Learning Rate", "3e-4"), ("clip", "Clip Range", "0.2"), ("ent", "Entropy Coef", "0.01"),
          ("nsteps", "N Steps", "2048"), ("nepochs", "N Epochs", "10"), ("batch", "Batch Size", "64"),
          ("gamma", "Gamma", "0.99"), ("gae", "GAE Lambda", "0.95"), ("vf", "VF Coef", "0.5")]
FT_MODES = {"mixed": "mixed (old+new ⭐)", "pure": "pure (new only)", "replay": "replay (all old+new)"}


def _clean_name(raw):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(raw or "").strip()).rstrip(". ")


# ------------------------------------------------------------------ pipeline
def pipeline_defaults():
    d = {"csv": "", "bt_csv": "", "use_m1": True, "model_name": "rl_pipeline_v1", "train_pct": "85", "steps": "200000",
         "window": "10", "conf": "0", "bt_mode": "pure_agent", "bt_start": "auto", "risk": "0.01", "max_pos": "1",
         "sl": "2.0", "tp": "4.0", "reward_profile": "balanced", "action_profile": "basic_4", "reward_mode": "realized",
         "reward_overrides": "", "reward_formula": "", "action_params": ""}
    d.update({k: v for k, _l, v, _h in PIPE_HP})
    return d


def pipeline_hparams(form):
    """CTk _get_pipeline_hparams: same checks, same messages; FormError field = pipe_<key>."""
    values = {k: str(form.get(k, "")).strip() for k, _l, _d, _h in PIPE_HP}
    try:
        floats = {
            "learning_rate": float(values["lr"]), "clip_range": float(values["clip"]), "ent_coef": float(values["ent"]),
            "gamma": float(values["gamma"]), "gae_lambda": float(values["gae"]), "vf_coef": float(values["vf"]),
        }
        ints = {
            "n_steps": input_number(values["nsteps"], "nsteps", 1, integer=True),
            "n_epochs": input_number(values["nepochs"], "nepochs", 1, integer=True),
            "batch_size": input_number(values["batch"], "batch", 1, integer=True),
            "max_hold": input_number(values["max_hold"], "max_hold", 1, integer=True),
            "ep_len": input_number(values["ep_len"], "ep_len", 1, integer=True),
        }
    except (ValueError, OverflowError):
        raise FormError("pipe_hparams", "Hyperparameters must be numeric, except net_arch.")
    try:
        commands.validate_ppo({**floats, **ints})
    except FormError as exc:
        raise FormError("pipe_hparams", exc.message)
    if floats["learning_rate"] <= 0:
        raise FormError("pipe_lr", "learning_rate must be greater than 0.")
    if floats["clip_range"] <= 0:
        raise FormError("pipe_clip", "clip_range must be greater than 0.")
    if floats["ent_coef"] < 0 or floats["vf_coef"] < 0:
        raise FormError("pipe_ent", "ent_coef and vf_coef must be 0 or greater.")
    if not (0 <= floats["gamma"] <= 1):
        raise FormError("pipe_gamma", "gamma must be >= 0 and <= 1.")
    if not (0 <= floats["gae_lambda"] <= 1):
        raise FormError("pipe_gae", "gae_lambda must be >= 0 and <= 1.")
    if any(v <= 0 for v in ints.values()):
        raise FormError("pipe_hparams", "n_steps, n_epochs, batch_size, max_hold, and ep_len must be greater than 0.")
    if ints["batch_size"] > ints["n_steps"]:
        raise FormError("pipe_batch", "batch_size must be <= n_steps.")
    net_arch = values["net_arch"].strip()
    if net_arch != "auto":
        try:
            layers = [int(x.strip()) for x in net_arch.split(",")]
        except ValueError:
            raise FormError("pipe_net_arch", "net_arch must be 'auto' or comma-separated integers, e.g. 256,128,64.")
        if not layers or any(x <= 0 for x in layers):
            raise FormError("pipe_net_arch", "net_arch layers must be positive integers.")

    reward_mode = str(form.get("reward_mode", "realized")).split()[0]
    reward_profile = form.get("reward_profile", "balanced")
    if reward_profile not in REWARD_PROFILES:
        raise FormError("pipe_reward_profile", f"unknown reward profile {reward_profile!r}")
    action_profile = form.get("action_profile", "basic_4")
    if action_profile not in ACTION_PROFILES:
        raise FormError("pipe_action_profile", f"unknown action profile {action_profile!r}")
    reward_overrides = str(form.get("reward_overrides", "")).strip()
    if reward_overrides:
        try:
            json.loads(reward_overrides)
        except Exception as exc:
            raise FormError("pipe_reward_overrides", f"Invalid reward overrides JSON: {exc}")
    reward_formula = str(form.get("reward_formula", "")).strip()
    if reward_formula:
        try:
            validate_reward_formula(reward_formula)
        except Exception as exc:
            raise FormError("pipe_reward_formula", f"Invalid reward formula: {exc}")
    action_params = str(form.get("action_params", "")).strip()
    if action_params:
        try:
            get_action_profile(action_profile, json.loads(action_params))
        except Exception as exc:
            raise FormError("pipe_action_params", f"Invalid action params JSON: {exc}")
    return {
        "learning_rate": str(floats["learning_rate"]), "clip_range": str(floats["clip_range"]),
        "ent_coef": str(floats["ent_coef"]), "n_steps": str(ints["n_steps"]), "n_epochs": str(ints["n_epochs"]),
        "batch_size": str(ints["batch_size"]), "gamma": str(floats["gamma"]), "gae_lambda": str(floats["gae_lambda"]),
        "vf_coef": str(floats["vf_coef"]), "max_hold": str(ints["max_hold"]), "ep_len": str(ints["ep_len"]),
        "net_arch": net_arch, "reward_mode": reward_mode, "reward_profile": reward_profile,
        "reward_overrides": reward_overrides, "reward_formula": reward_formula,
        "action_profile": action_profile, "action_params": action_params,
    }


def build_pipeline(form, root, python=None):
    """CTk _run_full_pipeline up to the review step -> plan dict (train_cmd + context + stage inputs)."""
    python = python or sys.executable
    root = Path(root)
    csv_name = str(form.get("csv", "")).strip()
    if not csv_name or csv_name == NONE:
        raise FormError("pipe_csv", "Please select a dataset CSV first.")
    if not (root / csv_name).exists():
        raise FormError("pipe_csv", f"CSV not found: {root / csv_name}")
    bt_csv = str(form.get("bt_csv", "")).strip()
    if not bt_csv or bt_csv == NONE:
        bt_csv = csv_name
    if not (root / bt_csv).exists():
        raise FormError("pipe_bt_csv", f"Backtest CSV not found: {root / bt_csv}")
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(form.get("model_name", "")).strip()).strip("._-")
    if not model_name:
        raise FormError("pipe_model_name", "Model name is empty after cleanup.")

    def field(key, fn):
        try:
            return fn(form.get(key, ""))
        except ValueError as exc:
            raise FormError("pipe_" + key, str(exc))
    steps = field("steps", lambda v: input_number(v, "Steps", 1, integer=True))
    window = field("window", lambda v: input_number(v, "Window", 1, integer=True))
    conf = field("conf", lambda v: input_number(v, "Confidence", 0, 1))
    train_pct = field("train_pct", lambda v: input_fraction(v, "Train %", allow_zero=False))
    risk = field("risk", lambda v: input_number(v, "Risk", 0, 1))
    max_positions = field("max_pos", lambda v: input_number(v, "Max positions", 1, integer=True))
    atr_sl = field("sl", lambda v: input_number(v, "ATR SL", 0))
    atr_tp = field("tp", lambda v: input_number(v, "ATR TP", 0))
    bt_start_raw = str(form.get("bt_start", "")).strip().lower()
    bt_start = 0.0 if bt_start_raw == "auto" else field("bt_start", lambda v: input_fraction(bt_start_raw, "Backtest start", allow_one=False))

    hparams = pipeline_hparams(form)
    hparams["use_m1"] = bool(form.get("use_m1", True))
    hparams.update({"bt_start": str(bt_start), "bt_start_auto": bt_start_raw == "auto", "risk": str(risk),
                    "max_positions": str(max_positions), "atr_sl": str(atr_sl), "atr_tp": str(atr_tp)})
    mode = "pure_agent" if form.get("bt_mode", "pure_agent") == "pure_agent" else "agent_sltp"
    train_cmd = pipeline_shared.train_command(python, csv_name, bt_csv, model_name, steps, window, train_pct, hparams)
    context = {"backtest_csv": bt_csv, "confidence": conf, "mode": mode, **hparams}
    return {"train_cmd": train_cmd, "context": context, "csv": csv_name, "bt_csv": bt_csv, "model_name": model_name,
            "steps": steps, "window": window, "conf": conf, "mode": mode, "train_pct": train_pct, "hparams": hparams}


def pipeline_stages(plan, confirmed_train_cmd, python=None):
    """The stage commands the CTk _pipeline_worker runs after confirmation (relabel/build are
    switched off in the CTk app, so: Train PPO -> Backtest -> Chart)."""
    python = python or sys.executable
    hp = dict(plan["hparams"])
    resolved = plan["context"]["resolved_backtest"]
    hp["bt_start"] = str(resolved["start"])
    bt = pipeline_shared.backtest_command(python, plan["model_name"], plan["bt_csv"], plan["conf"], plan["mode"],
                                          plan["window"], hp, resolved["m1_csv"])
    return [("Train PPO", list(confirmed_train_cmd)), ("Backtest", bt),
            ("Chart", pipeline_shared.chart_command(python, plan["model_name"], plan["bt_csv"]))]


# ------------------------------------------------------------------ walk-forward
def wf_defaults():
    d = {"csv": "", "windows": "5", "steps": "50000", "window": "10", "action_profile": "basic_4",
         "reward_mode": "realized", "reward_profile": "balanced", "max_hold": "30", "corr": "", "name": "wf_prod",
         "reward_overrides": "", "action_params": "", "action_profile_json": "", "reward_formula": ""}
    d.update({k: v for k, _l, v in WF_PPO})
    return d


def wf_from_train(train_form):
    """'⧉ Copy settings from Train' (CTk _wf_copy_from_train): the WF fields + the hidden
    extras (reward overrides / formula / action params / action JSON) for the Train form."""
    try:
        overrides = commands.reward_overrides(train_form, compare_loaded_json=False)
        formula = commands.reward_formula(train_form)
        action_params = commands.action_params(train_form, compare_loaded_json=False)
    except FormError as exc:
        raise FormError("wf_copy", f"Train page: {exc.message}")
    # the CTk getters return compact JSON text; commands.* return dicts
    overrides = json.dumps(overrides, separators=(",", ":")) if isinstance(overrides, dict) else overrides
    action_params = json.dumps(action_params, separators=(",", ":")) if isinstance(action_params, dict) else action_params
    out = {"window": train_form.get("window", ""), "max_hold": train_form.get("max_hold", ""),
           "corr": train_form.get("corr_threshold", ""), "reward_mode": train_form.get("reward_mode", "realized"),
           "reward_profile": train_form.get("reward_profile", "balanced"),
           "action_profile": train_form.get("action_profile", "basic_4"),
           "reward_overrides": "" if overrides in ("{}", "") else overrides, "reward_formula": formula,
           "action_params": "" if action_params in ("{}", "") else action_params,
           "action_profile_json": (train_form.get("action_json") or {}).get("path", "") if isinstance(train_form.get("action_json"), dict) else ""}
    for k, _l, _d in WF_PPO:
        out[k] = train_form.get(k, "")

    def summarize(raw, none_text="(none)"):
        if not raw:
            return none_text
        try:
            d = json.loads(raw)
            return f"{len(d)} ค่า: " + ", ".join(list(d)[:4]) + ("…" if len(d) > 4 else "")
        except Exception:
            return "(set)"
    out["hint"] = (f"จาก Train — reward overrides: {summarize(out['reward_overrides'])} · "
                   f"action params: {summarize(out['action_params'])} · "
                   f"action JSON: {out['action_profile_json'] or '(none)'} · "
                   f"formula: {'ใช้' if out['reward_formula'] else '(none)'}")
    return out


def build_wf_command(form, python=None):
    """CTk _run_walkforward."""
    python = python or sys.executable
    csv = str(form.get("csv", "")).strip()
    if csv in (NONE, ""):
        raise FormError("wf_csv", "Please select dataset")
    windows = commands._int(form, "windows", "WF Windows")
    steps = commands._int(form, "steps", "WF Steps")
    window = commands._int(form, "window", "WF Window")
    max_hold = commands._int(form, "max_hold", "WF Max Hold")
    nsteps = commands._int(form, "nsteps", "WF N Steps")
    nepochs = commands._int(form, "nepochs", "WF N Epochs")
    batch = commands._int(form, "batch", "WF Batch Size")
    raw_corr = str(form.get("corr", "")).strip()
    wf_corr = ""
    if raw_corr:
        try:
            wf_corr = input_number(raw_corr, "WF Correlation threshold", 0, 1)
            if wf_corr <= 0:
                raise ValueError("WF Correlation threshold: must be > 0 (use 1 to keep every non-constant feature)")
        except ValueError as exc:
            raise FormError("corr", str(exc))
    output_name = _clean_name(form.get("name", ""))
    if not output_name:
        raise FormError("name", "Walk-Forward output name is empty after cleanup.")
    action_profile = form.get("action_profile", "basic_4")
    if action_profile not in ACTION_PROFILES:
        raise FormError("action_profile", f"unknown action profile {action_profile!r}")
    reward_profile = form.get("reward_profile", "balanced")
    if reward_profile not in REWARD_PROFILES:
        raise FormError("reward_profile", f"unknown reward profile {reward_profile!r}")
    action_params = str(form.get("action_params") or "").strip()
    reward_overrides = str(form.get("reward_overrides") or "").strip()
    reward_formula = str(form.get("reward_formula") or "").strip()
    g = lambda k: str(form.get(k, ""))
    cmd = [
        python, "rl_walkforward.py", csv,
        "--windows", str(windows), "--steps", str(steps), "--window", str(window), "--max_hold", str(max_hold),
        "--name", output_name, "--action_profile", action_profile,
        "--reward_mode", str(form.get("reward_mode", "realized")).split()[0], "--reward_profile", reward_profile,
        "--learning_rate", g("lr"), "--clip_range", g("clip"), "--ent_coef", g("ent"),
        "--n_steps", str(nsteps), "--n_epochs", str(nepochs), "--batch_size", str(batch),
        "--gamma", g("gamma"), "--gae_lambda", g("gae"), "--vf_coef", g("vf"),
    ]
    if wf_corr != "":
        cmd.extend(["--corr_threshold", f"{wf_corr:g}"])
    if action_params:
        cmd.extend(["--action_params", action_params])
    ajson = str(form.get("action_profile_json") or "")
    if ajson:
        try:
            load_action_profile_json(ajson)
        except (ValueError, OSError) as exc:
            raise FormError("wf_copy", f"Invalid action JSON: {exc}")
        cmd.extend(["--action_profile_json", ajson])
    if reward_overrides:
        cmd.extend(["--reward_overrides", reward_overrides])
    if reward_formula:
        cmd.extend(["--reward_formula", reward_formula])
    numeric = ("learning_rate", "clip_range", "ent_coef", "n_steps", "n_epochs", "batch_size", "gamma", "gae_lambda", "vf_coef")
    try:
        commands.validate_ppo({k: float(cmd[cmd.index("--" + k) + 1]) for k in numeric})
    except (FormError, ValueError) as exc:
        raise FormError("wf_ppo", getattr(exc, "message", None) or f"Invalid training settings: {exc}")
    return {"cmd": cmd, "name": output_name}


# ------------------------------------------------------------------ fine-tune
def ft_defaults():
    return {"base": "", "old_csv": "", "new_csv": "", "mode": "mixed", "mix": "0.3", "lr": "1e-4", "steps": "50000", "name": "rl_prod_v2"}


def build_ft_command(form, python=None):
    """CTk _run_finetune."""
    python = python or sys.executable
    base, old, new = (str(form.get(k, "")).strip() for k in ("base", "old_csv", "new_csv"))
    for key, val in (("base", base), ("old_csv", old), ("new_csv", new)):
        if val in (NONE, ""):
            raise FormError(key, "Please select all 3: base model + old + new CSV")
    steps = commands._int(form, "steps", "Fine-tune Steps")
    try:
        mix_ratio = input_fraction(form.get("mix", ""), "Mix ratio", allow_one=False)
    except ValueError as exc:
        raise FormError("mix", str(exc))
    try:
        lr = input_number(form.get("lr", ""), "Learning rate", 0)
        if lr == 0:
            raise ValueError("Learning rate must be > 0")
    except ValueError as exc:
        raise FormError("lr", str(exc))
    output_name = _clean_name(form.get("name", ""))
    if not output_name:
        raise FormError("name", "New Model Name is empty after cleanup.")
    if output_name.casefold() == base.casefold():
        raise FormError("name", "New Model Name must differ from the base model. Try adding _v2 or _ft.")
    mode = str(form.get("mode", "mixed")).split()[0]
    cmd = [python, "rl_finetune.py", base, "--old_csv", old, "--new_csv", new, "--mode", mode,
           "--mix_ratio", str(mix_ratio), "--lr", str(lr), "--steps", str(steps), "--name", output_name]
    return {"cmd": cmd, "name": output_name}
