"""Pipeline / Walk-Forward / Fine-tune rules shared by rl_app.py and the web GUI.

Moved verbatim out of RLTradingStudio so both front ends build the same commands:
- Pipeline: the train command (_pipeline_train_command), the backtest range resolved at
  review time (_resolve_pipeline_preview), the backtest + chart stage commands
  (_pipeline_worker), the Train/Validation/Test split preview and the Backtest-CSV pick
  that follows the Train CSV.
- Walk-Forward: WF_PROGRESS parsing and the ROBUST / MOSTLY ROBUST / UNSTABLE / NOT ROBUST
  verdict read from rl_walkforward.py stdout, and the chart lookup.
- Fine-tune: the inherited-recipe hint and the per-mode hint.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


# ------------------------------------------------------------------ pipeline
def train_command(python, train_csv, bt_csv, model_name, steps, window, train_pct, hparams):
    train_cmd = [
        python, "rl_train.py", train_csv,
        "--steps", str(steps),
        "--window", str(window),
        "--name", model_name,
        "--train_pct", str(train_pct),
        "--algo", "ppo",
        "--reward_mode", hparams["reward_mode"],
        "--reward_profile", hparams["reward_profile"],
        "--reward_overrides", hparams.get("reward_overrides", ""),
        "--action_profile", hparams["action_profile"],
        "--action_params", hparams.get("action_params", ""),
        "--max_hold", hparams["max_hold"],
        "--ep_len", hparams["ep_len"],
        "--net_arch", hparams["net_arch"],
        "--learning_rate", hparams["learning_rate"],
        "--clip_range", hparams["clip_range"],
        "--ent_coef", hparams["ent_coef"],
        "--n_steps", hparams["n_steps"],
        "--n_epochs", hparams["n_epochs"],
        "--batch_size", hparams["batch_size"],
        "--gamma", hparams["gamma"],
        "--gae_lambda", hparams["gae_lambda"],
        "--vf_coef", hparams["vf_coef"],
    ]
    if Path(bt_csv).resolve() != Path(train_csv).resolve():
        train_cmd.extend(["--eval_csv", bt_csv])
    if hparams.get("reward_formula"):
        train_cmd.extend(["--reward_formula", hparams["reward_formula"]])
    return train_cmd


def resolve_backtest(recipe, context, frame, find_m1):
    """Fix the pipeline's backtest range at review time (CTk _resolve_pipeline_preview).
    `frame` = the validated backtest CSV; `find_m1(csv)` = companion-M1 lookup."""
    import pandas as pd
    if context["bt_start_auto"]:
        period = recipe["test_period"]
        skip = int((frame.timestamp < pd.Timestamp(period["start"])).sum()) if period else 0
        start = skip / len(frame)
        if int(start * len(frame)) < skip:
            start = math.nextafter(start, 1.0)
    else:
        start = float(context["bt_start"])
        skip = int(len(frame) * start)
    if len(frame) - skip <= recipe["settings"]["window"] + 2:
        raise ValueError("Backtest range is too short for the selected Window")
    context["resolved_backtest"] = {
        "start": start, "rows": len(frame) - skip,
        "period": {"start": str(frame.timestamp.iloc[skip]), "end": str(frame.timestamp.iloc[-1])},
        "m1_csv": find_m1(context["backtest_csv"]) if context["use_m1"] else None,
        "stop_slippage": 0.0001, "intrabar": "pessimistic", "random_baseline": 20,
        "mc": 1000, "swap_long": 0, "swap_short": 0,
    }
    return context["resolved_backtest"]


def backtest_command(python, model_name, bt_csv, conf, mode, window, hparams, m1_file):
    backtest_cmd = [
        python, "backtest_live.py", model_name, bt_csv,
        "--conf", str(conf),
        "--risk", hparams.get("risk", "0.01"),
        "--max_positions", hparams.get("max_positions", "1"),
        "--max_hold", hparams["max_hold"],
        "--atr_sl", hparams.get("atr_sl", "2.0"),
        "--atr_tp", hparams.get("atr_tp", "4.0"),
        "--window", str(window),
        "--start", hparams.get("bt_start", "0"),
        "--mode", mode,
        "--stop_slippage", "0.0001",
    ]
    if m1_file:
        backtest_cmd += ["--m1_csv", m1_file]
    # Statistical validation — same defaults as the Backtest page
    # (--mc 1000 is already the CLI default; baseline is opt-in on CLI)
    backtest_cmd += ["--random_baseline", "20"]
    return backtest_cmd


def chart_command(python, model_name, bt_csv):
    return [python, "backtest_chart.py", model_name, bt_csv, "--limit", "5000"]


def split_preview(ts, train_pct_raw):
    """Train / Validation / Test date ranges for the pipeline's Train pct -> (text, level)."""
    from backtest_report import format_range_line
    from ui_values import fraction
    if ts is None or len(ts) == 0:
        return "No valid timestamp rows", "warn"
    try:
        train_pct = fraction(train_pct_raw, "pipe_train_pct", allow_zero=False, allow_one=True)
    except ValueError as exc:
        return str(exc), "warn"
    split = max(0, min(int(len(ts) * train_pct), len(ts)))
    train_line = format_range_line("Train", ts, 0, split - 1)
    middle = split + (len(ts) - split) // 2
    tail_line = (format_range_line("Validation", ts, split, middle - 1) + "\n" +
                 format_range_line("Test", ts, middle, len(ts) - 1))
    if split == len(ts):
        tail_line = "No unseen Test: in-sample diagnostic only"
    return f"{train_line}\n{tail_line}", ""


def backtest_csv_for(train_csv, work_dir=ROOT):
    """The Backtest CSV the pipeline picks when the Train CSV changes: <stem>_test for a
    <stem>_train file when it exists, else the same file. None = leave the choice alone."""
    work_dir = Path(work_dir)
    train_path = Path(train_csv)
    candidates = []
    stem = train_path.stem
    if stem.endswith("_train"):
        candidates.append(train_path.with_stem(stem[:-len("_train")] + "_test").name)
    candidates.append(train_csv)
    for name in candidates:
        if (work_dir / name).exists():
            return name
    return None


# ------------------------------------------------------------------ walk-forward
def parse_wf_progress(line):
    m = re.match(r'WF_PROGRESS\s+(\d[\d,]*)\s+(\d[\d,]*)(?:\s+(.*))?$', line)
    if not m:
        return None
    try:
        cur = int(m.group(1).replace(',', ''))
        total = int(m.group(2).replace(',', ''))
    except ValueError:
        return None
    label = (m.group(3) or "").strip()
    return cur, total, label


def wf_verdict(line):
    """(icon, title, tone, sub) when the line carries the walk-forward verdict, else None.
    Order matters: 'MOSTLY ROBUST' / 'NOT ROBUST' contain 'ROBUST'."""
    if 'MOSTLY ROBUST' in line:
        match = re.search(r'(\d+/\d+)\s+windows', line)
        suffix = f" ({match.group(1)} passed)" if match else ""
        return "🟡", "MOSTLY ROBUST", "warn", f"Most windows passed{suffix} — use with caution"
    if 'NOT ROBUST' in line:
        return "❌", "NOT ROBUST", "bad", "Most windows failed"
    if 'UNSTABLE' in line:
        return "⚠️", "UNSTABLE", "warn", "Some windows failed"
    if 'ROBUST' in line:
        return "✅", "ROBUST", "good", "All windows passed gate"
    return None


def sanitize_wf_name(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (name or "wf_prod").strip()).rstrip(". ") or "wf"


def wf_chart_path(name, work_dir=ROOT):
    """Newest <name>_chart*.png written by rl_walkforward.py (None when there is none)."""
    name = sanitize_wf_name(name)
    candidates = sorted(Path(work_dir).glob(f"{name}_chart*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


# ------------------------------------------------------------------ fine-tune
FT_MODE_HINTS = {
    "mixed": "mixed: ผสม data เก่าตาม ratio กันลืมของเดิม · LR ต่ำ (1e-4) คือหัวใจของ fine-tune — สูงไป = ลืมของเก่า (catastrophic forgetting)",
    "pure": "pure: เทรนต่อด้วย data ใหม่ล้วน (Mix Ratio ไม่ถูกใช้) — เร็วแต่เสี่ยง catastrophic forgetting ที่สุด ใช้เมื่อ regime เปลี่ยนจริงเท่านั้น",
    "replay": "replay: ใช้ old+new ทั้งหมดแบบไม่สุ่ม (Mix Ratio ไม่ถูกใช้) — กันลืมได้สุดแต่ช้าสุด และ data ใหม่มีน้ำหนักน้อยสุดเพราะโดนของเก่าท่วม",
}


def ft_recipe_hint(base):
    """The recipe a base model hands down to fine-tune -> (text, level)."""
    if base in ("(none)", "", None):
        return "เลือก base model เพื่อดูสูตรที่จะสืบทอด (reward / action / window / max hold)", ""
    try:
        from artifact_paths import train_meta_path
        mp = train_meta_path(base)
        if not mp.exists():
            return (f"⚠ ไม่พบ meta ของ {base} — จะ fine-tune ด้วยค่า default "
                    "(balanced / basic_4) ซึ่งอาจไม่ตรงกับสูตรที่ model นี้ถูกเทรนมา"), "warn"
        hp = json.loads(mp.read_text(encoding="utf-8")).get("hyperparameters", {})
        formula = (hp.get("reward_formula") or "").strip()
        parts = [
            f"reward: {hp.get('reward_mode', 'realized')}/{hp.get('reward_profile', 'balanced')}"
            + (" +formula" if formula else ""),
            f"action: {hp.get('action_profile', 'basic_4')}",
            f"window: {hp.get('window', '?')}",
            f"max hold: {hp.get('max_hold', '?')}",
        ]
        return ("สืบทอดจาก base model → " + " · ".join(parts)
                + " (fine-tune ใช้สูตรเดิมเสมอ — เปลี่ยน reward/action = ต้อง retrain)"), ""
    except Exception as e:
        return f"⚠ อ่าน meta ไม่ได้: {e}", "warn"
