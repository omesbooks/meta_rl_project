"""Python side of the web GUI: one method per user action, exposed to the page as
window.pywebview.api. Methods take and return plain JSON-able values.

Long work never blocks the bridge: training goes through Runner, and a pump thread
forwards stdout as event batches (`emit`). `emit` is injected so tests run headless.
"""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from webgui import commands, logparse
from webgui.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
VIZ_MODES = {"monitor": "Monitor (verdicts)", "lite": "Lite (low CPU)", "cinematic": "Cinematic", "dashboard": "Dashboard (lowest CPU)"}


class Api:
    """Exposed to pywebview: keep attributes private (leading underscore) so the bridge
    does not try to serialise the runner/threads."""

    def __init__(self, root=ROOT, emit=None, python=None, set_title=None, settings_dir=None):
        self._root = Path(root)
        self._set_title = set_title or (lambda title: None)
        # where branding_config.json / api_keys.json / web_settings.json live (tests use a temp dir)
        self._settings_dir = Path(settings_dir) if settings_dir else None
        self._model_rows = {}
        self._emit = emit or (lambda events: None)
        self._python = python or sys.executable
        self._runner = Runner(self._root)
        self._pending = {}                    # review token -> {"cmd", "sha", "steps", "name"}
        self._viz_proc = None
        self._viz_run_dir = None
        self._viz_mode = "monitor"
        self._train_total = None
        self._pump_on = False

    # ------------------------------------------------------------------ meta
    def hello(self):
        return {"python": sys.version.split()[0], "root": str(self._root)}

    def catalog(self):
        from action_profiles import ACTION_PARAM_SPECS, ACTION_PROFILES
        from metric_health import METRIC_INFO
        from reward_profiles import REWARD_PARAM_SPECS, REWARD_PROFILES
        clean = lambda d: {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
        return {
            "defaults": commands.form_defaults(),
            "algos": commands.ALGOS, "reward_modes": commands.REWARD_MODES, "presets": commands.PPO_PRESETS,
            "viz_modes": VIZ_MODES,
            "reward_profiles": [{"key": k, "label": v["label"], "description": v.get("description", ""), "params": v["params"]} for k, v in REWARD_PROFILES.items()],
            "reward_specs": [clean(s) for s in REWARD_PARAM_SPECS],
            "action_profiles": [{"key": k, "label": v["label"], "description": v.get("description", ""), "params": v["params"],
                                 "actions": [a["name"] if isinstance(a, dict) else a for a in v.get("actions", [])]} for k, v in ACTION_PROFILES.items()],
            "action_specs": [clean(s) for s in ACTION_PARAM_SPECS],
            "metrics": {k: {"label": v.get("label", k), "good": list(v["good"]), "warn": list(v["warn"]), "icon": v.get("icon", "")} for k, v in METRIC_INFO.items()},
        }

    # ------------------------------------------------------------------ data
    def list_csvs(self):
        return sorted(p.name for p in self._root.glob("*.csv"))

    def _csv_path(self, csv):
        p = Path(csv)
        return p if p.is_absolute() else self._root / p

    def csv_info(self, csv, train_pct="85", window="10"):
        import pandas as pd
        p = self._csv_path(csv)
        if not p.is_file():
            return {"ok": False, "error": f"{csv} not found"}
        if p.name.endswith("_m1.csv"):
            return {"ok": False, "error": f"{p.name} is raw M1 execution data for backtests, not a feature dataset"}
        try:
            head = pd.read_csv(p, nrows=1)
            with p.open("rb") as f:
                rows = sum(1 for _ in f) - 1
        except Exception as exc:
            return {"ok": False, "error": f"error reading: {exc}"}
        params = p.with_suffix(".params.json")
        split = []
        if "timestamp" in head.columns:
            try:
                from ui_values import fraction
                ts = pd.read_csv(p, usecols=["timestamp"])["timestamp"].astype(str).tolist()
                pct = fraction(train_pct, "Train %", allow_zero=False)
                cut = max(0, min(int(len(ts) * pct), len(ts)))
                mid = cut + (len(ts) - cut) // 2
                rng = lambda a, b: f"{ts[a][:16]} → {ts[b][:16]} ({b - a + 1:,} rows)" if 0 <= a <= b < len(ts) else "–"
                split = [["Train", rng(0, cut - 1)]]
                split += [["Validation", rng(cut, mid - 1)], ["Test", rng(mid, len(ts) - 1)]] if cut < len(ts) else [["Test", "none — in-sample diagnostic only"]]
            except Exception as exc:
                split = [["split", f"preview failed: {exc}"]]
        gates = [c for c in head.columns if c.lower().endswith("_sig") or c.lower().startswith("gate")]
        return {"ok": True, "name": p.name, "path": str(p), "rows": rows, "cols": len(head.columns),
                "params": params.is_file(), "split": split, "gate_columns": gates,
                "suggest": commands.suggest_steps(rows, train_pct, window)}

    # ------------------------------------------------------------------ dialogs
    def _window(self):
        import webview
        return webview.windows[0] if webview.windows else None

    def _open_dialog(self, folder, types):
        import webview
        w = self._window()
        if w is None:
            return None
        r = w.create_file_dialog(webview.OPEN_DIALOG, directory=str(folder), file_types=types)
        return r[0] if r else None

    def browse_csv(self):
        return self._open_dialog(self._root, ("CSV files (*.csv)", "All files (*.*)"))

    def attach_params(self, csv):
        src = self._open_dialog(self._root, ("Params JSON (*.json)", "All files (*.*)"))
        if not src:
            return {"ok": False, "cancelled": True}
        try:
            if not isinstance(json.loads(Path(src).read_text(encoding="utf-8-sig")), dict):
                return {"ok": False, "error": "the file must contain a JSON object"}
        except Exception as exc:
            return {"ok": False, "error": f"invalid JSON: {exc}"}
        dst = self._csv_path(csv).with_suffix(".params.json")
        shutil.copyfile(src, dst)
        return {"ok": True, "path": str(dst)}

    def load_reward_json(self):
        from reward_profiles import DEFAULT_REWARD_PROFILE_CONFIG_DIR, load_reward_profile_json
        DEFAULT_REWARD_PROFILE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = self._open_dialog(DEFAULT_REWARD_PROFILE_CONFIG_DIR, ("Reward profile JSON (*.json)", "All files (*.*)"))
        if not path:
            return {"ok": False, "cancelled": True}
        try:
            loaded = load_reward_profile_json(path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": path, "name": loaded["name"], "base_profile": loaded["base_profile"],
                "overrides": loaded["overrides"], "formula": loaded.get("formula", "")}

    def save_reward_json(self, form):
        import webview
        from reward_profiles import DEFAULT_REWARD_PROFILE_CONFIG_DIR, reward_profile_json_payload
        try:
            overrides = commands.reward_overrides(form, compare_loaded_json=False)
            formula = commands.reward_formula(form)
        except commands.FormError as exc:
            return {"ok": False, "field": exc.field, "error": exc.message}
        DEFAULT_REWARD_PROFILE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        w = self._window()
        r = w.create_file_dialog(webview.SAVE_DIALOG, directory=str(DEFAULT_REWARD_PROFILE_CONFIG_DIR),
                                 save_filename=f"{form.get('reward_profile', 'balanced')}_custom.json") if w else None
        path = r if isinstance(r, str) else (r[0] if r else None)
        if not path:
            return {"ok": False, "cancelled": True}
        payload = reward_profile_json_payload(Path(path).stem, form.get("reward_profile", "balanced"), overrides,
                                              formula=formula, notes="Saved from MetaFXClub RL Studio (web) Train page.")
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "path": path, "overrides": overrides, "has_formula": bool(formula)}

    def load_action_json(self):
        from action_profiles import DEFAULT_ACTION_PROFILE_CONFIG_DIR, load_action_profile_json
        DEFAULT_ACTION_PROFILE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = self._open_dialog(DEFAULT_ACTION_PROFILE_CONFIG_DIR, ("Action profile JSON (*.json)", "All files (*.*)"))
        if not path:
            return {"ok": False, "cancelled": True}
        try:
            loaded = load_action_profile_json(path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": path, "name": loaded["name"], "base_profile": loaded["base_profile"], "params": loaded["params"]}

    def save_action_json(self, form):
        import webview
        from action_profiles import DEFAULT_ACTION_PROFILE_CONFIG_DIR, action_profile_json_payload
        try:
            params = commands.action_params(form, compare_loaded_json=False)
        except commands.FormError as exc:
            return {"ok": False, "field": exc.field, "error": exc.message}
        DEFAULT_ACTION_PROFILE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        w = self._window()
        r = w.create_file_dialog(webview.SAVE_DIALOG, directory=str(DEFAULT_ACTION_PROFILE_CONFIG_DIR),
                                 save_filename=f"{form.get('action_profile', 'basic_4')}_custom.json") if w else None
        path = r if isinstance(r, str) else (r[0] if r else None)
        if not path:
            return {"ok": False, "cancelled": True}
        payload = action_profile_json_payload(Path(path).stem, form.get("action_profile", "basic_4"), params=params,
                                              notes="Saved from MetaFXClub RL Studio (web) Train page.")
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "path": path, "params": params}

    def validate_formula(self, text):
        from reward_formula import validate_reward_formula
        try:
            validate_reward_formula(text)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------ training
    def busy(self):
        return self._runner.is_running()

    def review_training(self, form):
        """Validate + prepare exactly like the CTk confirm step; nothing is launched here."""
        if self._runner.is_running():
            return {"ok": False, "error": "Already training"}
        try:
            built = commands.build_train_command(form, python=self._python)
        except commands.FormError as exc:
            return {"ok": False, "field": exc.field, "error": exc.message}
        from artifact_paths import find_model_path
        from rl_train import build_parser, prepare_training
        try:
            prepared = prepare_training(build_parser().parse_args(list(built["cmd"][2:])))
        except (Exception, SystemExit) as exc:
            return {"ok": False, "error": f"Cannot start training: {exc}"}
        recipe, sha = prepared["recipe"], prepared["recipe_sha256"]
        existing = find_model_path(built["name"], "auto")
        token = secrets.token_hex(8)
        self._pending = {token: {"cmd": built["cmd"], "sha": sha, "steps": built["steps"], "name": built["name"],
                                 "viz_mode": form.get("viz_mode", "monitor")}}
        st = recipe["settings"]
        period = lambda d: f"{str(d['start'])[:16].replace('T', ' ')} → {str(d['end'])[:16].replace('T', ' ')}" if isinstance(d, dict) and d.get("start") else "none"
        summary = [
            ["Model", f"{st['name']} | {st['algo'].upper()}"],
            ["Features after selection", str(len(recipe["features"]))],
            ["Train / Validation / Test rows", f"{recipe['train_rows']} / {recipe['validation_rows']} / {recipe['test_rows']}"],
            ["Train dates", period(recipe["train_period"])],
            ["Validation dates", period(recipe["validation_period"])],
            ["Test dates", period(recipe["test_period"])],
            ["Window · Steps · Episode · Network", f"{st['window']} · {st['steps']} · {recipe['effective_episode_steps']} · {recipe['net_arch']}"],
            ["Reward mode", f"{st['reward_mode']} (profile/formula active: {st['reward_mode'] != 'mtm'})"],
            ["Collector params", "available" if Path(st["csv"]).with_suffix(".params.json").is_file() else "MISSING"],
            ["Evaluation", str(recipe["evaluation_role"])],
            ["Output", f"{recipe['output_root']}/<run_id>/"],
        ]
        return {"ok": True, "token": token, "summary": summary, "existing": str(existing) if existing else "",
                "recipe_sha256": sha, "warnings": built["warnings"], "command": " ".join(built["cmd"][1:]),
                "recipe": json.loads(json.dumps(recipe, default=str))}

    def confirm_training(self, token):
        job = self._pending.pop(token, None)
        if not job:
            return {"ok": False, "error": "review expired — review the training again"}
        cmd = list(job["cmd"]) + ["--expected_recipe_sha256", job["sha"]]
        self._train_total = job["steps"]
        self._viz_run_dir = None
        self._viz_mode = job.get("viz_mode") or "monitor"
        if not self._runner.start(cmd, "train"):
            return {"ok": False, "error": "Already training"}
        self.start_pump()
        self._emit([{"type": "start", "page": "train", "command": " ".join(cmd[1:]), "name": job["name"], "total": job["steps"]}])
        return {"ok": True}

    def stop_training(self):
        if self._runner.is_running():
            self._runner.stop()
            return {"ok": True}
        return {"ok": False}

    # ------------------------------------------------------------------ event pump
    def start_pump(self):
        if self._pump_on:
            return
        self._pump_on = True
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while self._pump_on:
            batch = []
            for kind, page, data in self._runner.drain():
                if kind == "line":
                    ev = logparse.event(data, page, self._train_total)
                    if ev.get("viz"):
                        self._on_viz(ev["viz"])
                    batch.append(ev)
                else:
                    batch.append({"type": "done", "page": page, **data})
            if batch:
                try:
                    self._emit(batch)
                except Exception:
                    pass
            time.sleep(0.1)

    # ------------------------------------------------------------------ live visual
    def _viz_running(self):
        return self._viz_proc is not None and self._viz_proc.poll() is None

    def _launch_viz(self, args):
        if self._viz_running():
            try:
                self._viz_proc.terminate()
            except Exception:
                pass
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._viz_proc = subprocess.Popen([self._python, str(self._root / "tools" / "viz" / "live_viewer.py"), *args,
                                           "--mode", self._viz_mode], cwd=str(self._root), creationflags=flags)

    def _on_viz(self, v):
        viz_dir = Path(v["path"]).parent
        if self._viz_run_dir != viz_dir:
            self._viz_run_dir = viz_dir
            self._launch_viz(["--run_dir", str(viz_dir)])

    def open_visual(self, name, mode="monitor"):
        self._viz_mode = mode if mode in VIZ_MODES else "monitor"
        if self._viz_run_dir and Path(self._viz_run_dir).exists():
            self._launch_viz(["--run_dir", str(self._viz_run_dir)])
            return {"ok": True}
        from artifact_paths import model_dir
        try:
            viz = model_dir(name) / "viz"
        except Exception:
            viz = None
        if viz is not None and (viz.exists()):
            self._launch_viz(["--run_dir", str(viz)])
            return {"ok": True}
        return {"ok": False, "error": f"No visual for '{name}' yet — tick 'Live visual' before training."}

    # ------------------------------------------------------------------ models
    def list_models(self):
        """Models table: same rows + cell text as the CTk Models page (model_results)."""
        import model_results as mr
        rows = mr.collect_model_rows(self._root)
        self._model_rows = {r["iid"]: r for r in rows}
        out = []
        for r in rows:
            out.append({
                "iid": r["iid"], "model": r["model"], "source": r["source"],
                "cells": list(mr.table_values(r, self._root)),
                # raw numbers so the page can sort; inf/NaN become null in to_json
                "sort": [r["model"], r["source"], r.get("trades"), r.get("win_rate"), r.get("profit_factor"),
                         r.get("return_pct"), r.get("max_dd"), r.get("realism") or "", bool(r.get("chart_path")), None],
                "pf_inf": r.get("profit_factor") == float("inf"),
                "has_chart": bool(r.get("chart_path")),
            })
        return {"columns": list(mr.COLUMNS), "headings": [mr.HEADINGS[c] for c in mr.COLUMNS], "rows": out}

    def _model_row(self, iid):
        row = self._model_rows.get(iid)
        if row is None:
            self.list_models()
            row = self._model_rows.get(iid)
        return row

    def _row_equity(self, row):
        import model_results as mr
        path = row.get("equity_path")
        if not path and row.get("source") != "env backtest":
            path = mr.equity_path_for_model(row["model"], self._root)
        return Path(path) if path else None

    def model_equity(self, iid):
        """Equity PNG of one row as a data URL (thumbnail, like the CTk viewer)."""
        import base64
        import io
        row = self._model_row(iid)
        if not row:
            return {"ok": False, "error": "model row not found - refresh the table"}
        path = self._row_equity(row)
        if not path or not path.exists():
            if row.get("source") == "model only":
                msg = (f"'{row['model']}' เทรนแล้วแต่ยังไม่เคย backtest - ไปหน้า Backtest เลือก model นี้ + dataset ที่เทรนมา แล้วรัน "
                       f"(แถวที่ Result source = 'live backtest' เท่านั้นที่มี equity curve)")
            else:
                msg = f"No equity image found for {row['model']}. Run backtest first."
            return {"ok": False, "missing": True, "error": msg}
        try:
            from PIL import Image
            img = Image.open(path)
            img.thumbnail((1400, 700), Image.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG", optimize=True)
        except Exception as exc:
            return {"ok": False, "error": f"Equity image: {path.name} - preview failed: {exc}"}
        return {"ok": True, "name": path.name, "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")}

    def _startfile(self, path):
        import os
        os.startfile(str(path))                            # Windows only, like rl_app

    def open_model_file(self, iid, kind):
        row = self._model_row(iid)
        if not row:
            return {"ok": False, "error": "Select a model result first."}
        if kind == "equity":
            path = self._row_equity(row)
            if not path:
                return {"ok": False, "error": f"No equity image found for {row['model']}."}
        elif kind == "chart":
            path = row.get("chart_path")
            if not path:
                return {"ok": False, "error": f"No HTML chart found for {row['model']}."}
        else:
            return {"ok": False, "error": f"unknown file kind {kind!r}"}
        self._startfile(path)
        return {"ok": True, "path": str(path)}

    def export_models_pdf(self, iid=""):
        import model_results as mr
        rows = mr.collect_model_rows(self._root)
        if not rows:
            return {"ok": False, "error": "No model/backtest results found to export."}
        selected = next((r for r in rows if r["iid"] == iid), rows[0])
        out = mr.export_report_pdf(rows, selected, self._root)
        try:
            self._startfile(out)
        except Exception:
            pass
        return {"ok": True, "path": str(out)}

    # ------------------------------------------------------------------ settings
    def _sfile(self, name):
        import app_settings as s
        default = {"branding": s.BRANDING_CONFIG_FILE, "keys": s.API_KEYS_FILE, "web": s.WEB_SETTINGS_FILE}[name]
        return (self._settings_dir / default.name) if self._settings_dir else default

    def _keys(self):
        import app_settings as s
        return s.load_api_keys(self._sfile("keys"))

    def _key_status(self):
        key = str(self._keys().get("gemini") or "")
        return {"gemini": {"set": bool(key), "tail": key[-4:] if len(key) >= 8 else ""}}

    def settings_get(self):
        """Never returns a stored key - only whether one is set and its last 4 characters."""
        import app_settings as s
        eff = s.effective_branding(self._sfile("branding"))
        return {
            "branding": eff["branding"], "colors": eff["colors"], "saved": eff["saved"],
            "text_fields": [list(f) for f in s.BRANDING_TEXT_FIELDS],
            "color_fields": [list(f) for f in s.COLOR_FIELDS],
            "presets": s.COLOR_PRESETS,
            "web": s.load_web_settings(self._sfile("web")), "web_themes": s.WEB_THEMES,
            "keys": self._key_status(),
            "work_dir": str(self._root), "config_path": str(self._sfile("branding")),
            "python": sys.version.split()[0], "logo": self._logo_data(eff["branding"].get("logo_image", "")),
        }

    def _logo_data(self, rel):
        """Sidebar logo as a small data URL ('' when unset or unreadable), path relative to the root like rl_app."""
        if not rel:
            return ""
        path = Path(rel) if Path(rel).is_absolute() else self._root / rel
        try:
            import base64
            import io
            from PIL import Image
            img = Image.open(path)
            img.thumbnail((96, 96))
            buf = io.BytesIO()
            img.convert("RGBA").save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return ""

    def save_branding(self, form):
        """form = {"branding": {key: text}, "colors": {KEY: "#hex"}} - the CTk Save button."""
        import app_settings as s
        branding = {k: str((form.get("branding") or {}).get(k, "")).strip() for k, _l, _h in s.BRANDING_TEXT_FIELDS}
        colors = {}
        for key, label, _h in s.COLOR_FIELDS:
            val = str((form.get("colors") or {}).get(key, "")).strip()
            if not s.is_hex_color(val):
                return {"ok": False, "field": f"color_{key}", "error": f"{label} colour must be #rgb or #rrggbb (got {val!r})"}
            colors[key] = val
        s.save_branding_config(branding, colors, self._sfile("branding"))
        if branding.get("window_title"):
            self._set_title(branding["window_title"])
        return {"ok": True, "path": str(self._sfile("branding")), **self.settings_get()}

    def reset_branding(self):
        path = self._sfile("branding")
        if path.exists():
            path.unlink()
        return {"ok": True, **self.settings_get()}

    def open_branding_config(self):
        path = self._sfile("branding")
        if not path.exists():
            return {"ok": False, "error": "ยังไม่มีไฟล์ config - กด Save ก่อน เพื่อสร้างไฟล์"}
        self._startfile(path)
        return {"ok": True}

    def save_api_key(self, value):
        import app_settings as s
        value = str(value or "").strip()
        if not value:
            return {"ok": False, "field": "gemini_key", "error": "Paste a key first (use Clear to remove the saved one)."}
        if any(c.isspace() for c in value) or len(value) < 20:
            return {"ok": False, "field": "gemini_key", "error": "That does not look like an API key (no spaces, 20+ characters)."}
        keys = self._keys()
        keys["gemini"] = value
        s.save_api_keys(keys, self._sfile("keys"))
        return {"ok": True, "keys": self._key_status()}

    def clear_api_key(self):
        import app_settings as s
        keys = self._keys()
        keys["gemini"] = ""
        s.save_api_keys(keys, self._sfile("keys"))
        return {"ok": True, "keys": self._key_status()}

    def open_gemini_page(self):
        import webbrowser
        webbrowser.open("https://aistudio.google.com/app/apikey")
        return {"ok": True}

    def save_web_settings(self, values):
        import app_settings as s
        return {"ok": True, "web": s.save_web_settings(values or {}, self._sfile("web"))}

    # ------------------------------------------------------------------ app
    def open_classic(self, page=""):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([self._python, str(self._root / "rl_app.py")], cwd=str(self._root), creationflags=flags)
        return {"ok": True}

    def shutdown(self):
        self._pump_on = False
        self._runner.stop()
        if self._viz_running():
            try:
                self._viz_proc.terminate()
            except Exception:
                pass


class JsBridge:
    """What the window actually exposes. pywebview 5.4 drops a return value that carries
    some non-ASCII text (seen: the catalog's Thai descriptions + emoji never resolved in
    JS), so every call returns ASCII-escaped JSON text and app.js JSON.parses it. Api
    itself keeps returning plain dicts (tests, other callers). pywebview only exposes real
    methods of the class, so the wrappers are generated onto the class below."""

    def __init__(self, api):
        self._api = api


def to_json(value):
    """JSON text the page can always parse: ASCII only, and +-inf / NaN become null
    (METRIC_INFO uses inf bounds; json.dumps would emit the invalid token Infinity)."""
    import math

    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(value), ensure_ascii=True, default=str)


def _bridge_method(name):
    def call(self, *args):
        try:
            result = getattr(self._api, name)(*args)
        except Exception as exc:                           # never leave a JS promise hanging
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return to_json(result)
    call.__name__ = name
    return call


for _name, _fn in list(vars(Api).items()):
    if not _name.startswith("_") and callable(_fn):
        setattr(JsBridge, _name, _bridge_method(_name))
