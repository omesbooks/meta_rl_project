"""Demo: what rl_app.py would look like as a web GUI in a pywebview window.

    python tools/viz/gui_demo.py

Architecture shown by the demo (the point of the exercise):
  * the page (tools/viz/gui_demo/index.html) is plain HTML/CSS/JS — real glass, real
    layout freedom, same Clear Liquid theme as the Monitor
  * Python stays the backend: this file exposes an `Api` class to the page through
    window.pywebview.api (list_csvs / csv_info / list_models read the real project;
    start_training streams log lines back with window.evaluate_js). In a real port the
    training call would launch rl_train.py as a subprocess exactly like ProcessRunner
    does today and forward its stdout line by line
  * one process, no server, no browser install: pywebview uses the Edge WebView2
    runtime already on the machine (same as the live visual)
"""
from __future__ import annotations

import io
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import webview

if __name__ == "__main__" and sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_suffix("") / "index.html"


class Api:
    """Exposed to the page as window.pywebview.api (plain methods only: pywebview introspects
    the object, so it must not hold a Window - use webview.windows[0] instead)."""

    def __init__(self):
        self.stop = threading.Event()

    @property
    def _win(self):
        return webview.windows[0]

    def hello(self):
        return f"python {sys.version.split()[0]} · pywebview {getattr(webview, '__version__', '?')}"

    def list_csvs(self):
        return sorted(p.name for p in ROOT.glob("*.csv") if not p.name.endswith("_m1.csv"))[:60]

    def csv_info(self, name):
        p = ROOT / name
        if not p.is_file():
            return {"hint": "file not found", "suggested": 200000}
        with p.open("rb") as f:
            rows = sum(1 for _ in f) - 1
        train = int(rows * 0.7)
        return {"hint": f"{rows:,} rows · train {train:,} rows (70%) · {p.stat().st_size / 1e6:.1f} MB", "suggested": max(10000, train * 5)}

    def list_models(self):
        out = []
        for home in sorted((ROOT / "artifacts" / "models").glob("*")):
            runs = sorted((home / "runs").glob("*/"), key=lambda r: r.stat().st_mtime) if (home / "runs").is_dir() else []
            meta = {}
            try:                                              # the run's train meta, as the real app resolves it
                sys.path.insert(0, str(ROOT))
                from artifact_paths import train_meta_path
                mp = train_meta_path(home.name)
                if mp.exists():
                    meta = json.loads(mp.read_text(encoding="utf-8-sig")).get("hyperparameters", {})
            except Exception:
                pass
            size = sum(f.stat().st_size for f in home.rglob("*") if f.is_file())
            out.append({"name": home.name, "run": runs[-1].name[:8] if runs else "–", "steps": meta.get("steps") or meta.get("total_timesteps") or 0,
                        "gate": meta.get("entry_gate", ""), "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(home.stat().st_mtime)), "mb": round(size / 1e6, 1)})
        return sorted(out, key=lambda r: r["when"], reverse=True)[:40]

    def browse_csv(self):
        r = self._win.create_file_dialog(webview.OPEN_DIALOG, directory=str(ROOT), file_types=("CSV (*.csv)",))
        return Path(r[0]).name if r else None

    def start_training(self, cfg):
        """Simulated run: streams SB3-shaped lines so the page's parser/pills can be seen working.
        A real port would Popen rl_train.py here and forward stdout."""
        self.stop.clear()
        threading.Thread(target=self._simulate, args=(cfg,), daemon=True).start()
        return True

    def _emit(self, line, step=None, total=None, metrics=None):
        self._win.evaluate_js(f"window.onLogLine({json.dumps(line)}, {json.dumps(step)}, {json.dumps(total)}, {json.dumps(metrics)})")

    def _simulate(self, cfg):
        total = int(cfg.get("steps") or 200000); n_steps = 2048
        self._emit(f"$ python rl_train.py {cfg.get('csv')} --steps {total} --name {cfg.get('name')}" + (f" --entry_gate {cfg['gate']}" if cfg.get("gate") else "") + (" --viz" if cfg.get("viz") else ""))
        self._emit(f"[features] 41 -> 41; dropped: []"); self._emit(f"[split] train: 50,901 | validation: 10,908 | test: 10,908")
        rew, ent = -1.2, 1.38
        for step in range(n_steps, total + 1, n_steps):
            if self.stop.is_set():
                self._emit("[stop] training cancelled by the user"); break
            rew += random.gauss(0.03, 0.08); ent = max(0.5, ent - 0.004 + random.gauss(0, 0.01))
            m = {"ep_rew_mean": rew, "approx_kl": abs(random.gauss(0.012, 0.004)), "clip_fraction": abs(random.gauss(0.18, 0.04)),
                 "explained_variance": min(0.9, -0.3 + step / total), "entropy": ent, "value_loss": abs(random.gauss(0.02, 0.005))}
            self._emit(f"| ep_rew_mean {rew:+.3f} | approx_kl {m['approx_kl']:.4f} | clip {m['clip_fraction']:.3f} | entropy {ent:.3f} | total_timesteps {step}", step, total, m)
            if cfg.get("viz") and step == n_steps:
                self._emit(f"VIZ_ROLLOUT 0 {step} artifacts/models/{cfg.get('name')}/runs/<id>/viz/rollout_latest.json")
            if step % 10000 < n_steps:
                self._emit(f"VIZ_SNAPSHOT {step // 10000 - 1} {step} .../viz/snap_{step // 10000 - 1:03d}.json  (eval reward {rew * 0.6:+.2f})")
            time.sleep(0.12)
        else:
            self._emit(f"[saved] artifacts/models/{cfg.get('name')}/{cfg.get('name')}.zip  (demo — nothing was written)")
        self._win.evaluate_js("window.onDone()")

    def stop_training(self):
        self.stop.set(); return True

    def open_visual(self, name, mode):
        m = {"Monitor (verdicts)": "monitor", "Lite (low CPU)": "lite", "Cinematic": "cinematic", "Dashboard": "dashboard"}.get(mode, "monitor")
        subprocess.Popen([sys.executable, str(ROOT / "tools" / "viz" / "live_viewer.py"), "--model", name, "--mode", m])
        return True


def main():
    api = Api()
    webview.create_window("RL Trading Studio — web GUI demo", PAGE.as_uri(), js_api=api, width=1480, height=960, min_size=(1100, 700))
    webview.start()


if __name__ == "__main__":
    main()
