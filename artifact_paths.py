"""Central paths for generated training/backtest artifacts.

New artifacts are written under:
    artifacts/models/<model>/runs/<run_id>/
The model's current.json atomically selects the published generation.

Legacy root-level files are still discoverable so older runs remain usable.
"""
from pathlib import Path
import json
import os
import uuid
from datetime import datetime, timezone


ROOT = Path(__file__).parent.resolve()
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
_RUN_OUTPUTS = {}
_READ_OUTPUTS = {}


def pin_model_generation(name):
    """Pin all paths for a CLI operation to the same published generation."""
    _READ_OUTPUTS.pop(name, None)
    _READ_OUTPUTS[name] = model_dir(name)
    meta_path = train_meta_path(name)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        algo = meta.get("hyperparameters", {}).get("algo", "ppo")
        if algo != "ppo":
            raise ValueError(f"This workflow supports PPO only; selected model uses {algo}")


def _model_home(model_name):
    if not model_name or Path(model_name).name != model_name or any(
        c in model_name for c in '<>:"/\\|?*'
    ) or model_name.endswith((".", " ")) or model_name in (".", ".."):
        raise ValueError("Model name must be a filename, not a path")
    return MODELS_DIR / model_name


def _atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ArtifactRun:
    """Publish a complete generation with one atomic pointer replacement."""

    def __init__(self, name):
        self.name = name
        self.home = _model_home(name)
        self.run_id = uuid.uuid4().hex
        self.path = self.home / "runs" / self.run_id
        self.committed = False

    def __enter__(self):
        import psutil
        self.path.mkdir(parents=True)
        self.state = dict(run_id=self.run_id, model=self.name, pid=os.getpid(),
                          pid_created=psutil.Process().create_time(),
                          status="running", started_at=datetime.now(timezone.utc).isoformat())
        self._write_state()
        _RUN_OUTPUTS[self.name] = self.path
        ensure_model_dirs(self.name)
        return self

    def _write_state(self):
        _atomic_json(self.path / "run.json", self.state)

    def publish(self, model):
        import hashlib
        import numpy as np
        import pandas as pd

        meta_file = self.path / f"{self.name}.train.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        norm = pd.read_csv(self.path / f"{self.name}_norm.csv", index_col=0)
        features = meta["features"]
        hp = meta["hyperparameters"]
        if list(norm.index) != features or not np.isfinite(norm[["mean", "std"]]).all().all() or (norm["std"] <= 0).any():
            raise ValueError("Normalization does not match the trained feature contract")
        if model.observation_space.shape != (hp["window"] * len(features) + 3,):
            raise ValueError("Model observation dimension does not match metadata")
        if model.action_space.n != len(hp["action_profile_config"]["actions"]):
            raise ValueError("Model action dimension does not match metadata")
        weights = self.path / f"{self.name}.zip"
        if not weights.is_file():
            raise ValueError("Final model weights are missing")
        saved = model.__class__.load(weights, device="cpu")
        if (saved.observation_space.shape != model.observation_space.shape or
                saved.action_space.n != model.action_space.n):
            raise ValueError("Saved weights do not match the trained model")
        params = self.path / f"{self.name}.params.json"
        if params.exists():
            if not isinstance(json.loads(params.read_text(encoding="utf-8-sig")), dict):
                raise ValueError("Invalid collector sidecar")
        meta.update(run_id=self.run_id, status="completed", params_available=params.exists())
        meta["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (weights, self.path / f"{self.name}_norm.csv", params) if p.exists()
        }
        # Existing quick-eval code uses infinity for PF when there are no losses.
        def finite_json(value):
            if isinstance(value, dict):
                return {k: finite_json(v) for k, v in value.items()}
            if isinstance(value, list):
                return [finite_json(v) for v in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        _atomic_json(meta_file, finite_json(meta))
        self.state["status"] = "completed"
        self._write_state()
        _atomic_json(self.home / "current.json", {"run_id": self.run_id})
        self.committed = True

    def __exit__(self, kind, error, traceback):
        _RUN_OUTPUTS.pop(self.name, None)
        if not self.committed:
            self.state.update(status="cancelled" if kind is KeyboardInterrupt else "failed",
                              error=str(error or "Run ended without publication"))
            self._write_state()


def mark_process_runs(pid, status="cancelled"):
    """Called by the GUI after terminating its captured child process."""
    for path in MODELS_DIR.glob("*/runs/*/run.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("pid") == pid and state.get("status") == "running":
                state["status"] = status
                _atomic_json(path, state)
        except (OSError, ValueError):
            continue


def recover_interrupted_runs():
    import psutil
    recovered = []
    for path in MODELS_DIR.glob("*/runs/*/run.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            alive = psutil.pid_exists(state.get("pid", -1))
            if alive and state.get("pid_created"):
                try:
                    alive = psutil.Process(state["pid"]).create_time() == state["pid_created"]
                except psutil.Error:
                    alive = False
            if state.get("status") == "running" and not alive:
                state.update(status="failed", error="Process exited before publication")
                _atomic_json(path, state)
                recovered.append(str(path))
        except (OSError, ValueError, TypeError):
            continue
    return recovered


def model_dir(model_name: str) -> Path:
    if model_name in _RUN_OUTPUTS:
        return _RUN_OUTPUTS[model_name]
    if model_name in _READ_OUTPUTS:
        return _READ_OUTPUTS[model_name]
    home = _model_home(model_name)
    pointer = home / "current.json"
    if not pointer.exists():
        return home
    run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError(f"Invalid model generation pointer: {pointer}")
    return home / "runs" / run_id


def backtests_dir(model_name: str) -> Path:
    return model_dir(model_name) / "backtests"


def logs_dir(model_name: str) -> Path:
    return model_dir(model_name) / "logs"


def best_dir(model_name: str) -> Path:
    return model_dir(model_name) / "best"


def ensure_model_dirs(model_name: str) -> Path:
    base = model_dir(model_name)
    for path in (base, backtests_dir(model_name), logs_dir(model_name), best_dir(model_name)):
        path.mkdir(parents=True, exist_ok=True)
    return base


def final_model_path(model_name: str) -> Path:
    return model_dir(model_name) / f"{model_name}.zip"


def legacy_final_model_path(model_name: str) -> Path:
    return ROOT / f"{model_name}.zip"


def best_model_path(model_name: str) -> Path:
    return best_dir(model_name) / "best_model.zip"


def legacy_best_model_path(model_name: str) -> Path:
    return ROOT / f"{model_name}_best" / "best_model.zip"


def norm_path(model_name: str) -> Path:
    return model_dir(model_name) / f"{model_name}_norm.csv"


def legacy_norm_path(model_name: str) -> Path:
    return ROOT / f"{model_name}_norm.csv"


def params_path(model_name: str) -> Path:
    return model_dir(model_name) / f"{model_name}.params.json"


def legacy_params_path(model_name: str) -> Path:
    return ROOT / f"{model_name}.params.json"


def train_meta_path(model_name: str) -> Path:
    return model_dir(model_name) / f"{model_name}.train.json"


def backtest_meta_path(model_name: str) -> Path:
    return backtests_dir(model_name) / f"{model_name}_live_bt.meta.json"


def trades_path(model_name: str) -> Path:
    return backtests_dir(model_name) / f"{model_name}_live_bt_trades.csv"


def legacy_trades_path(model_name: str) -> Path:
    return ROOT / f"{model_name}_live_bt_trades.csv"


def equity_path(model_name: str) -> Path:
    return backtests_dir(model_name) / f"{model_name}_live_bt_equity.png"


def mc_chart_path(model_name: str) -> Path:
    """SQX-style Monte Carlo fan chart (shuffled-order equity paths)."""
    return backtests_dir(model_name) / f"{model_name}_live_bt_mc.png"


def train_diag_path(model_name: str) -> Path:
    """4-panel training-diagnosis chart rendered by train_diagnose.py."""
    return model_dir(model_name) / f"{model_name}_train_diag.png"


def legacy_equity_path(model_name: str) -> Path:
    return ROOT / f"{model_name}_live_bt_equity.png"


def chart_path(model_name: str) -> Path:
    return backtests_dir(model_name) / f"{model_name}_backtest_chart.html"


def legacy_chart_path(model_name: str) -> Path:
    return ROOT / f"{model_name}_backtest_chart.html"


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def _uses_generation(model_name):
    # Consult the pinned directory, not a pointer another process may replace.
    return model_dir(model_name) != _model_home(model_name)


def find_model_path(model_name: str, source: str = "final") -> Path | None:
    source = (source or "final").lower()
    final_candidates = [final_model_path(model_name), legacy_final_model_path(model_name)]
    best_candidates = [best_model_path(model_name), legacy_best_model_path(model_name)]
    if _uses_generation(model_name):
        final_candidates = final_candidates[:1]
        best_candidates = best_candidates[:1]
    if source == "final":
        return first_existing(final_candidates)
    if source == "best":
        return first_existing(best_candidates)
    if source == "auto":
        return first_existing(final_candidates + best_candidates)
    return None


def find_norm_path(model_name: str) -> Path | None:
    if _uses_generation(model_name):
        return first_existing([norm_path(model_name)])
    return first_existing([norm_path(model_name), legacy_norm_path(model_name)])


def find_params_path(model_name: str) -> Path | None:
    if _uses_generation(model_name):
        return first_existing([params_path(model_name)])
    return first_existing([params_path(model_name), legacy_params_path(model_name)])


def find_trades_path(model_name: str) -> Path | None:
    if _uses_generation(model_name):
        return first_existing([trades_path(model_name)])
    return first_existing([trades_path(model_name), legacy_trades_path(model_name)])


def find_equity_path(model_name: str) -> Path | None:
    if _uses_generation(model_name):
        return first_existing([equity_path(model_name)])
    return first_existing([equity_path(model_name), legacy_equity_path(model_name)])


def find_chart_path(model_name: str) -> Path | None:
    if _uses_generation(model_name):
        return first_existing([chart_path(model_name)])
    return first_existing([chart_path(model_name), legacy_chart_path(model_name)])


def model_names_from_artifacts() -> list[str]:
    names = set()
    for path in ROOT.glob("*.zip"):
        names.add(path.stem)
    for path in ROOT.glob("*_best"):
        if (path / "best_model.zip").exists() and path.name.endswith("_best"):
            names.add(path.name[:-5])
    for path in MODELS_DIR.glob("*"):
        if not path.is_dir():
            continue
        name = path.name
        active = model_dir(name)
        if (active / f"{name}.zip").exists() or (active / "best" / "best_model.zip").exists():
            names.add(name)
    return sorted(names)
