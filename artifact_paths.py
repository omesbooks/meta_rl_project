"""Central paths for generated training/backtest artifacts.

New artifacts are written under:
    artifacts/models/<model>/

Legacy root-level files are still discoverable so older runs remain usable.
"""
from pathlib import Path


ROOT = Path(__file__).parent.resolve()
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"


def model_dir(model_name: str) -> Path:
    return MODELS_DIR / model_name


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


def find_model_path(model_name: str, source: str = "final") -> Path | None:
    source = (source or "final").lower()
    final_candidates = [final_model_path(model_name), legacy_final_model_path(model_name)]
    best_candidates = [best_model_path(model_name), legacy_best_model_path(model_name)]
    if source == "final":
        return first_existing(final_candidates)
    if source == "best":
        return first_existing(best_candidates)
    if source == "auto":
        return first_existing(final_candidates + best_candidates)
    return None


def find_norm_path(model_name: str) -> Path | None:
    return first_existing([norm_path(model_name), legacy_norm_path(model_name)])


def find_params_path(model_name: str) -> Path | None:
    return first_existing([params_path(model_name), legacy_params_path(model_name)])


def find_trades_path(model_name: str) -> Path | None:
    return first_existing([trades_path(model_name), legacy_trades_path(model_name)])


def find_equity_path(model_name: str) -> Path | None:
    return first_existing([equity_path(model_name), legacy_equity_path(model_name)])


def find_chart_path(model_name: str) -> Path | None:
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
        if (path / f"{name}.zip").exists() or (path / "best" / "best_model.zip").exists():
            names.add(name)
    return sorted(names)
