"""Gated entries: TradingEnv opens a buy only where the raw gate column is > 0 and a
sell only where it is < 0; off-gate buys/sells are plain holds. rl_train carries the
RAW gate next to the normalised features and records it in the train meta."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from trading_env import TradingEnv  # noqa: E402


def _df(n=60):
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y.%m.%d %H:%M"),
                       "open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
                       "f0": rng.normal(size=n), "f1": rng.normal(size=n), "gate": 0.0})
    df.loc[20, "gate"] = 1.0; df.loc[30, "gate"] = -1.0
    return df


def test_env_gate_blocks_offgate_and_wrong_direction():
    df = _df()
    env = TradingEnv(df, ["f0", "f1"], window_size=5, random_start=False, max_steps=50, entry_gate="gate")
    env.reset()
    buy, sell, close = env.action_ids["buy"], env.action_ids["sell"], env.action_ids["close"]
    while env.t < 20:
        env.step(buy)
        assert env.position == 0                       # no gate -> buy is a hold
    env.step(sell); assert env.position == 0           # t == 20: gate > 0, a sell is blocked
    # position unchanged, t advanced: re-check at the next gate bar
    while env.t < 30:
        env.step(buy)
    assert env.position == 0                           # t == 30 reached flat (buy blocked after 20 passed)
    env.step(sell); assert env.position == -1          # gate < 0 -> sell opens
    env.step(close); assert env.position == 0


def test_env_gate_opens_in_signal_direction():
    df = _df()
    env = TradingEnv(df, ["f0", "f1"], window_size=5, random_start=False, max_steps=50, entry_gate="gate")
    env.reset()
    buy = env.action_ids["buy"]
    while env.t < 20:
        env.step(0)
    env.step(buy); assert env.position == 1            # gate > 0 at t == 20


def test_rl_train_records_gate_and_backtest_applies_it(tmp_path):
    """3-step end to end: the meta carries entry_gate and backtest_live resolves it."""
    df = _df(400)
    df["gate"] = np.where(np.arange(400) % 25 == 0, 1.0, np.where(np.arange(400) % 25 == 12, -1.0, 0.0))
    csv = tmp_path / "gated.csv"; df.to_csv(csv, index=False)
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    name = "test_gate_tmp"
    r = subprocess.run([py, "rl_train.py", str(csv), "--steps", "600", "--window", "5", "--name", name,
                        "--entry_gate", "gate", "--max_hold", "6", "--mc_eval", "0", "--n_steps", "64",
                        "--batch_size", "32", "--ep_len", "100"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    from artifact_paths import train_meta_path
    meta = json.loads(train_meta_path(name).read_text(encoding="utf-8-sig"))
    assert meta["hyperparameters"]["entry_gate"] == "gate"
    assert "entry_gate__raw" not in meta.get("features", [])
    r = subprocess.run([py, "backtest_live.py", name, str(csv), "--conf", "0", "--mode", "pure_agent",
                        "--start", "0.5", "--mc", "0", "--bootstrap", "0", "--random_baseline", "0"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "[entry_gate] gate (model meta)" in r.stdout
    trades = sorted((ROOT / "artifacts" / "models" / name).rglob("*_live_bt_trades.csv"), key=lambda q: q.stat().st_mtime)
    tr = pd.read_csv(trades[-1])
    if len(tr):
        g = df.set_index(pd.to_datetime(df["timestamp"], format="%Y.%m.%d %H:%M"))["gate"]
        at = g.reindex(pd.to_datetime(tr["open_time"])).to_numpy()
        assert np.all(np.where(tr["side"].str.lower().str.startswith("l"), at > 0, at < 0))
