"""tools/viz/snapshot.py must describe ANY actor-critic net the trainer can make:
depth/width from net_arch, Tanh or ReLU, 4 or 6 actions — and write_live must
keep a consistent index for the live viewer."""
import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("viz_snapshot", ROOT / "tools" / "viz" / "snapshot.py")
snap_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snap_mod)

from action_profiles import get_action_profile  # noqa: E402
from trading_env import TradingEnv  # noqa: E402

FEATS = [f"f{i}" for i in range(6)]
WINDOW = 5


def _frame(n=260, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h").strftime("%Y.%m.%d %H:%M"),
        "open": close * (1 + rng.normal(0, 0.001, n)), "close": close,
    })
    df["high"] = df[["open", "close"]].max(axis=1) * 1.002
    df["low"] = df[["open", "close"]].min(axis=1) * 0.998
    for f in FEATS:
        df[f] = rng.normal(0, 1, n)
    return df


def _policy_and_env(net_arch, activation, profile):
    from stable_baselines3 import PPO
    _, cfg = get_action_profile(profile)
    df = _frame()
    make = lambda: TradingEnv(df, FEATS, window_size=WINDOW, random_start=False,
                              max_steps=len(df) - WINDOW - 2, action_profile=cfg)
    model = PPO("MlpPolicy", make(), n_steps=64, batch_size=32, seed=0, device="cpu",
                policy_kwargs=dict(net_arch=net_arch, activation_fn=activation))
    return model.policy, make, cfg


@pytest.mark.parametrize("net_arch,activation,profile,n_actions", [
    ([32, 16], torch.nn.ReLU, "manage_6", 6),
    ([64, 64, 32], torch.nn.Tanh, "basic_4", 4),
])
def test_capture_shapes(net_arch, activation, profile, n_actions):
    policy, make, cfg = _policy_and_env(net_arch, activation, profile)
    snap = snap_mod.capture(policy, make(), WINDOW, 80, k=8, action_names=snap_mod.action_names(cfg),
                            meta={"step": 10000})
    n_in = WINDOW * len(FEATS) + 3
    assert snap["schema"] == snap_mod.SCHEMA
    assert snap["n_in"] == n_in and snap["n_features"] == len(FEATS)
    assert snap["actor_layers"] == net_arch and snap["critic_layers"] == net_arch
    assert len(snap["actions"]) == n_actions == len(snap["action_labels"])
    assert snap["width"] == n_in + 2 * sum(net_arch) + n_actions + 1
    q = np.frombuffer(base64.b64decode(snap["q"]), dtype=np.int8)
    assert q.size == snap["frames"] * snap["width"]
    for key in ("ts", "o", "h", "l", "c", "act", "pos", "eq", "tpnl"):
        assert len(snap[key]) == snap["frames"], key
    for tower in ("actor", "critic"):
        mats = snap["edges"][tower]
        assert len(mats) == len(net_arch) + 1
        assert [m["cols"] for m in mats][0] == n_in
        for m in mats:
            idx = np.frombuffer(base64.b64decode(m["idx"]), dtype=np.int16)
            assert idx.size == m["rows"] * m["k"] and idx.max() < m["cols"]
    assert snap["edges"]["actor"][-1]["rows"] == n_actions
    assert snap["edges"]["critic"][-1]["rows"] == 1
    json.dumps(snap)                                   # must be serialisable as-is


def test_rejects_non_actor_critic():
    class Q(torch.nn.Module):
        pass
    with pytest.raises(ValueError):
        snap_mod.capture(Q(), None, WINDOW, 10)


def test_write_live_index(tmp_path):
    policy, make, cfg = _policy_and_env([16], torch.nn.Tanh, "basic_4")
    for i in range(3):
        snap = snap_mod.capture(policy, make(), WINDOW, 20, k=4)
        snap_mod.write_live(tmp_path, i, snap, {"step": (i + 1) * 10000, "eval_reward": float(i)})
    snap_mod.write_live(tmp_path, 1, snap, {"step": 20000, "eval_reward": 9.0})   # rewrite keeps one entry
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert [s["i"] for s in sorted(index["snapshots"], key=lambda s: s["i"])] == [0, 1, 2]
    assert {s["i"]: s["eval_reward"] for s in index["snapshots"]}[1] == 9.0
    assert all((tmp_path / s["file"]).is_file() for s in index["snapshots"])
    assert not list(tmp_path.glob("*.tmp"))
