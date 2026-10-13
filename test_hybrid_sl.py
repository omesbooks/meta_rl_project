"""Continuous stop-loss head (hybrid_policy.HybridPolicy + TradingEnv sl_mode='continuous').

The hybrid distribution's log-prob / entropy must equal the hand-computed Categorical +
Gaussian values, deterministic actions must be (argmax, mu), the env must set the stop
from the RAW ATR at entry and fill it intrabar, and rl_train must save a model that
PPO.load brings back with the same policy class and a (2,) action."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch as th

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hybrid_policy import HybridDistribution, HybridPolicy, hybrid_action_space, split_action  # noqa: E402
from trading_env import TradingEnv  # noqa: E402
from training_data import atr_mt5  # noqa: E402


def test_hybrid_distribution_matches_hand_computation():
    d = HybridDistribution(n_actions=4, sl_ent_scale=0.1)
    out = th.tensor([[0.2, 1.4, -0.3, 0.1, 1.6]])            # 4 logits + mu
    log_std = th.tensor([math.log(0.4)])
    d.proba_distribution(out, log_std)
    a = th.tensor([[1.0, 1.9]])                              # buy, SL 1.9 ATR
    lp = d.log_prob(a).item()
    logp_cat = math.log(math.exp(1.4) / sum(math.exp(x) for x in (0.2, 1.4, -0.3, 0.1)))
    logp_g = -0.5 * ((1.9 - 1.6) / 0.4) ** 2 - math.log(0.4) - 0.5 * math.log(2 * math.pi)
    assert abs(lp - (logp_cat + logp_g)) < 1e-5
    probs = [math.exp(x) for x in (0.2, 1.4, -0.3, 0.1)]; Z = sum(probs)
    h_cat = -sum(p / Z * math.log(p / Z) for p in probs)
    h_g = 0.5 * math.log(2 * math.pi * math.e) + math.log(0.4)
    assert abs(d.entropy().item() - (h_cat + 0.1 * h_g)) < 1e-5
    m = d.mode()[0]
    assert m[0].item() == 1.0 and abs(m[1].item() - 1.6) < 1e-6
    s = th.stack([d.sample()[0] for _ in range(300)])
    assert set(s[:, 0].tolist()) <= {0.0, 1.0, 2.0, 3.0} and 1.4 < s[:, 1].mean() < 1.8 and 0.25 < s[:, 1].std() < 0.6
    assert split_action(np.array([1.4, 9.0]), 4, (0.5, 3.0)) == (1, 3.0)
    sp = hybrid_action_space(4, (0.5, 3.0)); assert sp.shape == (2,) and sp.high[0] == 3.0 and sp.low[1] == 0.5


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y.%m.%d %H:%M"),
                       "open": close, "high": close + 0.15, "low": close - 0.15, "close": close,
                       "f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    df["atr__raw"] = pd.Series(atr_mt5(df["high"], df["low"], df["close"], 14)).bfill()
    return df


def test_env_sets_stop_from_raw_atr_and_fills_intrabar():
    df = _df()
    df.loc[40, "low"] = df.loc[40, "close"] - 5.0           # a deep wick two bars after the entry
    env = TradingEnv(df, ["f0", "f1"], window_size=5, random_start=False, max_steps=100,
                     sl_mode="continuous", sl_range=(0.5, 3.0), atr_col="atr__raw", spread_pct=0.0, commission=0.0)
    env.reset()
    assert env.action_space.shape == (2,)
    while env.t < 38:
        env.step(np.array([0.0, 1.0]))
    env.step(np.array([1.0, 2.0]))                          # buy at close[38] with SL 2 ATR
    atr = float(df.loc[38, "atr__raw"]); entry = float(df.loc[38, "close"])
    assert env.position == 1 and abs(env.stop_price - (entry - 2.0 * atr)) < 1e-5
    env.step(np.array([0.0, 1.0]))                          # bar 39: no hit
    assert env.position == 1
    env.step(np.array([0.0, 1.0]))                          # bar 40: low breaches the stop -> filled AT the stop
    assert env.position == 0
    tr = env.trades[-1]
    assert tr["reason"] == "sl" and abs(tr["exit"] - (entry - 2.0 * atr)) < 1e-5 and tr["sl_atr"] == 2.0
    assert abs(tr["pnl"] - (-2.0 * atr / entry)) < 1e-5


def test_env_sell_stop_and_out_of_range_sl_is_clipped():
    df = _df(seed=3)
    env = TradingEnv(df, ["f0", "f1"], window_size=5, random_start=False, max_steps=100,
                     sl_mode="continuous", sl_range=(0.5, 3.0), atr_col="atr__raw", spread_pct=0.0, commission=0.0)
    env.reset()
    env.step(np.array([2.0, 7.5]))                          # sell, SL request 7.5 -> clipped to 3.0
    atr = float(df.loc[4, "atr__raw"]); entry = float(df.loc[4, "close"])
    assert env.position == -1 and abs(env.stop_price - (entry + 3.0 * atr)) < 1e-5
    with pytest.raises(ValueError):
        TradingEnv(df, ["f0", "f1"], window_size=5, sl_mode="continuous", atr_col="")


def test_rl_train_continuous_sl_saves_loadable_hybrid_model(tmp_path):
    df = _df(500, seed=5).drop(columns=["atr__raw"])
    csv = tmp_path / "sl.csv"; df.to_csv(csv, index=False)
    py = str(ROOT / ".venv" / "Scripts" / "python.exe"); name = "test_sl_tmp"
    r = subprocess.run([py, "rl_train.py", str(csv), "--steps", "600", "--window", "5", "--name", name,
                        "--sl_mode", "continuous", "--sl_range", "0.8,2.5", "--max_hold", "6", "--mc_eval", "0",
                        "--n_steps", "64", "--batch_size", "32", "--ep_len", "100"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "[sl] continuous head" in r.stdout
    from artifact_paths import train_meta_path, final_model_path
    meta = json.loads(train_meta_path(name).read_text(encoding="utf-8-sig"))["hyperparameters"]
    assert meta["sl_mode"] == "continuous" and meta["sl_range"] == [0.8, 2.5]
    from stable_baselines3 import PPO
    model = PPO.load(str(final_model_path(name)), device="cpu")
    assert isinstance(model.policy, HybridPolicy) and model.policy.sl_range == (0.8, 2.5)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    a, _ = model.predict(obs, deterministic=True)
    assert a.shape == (2,) and 0 <= a[0] <= 3 and 0.8 <= a[1] <= 2.5
    probs, sl, v = model.policy.heads(obs)
    assert abs(float(probs.sum()) - 1.0) < 1e-5 and 0.8 <= float(sl[0]) <= 2.5
