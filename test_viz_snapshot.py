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


def test_summary_matches_snapshot():
    policy, make, cfg = _policy_and_env([16], torch.nn.Tanh, "manage_6")
    snap = snap_mod.capture(policy, make(), WINDOW, 120, k=4, action_names=snap_mod.action_names(cfg))
    s = snap_mod.summary(snap, points=50)
    assert len(s["eq_curve"]) == 50 and s["eq_curve"][-1] == round(snap["eq"][-1], 5)
    assert len(s["action_counts"]) == 6 and sum(s["action_counts"]) == snap["frames"]
    assert s["trades"] == sum(1 for p in snap["tpnl"] if p) and s["max_dd"] <= 0
    json.dumps(s)


def test_edges_ranked_by_influence_not_raw_weight():
    """A tiny-scale input with a huge weight must lose to a normal-scale input
    with a moderate weight — that is what the connection actually contributes."""
    from stable_baselines3 import PPO
    _, cfg = get_action_profile("basic_4")
    df = _frame()
    df["f0"] = df["f0"] * 1e-3                         # std ~0.001
    make = lambda: TradingEnv(df, FEATS, window_size=WINDOW, random_start=False,
                              max_steps=len(df) - WINDOW - 2, action_profile=cfg)
    model = PPO("MlpPolicy", make(), n_steps=64, batch_size=32, seed=0, device="cpu",
                policy_kwargs=dict(net_arch=[8]))
    nf = len(FEATS)
    tiny = [b * nf + 0 for b in range(WINDOW)]
    normal = [b * nf + 1 for b in range(WINDOW)]
    with torch.no_grad():
        for seq in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
            W = next(m for m in seq if isinstance(m, torch.nn.Linear)).weight
            W.fill_(0.01)
            W[:, tiny] = 10.0
            W[:, normal] = 0.5
    snap = snap_mod.capture(model.policy, make(), WINDOW, 80, k=1)
    assert snap["edge_rank"] == "abs_weight_x_source_std"
    for tower in ("actor", "critic"):
        m = snap["edges"][tower][0]
        picked = np.frombuffer(base64.b64decode(m["idx"]), dtype=np.int16)
        assert set(picked.tolist()) <= set(normal), (tower, picked)


def test_rollout_callback_writes_what_the_agent_played(tmp_path):
    """Training-rollout replay: frames come from the PPO rollout buffer, form one
    contiguous stretch of bars, and each frame's observation is the window that
    ends at the bar it is labelled with."""
    from stable_baselines3 import PPO
    _, cfg = get_action_profile("basic_4")
    df = _frame(n=400)
    env = TradingEnv(df, FEATS, window_size=WINDOW, max_steps=120, action_profile=cfg)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=1, device="cpu",
                policy_kwargs=dict(net_arch=[16]))
    cb = snap_mod.make_rollout_callback(tmp_path, WINDOW, names=snap_mod.action_names(cfg),
                                        frames=50, min_interval=0.0, total_steps=192, model_name="t")
    model.learn(total_timesteps=192, callback=cb)
    meta = json.loads((tmp_path / "rollout_meta.json").read_text(encoding="utf-8"))
    snap = json.loads((tmp_path / "rollout_latest.json").read_text(encoding="utf-8"))
    assert meta["seq"] == 2 and snap["meta"]["source"] == "rollout" and snap["meta"]["rollout"] == 2
    n = snap["frames"]
    assert 1 <= n <= 50
    for key in ("act", "greedy", "pos", "eq", "tpnl", "reward", "adv", "ts", "c"):
        assert len(snap[key]) == n, key
    assert set(snap["act"]) <= {0, 1, 2, 3} and 0 <= snap["meta"]["explore"] <= 1
    row = {t: i for i, t in enumerate(df["timestamp"])}
    tix = [row[t] for t in snap["ts"]]
    assert tix == list(range(tix[0], tix[0] + n))           # contiguous bars, no episode jump
    q = np.frombuffer(base64.b64decode(snap["q"]), dtype=np.int8).reshape(n, snap["width"]) / 127
    nf = len(FEATS)
    feats = df[FEATS].to_numpy(dtype=np.float32)
    for f in (0, n // 2, n - 1):
        t = tix[f]
        expect = np.clip(feats[t - WINDOW + 1:t + 1].reshape(-1) / 4.0, -1, 1)
        assert np.allclose(q[f, :WINDOW * nf], expect, atol=0.01), f


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


def test_rollout_callback_writes_monitor_rows_with_gate_counterfactual(tmp_path):
    """monitor.json gets one row per rollout: trade frequency, holds, exits, action mix,
    and - on a gated env - acceptance plus the fixed-rule counterfactual split into the
    signal bars the agent accepted vs skipped."""
    from stable_baselines3 import PPO
    _, cfg = get_action_profile("basic_4")
    df = _frame(n=400)
    rng = np.random.default_rng(0)
    df["gate"] = 0.0
    df.loc[rng.choice(np.arange(20, 380), 60, replace=False), "gate"] = rng.choice([-1.0, 1.0], 60)
    env = TradingEnv(df, FEATS, window_size=WINDOW, max_steps=120, action_profile=cfg, entry_gate="gate")
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=2, device="cpu", policy_kwargs=dict(net_arch=[16]))
    cb = snap_mod.make_rollout_callback(tmp_path, WINDOW, names=snap_mod.action_names(cfg),
                                        frames=50, min_interval=0.0, total_steps=256, model_name="t")
    model.learn(total_timesteps=256, callback=cb)
    mon = json.loads((tmp_path / "monitor.json").read_text(encoding="utf-8"))
    rows = mon["rows"]
    assert len(rows) == 4 and [r["seq"] for r in rows] == [0, 1, 2, 3] and mon["entry_gate"] == "gate"
    tot = {k: sum(r[k] for r in rows) for k in ("bars", "trades_opened", "trades_closed", "signal_bars", "accepted")}
    assert tot["bars"] == 256 and tot["accepted"] <= tot["signal_bars"] and tot["trades_opened"] >= tot["accepted"]
    for r in rows:
        assert r["gate"] and len(r["action_mix"]) == 4 and sum(r["action_mix"]) == r["bars"]
        assert 0 <= r["action_entropy"] <= np.log(4) + 1e-6
        assert r["cf_accepted_n"] + r["cf_skipped_n"] <= r["signal_bars"]
        assert abs(r["trades_per_100"] - 100 * r["trades_opened"] / r["bars"]) < 0.02
    # every opened trade on a gated env sits on a signal bar in the signal's direction
    assert tot["trades_opened"] == tot["accepted"]
    closed = sum(len(r["exits"]) > 0 for r in rows)
    assert closed >= 1 and all(set(r["exits"]) <= {"signal", "max_hold", "episode_end", "sl", "managed_sl"} for r in rows)


def test_summary_behaviour_fields_and_gate_acceptance():
    """index.json entries carry hold/pnl/entries and, for gated envs, acceptance - the
    per-evaluation timeline reads these without downloading snapshots."""
    from stable_baselines3 import PPO
    _, cfg = get_action_profile("basic_4")
    df = _frame(n=300)
    df["gate"] = 0.0; df.loc[np.arange(10, 290, 7), "gate"] = 1.0
    make = lambda: TradingEnv(df, FEATS, window_size=WINDOW, random_start=False, max_steps=200, action_profile=cfg, entry_gate="gate")
    model = PPO("MlpPolicy", make(), n_steps=64, batch_size=32, seed=0, device="cpu", policy_kwargs=dict(net_arch=[16]))
    snap = snap_mod.capture(model.policy, make(), WINDOW, 150, k=4, action_names=snap_mod.action_names(cfg))
    assert len(snap["gate"]) == snap["frames"]
    s = snap_mod.summary(snap)
    for k in ("entries", "bars", "hold_mean", "pnl_bp_mean", "signal_bars", "accepted", "acceptance"):
        assert k in s, k
    assert s["bars"] == snap["frames"] and s["accepted"] <= s["signal_bars"] and s["entries"] == s["accepted"]
    if s["entries"]:
        assert s["hold_mean"] >= 1


def test_live_viewer_finds_sibling_seed_runs(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lv", ROOT / "tools" / "viz" / "live_viewer.py")
    lv = importlib.util.module_from_spec(spec); spec.loader.exec_module(lv)
    models = tmp_path / "models"
    for name, n_runs in (("abl_C_s0", 1), ("abl_C_s1", 2), ("abl_C_s2", 1), ("abl_C", 1), ("other_s0", 1)):
        for i in range(n_runs):
            viz = models / name / "runs" / f"r{i}" / "viz"; viz.mkdir(parents=True)
            (viz / "index.json").write_text("{}")
    peers = lv.find_peers(models / "abl_C_s0" / "runs" / "r0" / "viz")
    names = [n for n, _ in peers]
    assert names == ["abl_C_s0", "abl_C_s1", "abl_C_s2"]       # this run first, same stem, _s<seed> only
    assert peers[1][1].name == "viz" and peers[1][1].parent.name == "r1"   # newest run of the sibling


def test_attribution_is_gradient_times_input_and_lands_in_snapshots():
    """grad x input of the chosen action's logit: on a one-layer linear actor the
    attribution of input j for action a must equal W[a, j] * x[j]."""
    from stable_baselines3 import PPO
    _, cfg = get_action_profile("basic_4")
    df = _frame()
    make = lambda: TradingEnv(df, FEATS, window_size=WINDOW, random_start=False, max_steps=len(df) - WINDOW - 2, action_profile=cfg)
    model = PPO("MlpPolicy", make(), n_steps=64, batch_size=32, seed=0, device="cpu", policy_kwargs=dict(net_arch=[]))
    pol = model.policy
    obs = np.random.default_rng(0).normal(size=(5, pol.observation_space.shape[0])).astype(np.float32)
    acts = np.array([1, 2, 0, 3, 1])
    A = snap_mod.attribution(pol, obs, acts)
    W = pol.action_net.weight.detach().numpy()
    assert A.shape == obs.shape and np.allclose(A, W[acts] * obs, atol=1e-5)
    snap = snap_mod.capture(pol, make(), WINDOW, 40, k=4, action_names=snap_mod.action_names(cfg), feature_names=FEATS)
    n_in = snap["n_in"]
    assert len(snap["input_names"]) == n_in and snap["input_names"][0] == "f0@-4" and snap["input_names"][-1] == "bars_in_position"
    q = np.frombuffer(base64.b64decode(snap["attr"]), dtype=np.int8).reshape(snap["frames"], n_in)
    assert np.abs(q).max(axis=1).min() == 127 and len(snap["attr_scale"]) == snap["frames"]
    assert abs(sum(snap["attr_feature"]) - 1) < 1e-3 and abs(sum(snap["attr_lag"]) + sum(snap["attr_feature"][len(FEATS):]) - 1) < 1e-3
    assert snap["attr_top"][0][0] in snap["input_names"][0].split("@")[0:1] + FEATS + ["position", "unrealized_pnl", "bars_in_position"]
    assert "top_features" in snap_mod.summary(snap)


def test_feature_usage_flags_features_the_decision_ignores():
    """Permutation test: a feature with zero weight never changes a decision (flip = tv = 0);
    a heavily weighted one does. Eval snapshots carry it per feature (+ state inputs)."""
    from stable_baselines3 import PPO
    _, cfg = get_action_profile("basic_4")
    df = _frame()
    make = lambda: TradingEnv(df, FEATS, window_size=WINDOW, random_start=False, max_steps=len(df) - WINDOW - 2, action_profile=cfg)
    model = PPO("MlpPolicy", make(), n_steps=64, batch_size=32, seed=0, device="cpu", policy_kwargs=dict(net_arch=[]))
    pol = model.policy
    nfeat, n_in = len(FEATS), pol.observation_space.shape[0]
    with torch.no_grad():
        W = pol.action_net.weight
        W.zero_(); pol.action_net.bias.zero_()
        lag_cols = lambda f: [r * nfeat + f for r in range(WINDOW)]
        W[1, lag_cols(0)] = 3.0; W[2, lag_cols(0)] = -3.0        # f0 decides buy vs sell; every other input is ignored
    obs = np.random.default_rng(1).normal(size=(200, n_in)).astype(np.float32)
    flip, tv = snap_mod.feature_usage(pol, obs, WINDOW)
    assert len(flip) == nfeat + (n_in - WINDOW * nfeat)
    assert flip[0] > 0.2 and tv[0] > 0.05
    assert np.all(flip[1:] == 0) and np.all(tv[1:] < 1e-6)
    snap = snap_mod.capture(pol, make(), WINDOW, 40, k=4, action_names=snap_mod.action_names(cfg), feature_names=FEATS)
    assert len(snap["use_flip"]) == len(snap["use_tv"]) == nfeat + 3 and snap["use_tiers"]["used"] > 0
    bars = {k: [0.0] * 40 for k in ("o", "h", "l", "c", "eq", "tpnl")}
    bars.update(ts=["t"] * 40, act=[0] * 40, pos=[0] * 40)
    assert "use_flip" not in snap_mod.pack(pol, obs[:40], WINDOW, bars, k=4, usage=False)   # rollouts skip it


def test_static_snapshot_page_points_every_relative_import_at_tools_viz(tmp_path):
    """--html pages live in tools/viz/out: ./vendor, ./dashboard.js AND ./monitor.js must be rewritten
    (only dashboard.js was, so every static page stayed blank once monitor.js existed)."""
    spec = importlib.util.spec_from_file_location("cap", ROOT / "tools" / "viz" / "capture_snapshot.py")
    cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
    html = cap.page_html({"schema": "x"}, ROOT / "tools" / "viz" / "out")
    assert '"./' not in html.split("<script type=\"importmap\">", 1)[1].split("</script>", 1)[0]
    assert '"../monitor.js"' in html and '"../dashboard.js"' in html and '"../vendor/three/three.module.js"' in html
    assert '{"schema": "x"}' in html


def test_live_viewer_resolves_a_viz_dir_before_the_first_eval(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lv2", ROOT / "tools" / "viz" / "live_viewer.py")
    lv = importlib.util.module_from_spec(spec); spec.loader.exec_module(lv)
    run = tmp_path / "runs" / "r1"; viz = run / "viz"; viz.mkdir(parents=True)
    (viz / "rollout_meta.json").write_text("{}")                 # rollouts exist, no index.json yet
    assert lv.resolve_viz_dir(run_dir=viz) == viz.resolve()      # the GUI passes the viz folder
    assert lv.resolve_viz_dir(run_dir=run) == viz.resolve()      # or the run folder


def test_greedy_walk_stability_rows_and_file(tmp_path):
    """greedy_walk = argmax policy on a fresh env; stability rows keep per-bar probabilities
    (uint8), the greedy action and the share of decisions that changed since the last eval."""
    policy, make, cfg = _policy_and_env([16], torch.nn.Tanh, "basic_4")
    names = snap_mod.action_names(cfg)
    w = snap_mod.greedy_walk(policy, make(), 60)
    assert w["probs"].shape == (60, 4) and np.allclose(w["probs"].sum(axis=1), 1, atol=1e-4)
    assert (w["probs"].argmax(axis=1) == w["act"]).all()
    assert len(w["bars"]["c"]) == 60 == w["stats"]["bars"] and len(w["eq"]) == 60
    st = w["stats"]
    assert st["trades"] == st["wins"] + int((w["tpnl"][w["tpnl"] != 0] <= 0).sum())
    assert st["pf"] is None or st["pf"] >= 0
    json.dumps(st)                                        # no Infinity / NaN in the index entry
    row0 = snap_mod.stability_row(0, 10000, w, None)
    assert row0["changed"] is None and row0["bars"] == 60 and row0["n_actions"] == 4
    row1 = snap_mod.stability_row(1, 20000, w, w["act"])
    assert row1["changed"] == 0.0
    flipped = np.where(w["act"] == 0, 1, 0)               # every decision different
    assert snap_mod.stability_row(2, 30000, w, flipped)["changed"] == 1.0
    p = np.frombuffer(base64.b64decode(row1["p"]), dtype=np.uint8).reshape(60, 4)
    assert np.abs(p / 255 - w["probs"]).max() < 0.01
    for r in (row0, row1):
        snap_mod.write_stability(tmp_path, r, bars=w["bars"], actions=names, meta={"frames": 60})
    snap_mod.write_stability(tmp_path, dict(row1, entropy=0.5), bars=None, actions=names)   # rewrite keeps one row per eval
    data = json.loads((tmp_path / "stability.json").read_text(encoding="utf-8"))
    assert [e["i"] for e in data["evals"]] == [0, 1] and data["evals"][1]["entropy"] == 0.5
    assert data["bars"]["c"] == w["bars"]["c"] and data["actions"] == names and data["frames"] == 60
    assert "settled_since" in data and data["settled_since"] is None
    rows = [{"i": i, "changed": c} for i, c in enumerate([None, 0.5, 0.3, 0.05, 0.02, 0.01])]
    assert snap_mod.settled_since(rows) == 2              # evals 4..6 all changed <= 10%
    assert snap_mod.settled_since(rows[:3]) is None
    assert snap_mod.settled_since(rows[:4]) is None       # only one later eval below the threshold


def test_gap_verdict_rules():
    mk = lambda tr, va: [{"i": i, "step": (i + 1) * 10000, "train": {"return_pct": t}, "val": {"return_pct": v}}
                         for i, (t, v) in enumerate(zip(tr, va))]
    assert snap_mod.gap_verdict(mk([1, 2], [1, 1]))["cls"] == "grey"
    v = snap_mod.gap_verdict(mk([0, 1, 2, 3], [1, 0.5, 0, -0.5]))
    assert v["cls"] == "red" and "OVERFIT" in v["t"] and "+0.0% → +3.0%" in v["d"]
    assert snap_mod.gap_verdict(mk([0, 1, 2, 3], [0, 0.5, 1, 2]))["t"] == "GENERALISING"
    v = snap_mod.gap_verdict(mk([3, 3, 3, 3, 3, 3], [4, 2.5, 2.4, 2.3, 2.5, 2.4]))
    assert v["cls"] == "amber" and v["t"] == "VALIDATION PEAKED AT EVAL 1" and "step 10,000" in v["d"]
    v = snap_mod.gap_verdict(mk([8, 8, 8, 8], [-1, -1, -1, -1]))
    assert v["t"] == "MEMORISING THE TRAIN SLICE" and "gap +9.0 pp" in v["d"]
    assert snap_mod.gap_verdict(mk([1, 1, 1, 1], [0, 0, 0.1, 0]))["t"] == "NO CLEAR TREND"
    assert snap_mod.gap_verdict(mk([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 1, 2]))["t"] == "GENERALISING"   # window = last 4
