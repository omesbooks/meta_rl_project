"""Web GUI backend (webgui/): log parsing, the subprocess runner, form validation and the
review step - all headless. Command parity with the CTk app lives in
test_ui_training_contract.py (DashboardContractTests.test_web_command_equals_ctk_*)."""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from webgui import commands, logparse  # noqa: E402
from webgui.api import Api  # noqa: E402
from webgui.runner import Runner  # noqa: E402


def test_logparse_matches_the_ctk_rules():
    ev = logparse.event("|    approx_kl            | 0.012       |", "train", total=200000)
    assert ev["metric"] == {"name": "approx_kl", "value": 0.012, "status": "good", "label": ev["metric"]["label"]}
    assert ev["cls"] == "success"
    assert logparse.event("|    total_timesteps      | 36864       |", "train", total=200000)["progress"] == {"step": 36864, "total": 200000}
    assert logparse.event(" 18%|##  | 36,864/200,000 [03:42<16:21]", "train")["progress"] == {"step": 36864, "total": 200000}
    v = logparse.event(r"VIZ_ROLLOUT 0 2048 C:\x\runs\r1\viz\rollout_latest.json", "train")["viz"]
    assert v == {"kind": "rollout", "index": 0, "step": 2048, "path": r"C:\x\runs\r1\viz\rollout_latest.json"}
    assert logparse.classify("Traceback (most recent call last):") == "error"
    assert logparse.classify("[save] model written") == "success"
    assert "progress" not in logparse.event("|    total_timesteps | 10 |", "backtest", total=100)


def test_runner_streams_lines_and_reports_rc():
    r = Runner(ROOT)
    assert r.start([sys.executable, "-c", "print('a'); print('b'); raise SystemExit(3)"], "train")
    assert not r.start([sys.executable, "-c", "print(1)"], "train")          # one at a time
    got = []
    for _ in range(100):
        got += r.drain()
        if got and got[-1][0] == "done":
            break
        time.sleep(0.05)
    assert [g[2] for g in got if g[0] == "line"] == ["a", "b"]
    assert got[-1] == ("done", "train", {"rc": 3, "cancelled": False}) and not r.is_running()


def test_runner_stop_cancels():
    r = Runner(ROOT)
    r.start([sys.executable, "-c", "import time\nprint('x', flush=True)\ntime.sleep(30)"], "train")
    time.sleep(0.6)
    r.stop()
    for _ in range(100):
        done = [g for g in r.drain() if g[0] == "done"]
        if done:
            assert done[0][2]["cancelled"] is True
            return
        time.sleep(0.05)
    pytest.fail("runner did not stop")


def _csv(tmp_path):
    from test_training_improvements import data
    p = tmp_path / "web.csv"; data(600).to_csv(p, index=False); p.with_suffix(".params.json").write_text("{}")
    return p


def test_form_errors_name_the_field(tmp_path):
    f = commands.form_defaults(); f["csv"] = str(_csv(tmp_path))
    for key, bad in (("steps", "3.5"), ("window", "0"), ("train_pct", "0"), ("corr_threshold", "0"), ("mc_skip", "100"), ("name", "  ...  ")):
        g = dict(f, **{key: bad})
        with pytest.raises(commands.FormError) as e:
            commands.build_train_command(g)
        assert e.value.field == key, (key, e.value.field)
    with pytest.raises(commands.FormError) as e:
        commands.build_train_command(dict(f, reward_values=dict(f["reward_values"], trade_penalty=99)))
    assert e.value.field == "reward_values.trade_penalty"
    with pytest.raises(commands.FormError) as e:
        commands.build_train_command(dict(f, csv=""))
    assert e.value.field == "csv"


def test_web_only_options_are_appended_only_when_set(tmp_path):
    f = commands.form_defaults(); f["csv"] = str(_csv(tmp_path))
    base = commands.build_train_command(f)["cmd"]
    for flag in ("--entry_gate", "--spread_pct", "--commission", "--seed"):
        assert flag not in base
    cmd = commands.build_train_command(dict(f, entry_gate="srb_sig", spread_pct="3.8e-5", commission="0", seed="7"))["cmd"]
    tail = cmd[len(base):]
    assert tail == ["--entry_gate", "srb_sig", "--spread_pct", "3.8e-05", "--commission", "0", "--seed", "7"]
    assert commands.suggest_steps(600, "80", "2")["steps"] == 100000


def test_api_review_prepares_the_recipe_without_launching(tmp_path):
    csv = _csv(tmp_path)
    api = Api(ROOT, emit=lambda ev: None)
    f = commands.form_defaults()
    f.update(csv=str(csv), name="web_review_tmp", steps="32", window="2", max_hold="12", train_pct="80", mc_eval="0", nsteps="16", batch="8", nepochs="1", viz=False)
    r = api.review_training(f)
    assert r["ok"], r
    assert len(r["recipe_sha256"]) == 64 and r["token"] and not api.busy()
    assert dict(r["summary"])["Features after selection"] == "2"
    assert api.confirm_training("not-a-token")["ok"] is False
    bad = api.review_training(dict(f, steps="abc"))
    assert bad["ok"] is False and bad["field"] == "steps"
    info = api.csv_info(str(csv), "80", "2")
    assert info["ok"] and info["rows"] == 600 and info["params"] and [s[0] for s in info["split"]] == ["Train", "Validation", "Test"]
    cat = api.catalog()
    assert {p["key"] for p in cat["reward_profiles"]} >= {"balanced", "anti_overtrade"} and "approx_kl" in cat["metrics"]
