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


# ---------------------------------------------------------------- phase 2: Models + Settings
def test_models_table_equals_the_ctk_rows():
    """Exit test of phase 2: the web listing is the classic Models table, row for row."""
    import model_results as mr
    import rl_app

    class Fake:                                    # the CTk methods without building a window
        _model_rows_cache_sig = None
        _model_rows_cache = None
        _model_results_signature = lambda self: None
        _collect_model_rows = rl_app.RLTradingStudio._collect_model_rows

    ctk_rows = Fake()._collect_model_rows()
    web = Api(ROOT).list_models()
    assert [r["iid"] for r in web["rows"]] == [r["iid"] for r in ctk_rows]
    assert [r["cells"] for r in web["rows"]] == [list(mr.table_values(r, ROOT)) for r in ctk_rows]
    assert web["headings"][7] == "Realism" and len(web["headings"]) == 10


def test_metrics_and_env_rows_from_a_trades_file(tmp_path):
    import model_results as mr
    (tmp_path / "zz_webtest_trades.csv").write_text("pnl_dollars\n100\n-50\n200\n-25\n", encoding="utf-8")
    m = mr.metrics_from_trades(tmp_path / "zz_webtest_trades.csv", 10000)
    assert m["trades"] == 4 and m["win_rate"] == 50.0
    assert m["profit_factor"] == pytest.approx(300 / 75) and m["return_pct"] == pytest.approx(2.25)
    row = next(r for r in mr.collect_model_rows(tmp_path) if r["model"] == "zz_webtest")
    assert row["source"] == "env backtest"
    cells = mr.table_values(row, tmp_path)
    assert cells[:7] == ("zz_webtest", "env backtest", "4", "50.0%", "4.00", "+2.25%", f"{m['max_dd']:+.2f}%")
    assert mr.fmt_num(float("inf")) == "inf" and mr.fmt_pct(None) == "-"


def test_settings_round_trip_and_validation(tmp_path):
    import json
    titles = []
    api = Api(ROOT, settings_dir=tmp_path, set_title=titles.append)
    st = api.settings_get()
    assert st["saved"] is False and st["branding"]["window_title"] == "Metafxclub RL Studio"
    form = {"branding": dict(st["branding"], window_title="My Studio"), "colors": dict(st["colors"], ACCENT="#00d4ff")}
    bad = api.save_branding({"branding": form["branding"], "colors": dict(form["colors"], RED="red")})
    assert bad["ok"] is False and bad["field"] == "color_RED" and not (tmp_path / "branding_config.json").exists()
    ok = api.save_branding(form)
    assert ok["ok"] and ok["branding"]["window_title"] == "My Studio" and ok["colors"]["ACCENT"] == "#00d4ff" and titles == ["My Studio"]
    saved = json.loads((tmp_path / "branding_config.json").read_text(encoding="utf-8"))
    assert set(saved) == {"branding", "colors"} and saved["colors"]["ACCENT"] == "#00d4ff"   # the CTk file format
    assert api.reset_branding()["saved"] is False and not (tmp_path / "branding_config.json").exists()
    assert api.save_web_settings({"theme": "deep", "brand_accent": True})["web"] == {"theme": "deep", "brand_accent": True}
    assert api.save_web_settings({"theme": "nope"})["web"]["theme"] == "deep"


def test_api_key_is_saved_but_never_sent_to_the_page(tmp_path):
    from webgui.api import JsBridge, to_json
    api = Api(ROOT, settings_dir=tmp_path)
    key = "AIzaTESTkey0123456789abcdWXYZ"
    assert api.save_api_key("")["field"] == "gemini_key"
    assert api.save_api_key("has space in it 1234567890")["ok"] is False
    r = api.save_api_key(key)
    assert r["ok"] and r["keys"] == {"gemini": {"set": True, "tail": "WXYZ"}}
    assert '"gemini": "' + key in (tmp_path / "api_keys.json").read_text(encoding="utf-8")
    bridge = JsBridge(api)
    assert key not in bridge.settings_get() and key not in to_json(r)
    assert not any(key in str(v) for v in vars(JsBridge).values())
    assert api.clear_api_key()["keys"]["gemini"]["set"] is False


def test_branding_defaults_match_rl_app():
    """app_settings.BRANDING_TEXT_DEFAULTS must equal the literals in rl_app.BRANDING."""
    import ast
    import app_settings
    tree = ast.parse((ROOT / "rl_app.py").read_text(encoding="utf-8"))
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BRANDING")
    literals = {k.value: v.value for k, v in zip(node.keys, node.values) if isinstance(v, ast.Constant)}
    assert {k: literals[k] for k in app_settings.BRANDING_TEXT_DEFAULTS} == app_settings.BRANDING_TEXT_DEFAULTS


def test_report_pdf_is_written_from_any_thread(tmp_path):
    """export_report_pdf uses Figure objects, so the web bridge thread can call it."""
    import threading
    import model_results as mr
    rows = mr.collect_model_rows(ROOT)[:3]
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("p", mr.export_report_pdf(rows, None, tmp_path)))
    t.start(); t.join(60)
    assert out["p"].exists() and out["p"].read_bytes()[:4] == b"%PDF" and out["p"].parent == tmp_path


# ---------------------------------------------------------------- phase 3: Backtest
def test_m1_auto_select_follows_the_ctk_rule(tmp_path):
    import backtest_report as br
    (tmp_path / "eu_h4_m1.csv").write_text("x\n", encoding="utf-8")
    # derived names inherit the original collector's M1 file
    for derived in ("eu_h4.csv", "eu_h4_train.csv", "training_data_eu_h4_clean.csv", "eu_h4_from_2020-01-01.csv"):
        assert br.find_matching_m1(derived, tmp_path) == "eu_h4_m1.csv", derived
    assert br.find_matching_m1("gu_h4.csv", tmp_path) is None
    assert br.auto_select_m1("eu_h4.csv", "(none)", None, tmp_path) == ("eu_h4_m1.csv", "eu_h4_m1.csv")
    assert br.auto_select_m1("gu_h4.csv", "eu_h4_m1.csv", "eu_h4_m1.csv", tmp_path) == ("(none)", None)   # stale auto pick is reset
    assert br.auto_select_m1("gu_h4.csv", "my_m1.csv", None, tmp_path) == (None, None)                    # explicit choice kept
    api = Api(tmp_path)
    assert api.backtest_m1("eu_h4.csv", "", None) == {"value": "eu_h4_m1.csv", "auto": "eu_h4_m1.csv"}


def test_skip_preview_and_report_rows(tmp_path):
    import backtest_report as br
    import pandas as pd
    ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="4h"))
    text, level = br.skip_preview(ts, "80")
    assert text == "Skipped: 2024-01-01 -> 2024-01-14 04:00 (80 rows)\nBacktest: 2024-01-14 08:00 -> 2024-01-17 12:00 (20 rows)" and level == ""
    assert br.skip_preview(ts, "100%")[1] == "warn"
    (tmp_path / "d.csv").write_text("timestamp,close\n" + "\n".join(f"{t},1" for t in ts), encoding="utf-8")
    assert Api(tmp_path).backtest_range("d.csv", "80")["text"] == text
    meta = {"model": "m", "data_csv": "d.csv", "backtest_rows": 20, "settings": {"mode": "pure_agent"},
            "result": {"total_trades": 3, "win_rate": 2 / 3, "profit_factor": 2.5, "return_pct": 0.031, "max_drawdown": -0.012,
                       "random_baseline": {"beats": 19, "runs": 20, "percentile": 0.95, "verdict": "EDGE"},
                       "monte_carlo": {"dd_median": -0.02}}}
    lines = br.report_lines(meta)
    rows = br.report_rows(lines)
    assert rows[0][0] == "Run" and ["", ""] in rows
    assert ["Baseline", "beats 19/20 random agents (95%)  ·  random median —"] in rows and ["", "-> EDGE"] in rows
    assert any(r[0] == "MonteCarlo" and r[1].startswith("DD median -2.00%") for r in rows)
    assert br.stat_cards(meta) == {"win_rate": ("66.7%", ""), "profit_factor": ("2.50", "good"), "return": ("3.10%", "good"), "max_dd": ("-1.20%", "bad")}
    assert br.stat_cards({"result": {"total_trades": 0}})["return"] == ("—", "dim")


def test_backtest_api_errors_name_the_field(tmp_path):
    api = Api(tmp_path)
    assert api.start_backtest({"model": "", "csv": "x.csv"})["field"] == "bt_model"
    r = api.start_backtest({"model": "m", "csv": "x.csv", "max_pos": "0"})
    assert r["ok"] is False and r["field"] == "max_pos" and not api.busy()
    assert api.generate_chart({"model": "zz_never_trained", "csv": "x.csv"})["field"] == "bt_model"
    assert api.backtest_report("zz_never_trained")["missing"] is True
    cat = Api(ROOT).backtest_catalog()
    assert cat["m1_options"][0] == "(none)" and not any(d.endswith("_m1.csv") for d in cat["datasets"])
