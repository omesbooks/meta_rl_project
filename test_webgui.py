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

    def work():
        try:
            out["p"] = mr.export_report_pdf(rows, None, tmp_path)
        except Exception as exc:   # surface the real error instead of a KeyError below
            out["err"] = exc
    t = threading.Thread(target=work)
    t.start(); t.join(240)        # the full suite can make this slow; 60 s was flaky
    assert not t.is_alive(), "PDF export did not finish in 240 s"
    assert "err" not in out, out.get("err")
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
    # a stored report says when it was made and on which data (the panel is never a live feed)
    rep = Api(ROOT).backtest_report("rl_uj_extra")
    assert rep["ok"] and rep["created_at"][:4].isdigit() and "T" not in rep["created_at"] and rep["data_csv"].endswith(".csv")


# ---------------------------------------------------------------- phase 4: Pipeline / Walk-Forward / Fine-tune
def test_runner_sequence_is_gated_on_exit_code():
    r = Runner(ROOT)
    stages = [("A", [sys.executable, "-c", "print('a')"]), ("B", [sys.executable, "-c", "raise SystemExit(2)"]),
              ("C", [sys.executable, "-c", "print('never')"])]
    assert r.start_sequence(stages, "pipeline")
    assert not r.start([sys.executable, "-c", "print(1)"], "train")            # busy for the whole sequence
    got = []
    for _ in range(200):
        got += r.drain()
        if got and got[-1][0] == "done":
            break
        time.sleep(0.05)
    kinds = [(k, d.get("name") if isinstance(d, dict) else d) for k, _p, d in got]
    assert ("stage", "A") in kinds and ("stage_done", "A") in kinds and ("stage", "B") in kinds
    assert ("stage", "C") not in kinds and ("line", "never") not in kinds
    assert got[-1][2] == {"rc": 2, "cancelled": False, "failed_stage": "B"} and not r.is_running()


def test_walkforward_lines_and_flow_errors(tmp_path):
    ev = logparse.event("WF_PROGRESS 2 4 window 1/2 test", "walkfwd")
    assert ev["wf"] == {"progress": [2, 4, "window 1/2 test"]}
    v = logparse.event("VERDICT: NOT ROBUST - 1/5 windows passed", "walkfwd")["wf"]["verdict"]
    assert (v["title"], v["tone"]) == ("NOT ROBUST", "bad")
    assert logparse.event("VERDICT: ROBUST", "walkfwd")["wf"]["verdict"]["title"] == "ROBUST"
    assert "progress" in logparse.event("|    total_timesteps      | 2048       |", "pipeline", total=4096)
    api = Api(tmp_path)
    assert api.review_pipeline({"csv": ""})["field"] == "pipe_csv"
    assert api.start_walkforward({"csv": ""})["field"] == "wf_csv"
    assert api.start_finetune({"base": "m", "old_csv": "a.csv", "new_csv": "b.csv", "name": "M", "steps": "10", "mix": "0.3", "lr": "1e-4"})["field"] == "name"
    assert not api.busy()


# ---------------------------------------------------------------- phase 5: Data Tools / Regime / Analyze
def _frame(n=300, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, n).cumsum()
    f1 = rng.normal(size=n)
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="D"), "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1, "f1": f1, "f1_copy": f1 * 2 + 0.001,
                         **{f"f{i}": rng.normal(size=n) for i in range(2, 12)}})


def test_data_tools_split_and_clean(tmp_path):
    import data_tools
    import pandas as pd
    _frame().to_csv(tmp_path / "d.csv", index=False)
    (tmp_path / "d.params.json").write_text("{}", encoding="utf-8")
    logs, notices = [], []
    log = lambda t, tag="info": logs.append((t, tag))
    notice = lambda title, message, paths=None, level="success", popup=True: notices.append(title)
    s = {"csv": "d.csv", "split_date": "2020-08-01", "train_suffix": "_train", "test_suffix": "_test", "clean_after": True, "clean_threshold": "0.99"}
    written = data_tools.split_csv(s, tmp_path, log, notice)
    assert {p.name for p in written} == {"d_train.csv", "d_test.csv", "d_train.params.json", "d_test.params.json"}
    tr = pd.read_csv(tmp_path / "d_train.csv")
    assert "f1_copy" not in tr.columns and "f1" in tr.columns and tr.timestamp.max() < "2020-08-01"
    assert notices[-1] == "Created split files"
    assert data_tools.split_csv(dict(s, split_date="1990-01-01"), tmp_path, log, notice) == []      # empty train -> aborted
    out = data_tools.clean_features("d.csv", 0.99, 0.85, tmp_path, log, notice)
    assert out[0].name == "d_clean.csv" and (tmp_path / "d_clean.features.json").exists()
    assert data_tools.clean_deploy_name("1st model-v2") == "m_1st_model_v2" and data_tools.clean_deploy_name("@@") == ""
    plan = data_tools.export_plan("m", "m", "", data_tools.EXPORT_SOURCES[1], tmp_path)
    assert plan["cmd"][1:] == ["export_to_onnx.py", "m", "--name", "m", "--source", "best"] and plan["outdir"] == ""


def test_regime_results_table(tmp_path):
    import json
    import regime_tools
    (tmp_path / "regime_single_data.json").write_text(json.dumps({
        "breaks": ["2008-09-20", "2016-06-01", "2019-01-01", "bad"], "events": {"Lehman": "2008-09-15", "Brexit": "2016-06-23", "Truss": "2022-09-26"},
        "method": "hmm", "score": 2, "n_bars": 1234, "time_sec": 1.5, "csv": "x.csv"}), encoding="utf-8")
    r = regime_tools.load_results(tmp_path)
    assert r["rows"] == [["2008-09-20", "✓ Lehman", "5d"], ["2016-06-01", "✓ Brexit", "22d"], ["2019-01-01", "—", "—"], ["bad", "—", "—"]]
    assert r["summary"].startswith("Method: HMM  ·  4 breakpoints  ·  2/3 known events matched") and r["good"]
    assert regime_tools.event_status_text(tmp_path).startswith("● No known_events.json")
    assert regime_tools.cutoff_target(tmp_path / "eu.csv", "2016-06-23").name == "eu_from_2016-06-23.csv"


def test_tool_job_streams_log_notice_and_done(tmp_path):
    _frame().to_csv(tmp_path / "d.csv", index=False)
    events = []
    api = Api(tmp_path, emit=events.extend)
    assert api.tools_split({"csv": "d.csv", "split_date": "2020-08-01"})["ok"]
    assert api.busy() or any(e.get("type") == "done" for e in events)
    for _ in range(100):
        if any(e.get("type") == "done" and e.get("page") == "tools" for e in events):
            break
        time.sleep(0.1)
    done = next(e for e in events if e.get("type") == "done")
    assert done["kind"] == "split" and done["ok"] and len(done["paths"]) == 2
    assert any(e.get("type") == "notice" and e["title"] == "Created split files" for e in events)
    assert any(e.get("type") == "line" and e["line"].startswith("✓ Saved: d_train.csv") and e["cls"] == "success" for e in events)
    assert not api.busy()
    assert api.tools_clean({"csv": "", "threshold": "0.99", "train_pct": "85"})["field"] == "tool_feat_csv"
    assert api.tools_correlation({"csv": "d.csv", "threshold": "0", "train_pct": "85"})["field"] == "tool_feat_thr"
    assert api.start_analyze({"model": "", "csv": "d.csv"})["field"] == "an_model"
    api.shutdown()


# ---------------------------------------------------------------- phase 6: launcher + startup helpers
def test_launcher_finds_the_project_and_startup_helpers(tmp_path, monkeypatch):
    import importlib.util
    import rl_web
    spec = importlib.util.spec_from_file_location("rl_studio_launcher", ROOT / "tools" / "build" / "rl_studio_launcher.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert launcher.project_root() == ROOT                         # walks up from tools/build/
    assert isinstance(rl_web.webview2_installed(), bool)
    ico = rl_web.window_icon_file()
    assert ico is None or ico.suffix == ".ico"
    calls = []
    monkeypatch.setattr(rl_web.subprocess, "call", lambda cmd, cwd=None: calls.append(cmd) or 0)
    assert rl_web.main(["--classic"]) == 0 and calls[0][-1].endswith("rl_app.py")
    monkeypatch.setattr(rl_web, "webview2_installed", lambda: False)
    assert rl_web.main(["--smoke"]) == 2                             # no WebView2: smoke fails, no window
    calls.clear()
    assert rl_web.main([]) == 0 and calls and calls[0][-1].endswith("rl_app.py")   # ...and the app falls back
    bat = (ROOT / "run_rl_app.bat").read_text(encoding="ascii")
    assert 'set "APP=rl_web.py"' in bat and "--classic" in bat and "\r\n" in (ROOT / "run_rl_app.bat").read_bytes().decode("ascii")


def test_live_visual_follows_the_app_theme(tmp_path):
    api = Api(ROOT, settings_dir=tmp_path)
    cmd = api._viz_cmd(["--run_dir", "x"])
    assert cmd[1].endswith("live_viewer.py") and "--query" not in cmd           # Clear Liquid -> light 3D scene (default)
    api.save_web_settings({"theme": "deep"})
    cmd = api._viz_cmd(["--run_dir", "x"])
    assert cmd[cmd.index("--query") + 1] == "theme=dark"                         # Deep Glass -> the dark neon scene
    html = (ROOT / "tools" / "viz" / "network3d.html").read_text(encoding="utf-8")
    assert "const LIGHT = document.body.classList.contains(\"clear\")" in html and "THREE.AdditiveBlending" in html
    assert html.count("blending: BLEND") >= 9 and "blending: THREE.AdditiveBlending" not in html
