"""Dashboard -> confirmed command -> actual training metadata contract tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import artifact_paths as paths
import rl_app
from action_profiles import coerce_action_params
from rl_train import build_parser, prepare_training
from ui_values import fraction, number, percent_units
from test_training_improvements import data

ROOT = Path(__file__).parent.resolve()


def set_entry(entry, value):
    entry.delete(0, "end")
    entry.insert(0, str(value))


def prepare(cmd):
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_training(build_parser().parse_args(cmd[2:]))


class StrictInputTests(unittest.TestCase):
    def test_percent_formats_and_boundaries(self):
        for raw in ("85", "85%", ".85", "0.85"):
            self.assertEqual(fraction(raw), .85)
        self.assertEqual(fraction("1%"), .01)
        self.assertEqual(fraction("1"), 1)
        self.assertEqual(percent_units("-0.03", "swap"), -.0003)
        for raw in ("", "bad", "nan", "inf", "-1", "101", "-inf"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                fraction(raw)
        with self.assertRaises(ValueError):
            fraction("0", allow_zero=False)
        with self.assertRaises(ValueError):
            fraction("100%", allow_one=False)

    def test_no_numeric_fallback_or_fractional_integer(self):
        for raw in ("", "nan", "inf", "3.5", "1,,000"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                number(raw, "steps", 1, integer=True)
        for raw in ("2,000", "2_000", "2e3"):
            self.assertEqual(number(raw, "steps", 1, integer=True), 2000)
        with self.assertRaises(ValueError):
            coerce_action_params({"trail_atr_period": 2.5})


class DashboardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = rl_app.RLTradingStudio()
        cls.app.withdraw()
        for page in ("train", "pipeline", "walkfwd", "backtest", "finetune", "tools", "regime"):
            cls.app.show_page(page)

    @classmethod
    def tearDownClass(cls):
        for callback in cls.app.tk.call("after", "info"):
            cls.app.tk.call("after", "cancel", callback)
        cls.app.destroy()

    def setUp(self):
        for name in ("showinfo", "showerror", "showwarning"):
            p = patch.object(rl_app.messagebox, name)
            mocked = p.start()
            self.addCleanup(p.stop)
            if name != "showinfo":
                mocked.side_effect = lambda title, message: self.fail(f"{title}: {message}")
        self.tmp = tempfile.TemporaryDirectory(prefix="ui_recipe_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / "raw.csv"
        data(600).to_csv(self.csv, index=False)
        self.csv.with_suffix(".params.json").write_text("{}")
        self.models = self.root / "models"
        for obj, attr, value in ((paths, "MODELS_DIR", self.models), (rl_app, "MODELS_DIR", self.models),
                                 (rl_app, "WORK_DIR", self.root),
                                 (rl_app, "DEFAULT_ACTION_PROFILE_CONFIG_DIR", self.root / "actions"),
                                 (rl_app, "DEFAULT_REWARD_PROFILE_CONFIG_DIR", self.root / "rewards")):
            p = patch.object(obj, attr, value)
            p.start()
            self.addCleanup(p.stop)
        paths._READ_OUTPUTS.clear()
        app = self.app
        app._preview_pending = False
        app.pipeline_running = False
        app.wf_action_profile_json_value = ""
        app.train_csv_path = str(self.csv)
        app.train_reward_profile_json_path = ""
        app.train_action_profile_json_path = ""
        app.train_reward_profile_json_has_formula = False
        app.train_reward_json_overrides = {}
        app.train_action_json_params = {}
        app.reward_formula_enabled.set(False)
        app.train_reward_profile.set(rl_app.REWARD_PROFILES["balanced"]["label"])
        app.train_action_profile.set(rl_app.ACTION_PROFILES["basic_4"]["label"])
        app._apply_train_reward_profile_defaults()
        app._apply_train_action_profile_defaults()
        app.train_reward.set("realized (recommended)")
        for key, val in {"name":"ui_model", "steps":32, "window":2, "maxhold":12,
                         "pct":80, "mc_eval":0, "mc_skip":10, "lr":"0.0002", "clip":.15,
                         "ent":.02, "nsteps":16, "nepochs":1, "batch":8,
                         "gamma":.97, "gae":.9, "vf":.6}.items():
            set_entry(getattr(app, "train_"+key), val)
        app.pipe_csv.set(str(self.csv))
        app.pipe_bt_csv.set(str(self.csv))
        for key, val in {"model_name":"ui_pipe", "steps":32, "window":2, "conf":0,
                         "train_pct":80, "risk":.01, "max_pos":1, "sl":2,
                         "tp":4, "bt_start":"auto"}.items():
            set_entry(getattr(app, "pipe_"+key), val)
        for key, val in {"lr":.0002, "clip":.15, "ent":.02, "nsteps":16, "nepochs":1,
                         "batch":8, "gamma":.97, "gae":.9, "vf":.6, "max_hold":12,
                         "ep_len":32, "net_arch":"8"}.items():
            set_entry(app.pipe_hparams[key], val)
        set_entry(app.pipe_reward_overrides, "")
        set_entry(app.pipe_reward_formula, "")
        set_entry(app.pipe_action_params, "")

    def capture_train(self):
        with patch.object(self.app, "_review_training_command") as review, \
                patch.object(self.app, "_start_runner") as start:
            self.app._start_training()
            start.assert_not_called()
            self.assertEqual(review.call_count, 1)
        return review.call_args.args[0]

    def run_confirmed(self, cmd, prepared):
        args = list(cmd[2:]) + ["--expected_recipe_sha256", prepared["recipe_sha256"]]
        wrapper = (
            "import sys,runpy; from pathlib import Path; import artifact_paths; "
            f"artifact_paths.MODELS_DIR=Path({str(self.models)!r}); "
            f"sys.argv={['rl_train.py', *args]!r}; runpy.run_path('rl_train.py',run_name='__main__')"
        )
        done = subprocess.run([sys.executable, "-c", wrapper], cwd=ROOT,
                              env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(done.returncode, 0, (done.stdout+done.stderr)[-5000:])
        name = prepared["args"].name
        meta = json.loads(paths.train_meta_path(name).read_text(encoding="utf-8"))
        self.assertEqual(meta["confirmed_recipe"], prepared["recipe"])
        self.assertEqual(meta["recipe_sha256"], prepared["recipe_sha256"])
        self.assertEqual(meta["features"], prepared["recipe"]["features"])
        self.assertEqual(meta["hyperparameters"]["reward_profile_config"], prepared["recipe"]["reward_config"])
        self.assertEqual(meta["hyperparameters"]["action_profile_config"], prepared["recipe"]["action_config"])
        for key in ("learning_rate", "clip_range", "ent_coef", "gamma", "gae_lambda", "vf_coef", "n_steps", "n_epochs", "batch_size", "max_hold", "window"):
            self.assertEqual(meta["hyperparameters"][key], prepared["recipe"]["settings"][key])
        self.assertEqual(meta["train_period"], prepared["recipe"]["train_period"])
        self.assertEqual(meta["validation_period"], prepared["recipe"]["validation_period"])
        return meta

    def test_train_defaults_summary_and_actual_metadata(self):
        cmd = self.capture_train()
        prepared = prepare(cmd)
        self.assertEqual(len(prepared["recipe"]["features"]), 2)
        self.assertFalse(self.models.exists())
        self.run_confirmed(cmd, prepared)

    def test_custom_json_formula_and_slider_precedence(self):
        reward = self.root/"reward.json"
        reward.write_text(json.dumps({"schema":"metafxclub.reward_profile.v1", "base_profile":"balanced",
                                     "overrides":{"trade_penalty":.012},
                                     "developer_mode":{"enabled":True,"formula":"trade_closed_pnl * 40"}}))
        action = self.root/"action.json"
        action.write_text(json.dumps({"schema":"metafxclub.action_profile.v1", "base_profile":"manage_6",
                                     "params":{"trail_atr_mult":3.5}}))
        with patch.object(rl_app.filedialog, "askopenfilename", return_value=str(reward)):
            self.app._load_train_reward_json()
        with patch.object(rl_app.filedialog, "askopenfilename", return_value=str(action)):
            self.app._load_train_action_json()
        self.app._apply_train_reward_overrides_to_controls({"trade_penalty":.015})
        cmd = self.capture_train()
        prepared = prepare(cmd)
        self.assertEqual(prepared["recipe"]["reward_config"]["trade_penalty"], .015)
        self.assertEqual(prepared["recipe"]["reward_formula"], "trade_closed_pnl * 40")
        self.assertEqual(len(prepared["recipe"]["action_config"]["actions"]), 6)
        self.assertEqual(prepared["recipe"]["action_config"]["params"]["trail_atr_mult"], 3.5)
        self.run_confirmed(cmd, prepared)

    def test_developer_off_retains_json_numeric_values(self):
        path = self.root/"reward.json"
        path.write_text(json.dumps({"base_profile":"balanced", "overrides":{"trade_penalty":.014},
                                   "developer_mode":{"enabled":True,"formula":"trade_closed_pnl * 40"}}))
        with patch.object(rl_app.filedialog, "askopenfilename", return_value=str(path)):
            self.app._load_train_reward_json()
        self.app.reward_formula_enabled.set(False)
        cmd = self.capture_train()
        self.assertNotIn("--reward_profile_json", cmd)
        self.assertNotIn("--reward_formula", cmd)
        prepared = prepare(cmd)
        self.assertEqual(prepared["recipe"]["reward_config"]["trade_penalty"], .014)
        self.assertEqual(prepared["recipe"]["reward_formula"], "")
        self.run_confirmed(cmd, prepared)

    # ---- web GUI parity: webgui.commands must build the CTk command for the same inputs
    def web_form(self, **over):
        from webgui.commands import form_defaults
        f = form_defaults()
        f.update(csv=str(self.csv), name="ui_model", steps="32", window="2", max_hold="12", train_pct="80",
                 mc_eval="0", mc_skip="10", lr="0.0002", clip="0.15", ent="0.02", nsteps="16", nepochs="1",
                 batch="8", gamma="0.97", gae="0.9", vf="0.6", viz=False)
        f.update(over)
        return f

    def test_web_command_equals_ctk_command_defaults(self):
        from webgui.commands import build_train_command
        self.assertEqual(build_train_command(self.web_form())["cmd"], self.capture_train())

    def test_web_command_equals_ctk_command_with_json_and_sliders(self):
        from webgui.commands import build_train_command
        from reward_profiles import REWARD_PROFILES
        from action_profiles import ACTION_PROFILES
        reward = self.root/"reward.json"
        reward.write_text(json.dumps({"schema":"metafxclub.reward_profile.v1", "base_profile":"balanced",
                                     "overrides":{"trade_penalty":.012},
                                     "developer_mode":{"enabled":True,"formula":"trade_closed_pnl * 40"}}))
        action = self.root/"action.json"
        action.write_text(json.dumps({"schema":"metafxclub.action_profile.v1", "base_profile":"manage_6",
                                     "params":{"trail_atr_mult":3.5}}))
        with patch.object(rl_app.filedialog, "askopenfilename", return_value=str(reward)):
            self.app._load_train_reward_json()
        with patch.object(rl_app.filedialog, "askopenfilename", return_value=str(action)):
            self.app._load_train_action_json()
        self.app._apply_train_reward_overrides_to_controls({"trade_penalty":.015})
        ctk_cmd = self.capture_train()
        rv = dict(REWARD_PROFILES["balanced"]["params"], trade_penalty=.015)
        av = dict(ACTION_PROFILES["manage_6"]["params"], trail_atr_mult=3.5)
        web = self.web_form(reward_profile="balanced", action_profile="manage_6", reward_values=rv, action_values=av,
                            reward_formula_enabled=True, reward_formula="trade_closed_pnl * 40",
                            reward_json={"path": str(reward), "overrides": {"trade_penalty": .012}, "has_formula": True},
                            action_json={"path": str(action), "params": {"trail_atr_mult": 3.5}})
        self.assertEqual(build_train_command(web)["cmd"], ctk_cmd)

    def test_web_command_equals_ctk_command_developer_off(self):
        from webgui.commands import build_train_command
        from reward_profiles import REWARD_PROFILES
        path = self.root/"reward.json"
        path.write_text(json.dumps({"base_profile":"balanced", "overrides":{"trade_penalty":.014},
                                   "developer_mode":{"enabled":True,"formula":"trade_closed_pnl * 40"}}))
        with patch.object(rl_app.filedialog, "askopenfilename", return_value=str(path)):
            self.app._load_train_reward_json()
        self.app.reward_formula_enabled.set(False)
        ctk_cmd = self.capture_train()
        web = self.web_form(reward_values=dict(REWARD_PROFILES["balanced"]["params"], trade_penalty=.014),
                            reward_formula_enabled=False,
                            reward_json={"path": str(path), "overrides": {"trade_penalty": .014}, "has_formula": True})
        out = build_train_command(web)
        self.assertEqual(out["cmd"], ctk_cmd)
        self.assertTrue(out["warnings"])

    # ---- web GUI parity (phase 3): webgui.backtest must build the CTk backtest command
    def ctk_backtest(self, model="bt_probe", **widgets):
        app = self.app
        app.bt_model.set(model)
        app.bt_csv.set(str(self.csv))
        app.bt_m1_csv.set(widgets.pop("m1", "(none)"))
        app.bt_mode.set(widgets.pop("mode", "Pure Agent (matches training) ⭐"))
        app.bt_intrabar.set(widgets.pop("intrabar", "pessimistic (SL first) ⭐"))
        app.bt_model_source.set(widgets.pop("source", "Final ⭐"))
        values = {"conf": "0", "start": "0", "risk": "0.01", "max_pos": "1", "sl": "2.0", "tp": "4.0",
                  "stop_slip": "0.01", "swap_long": "0", "swap_short": "0", "random_baseline": "20",
                  "mc": "1000", "window": "auto"}
        values.update(widgets)
        for key, val in values.items():
            set_entry(getattr(app, "bt_" + key), val)
        with patch.object(app, "_start_runner") as start:
            app._run_backtest()
        self.assertEqual(start.call_count, 1)
        return start.call_args.args[0]

    def test_web_backtest_command_equals_ctk_defaults(self):
        from webgui.backtest import build_backtest_command, form_defaults
        form = dict(form_defaults(), model="bt_probe", csv=str(self.csv))
        self.assertEqual(build_backtest_command(form)["cmd"], self.ctk_backtest())

    def test_web_backtest_command_equals_ctk_full(self):
        from webgui.backtest import build_backtest_command, form_defaults
        meta = paths.train_meta_path("bt_meta")                     # max_hold must come from the model meta
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"hyperparameters": {"max_hold": 17}}), encoding="utf-8")
        m1 = self.root / "raw_m1.csv"
        m1.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
        widgets = {"conf": "0.35", "start": "85", "risk": "0.02", "max_pos": "1", "sl": "1.5", "tp": "3",
                   "stop_slip": "0.02", "swap_long": "-0.005", "swap_short": "0.001", "random_baseline": "5",
                   "mc": "0", "window": "15"}
        ctk = self.ctk_backtest("bt_meta", m1=str(m1), mode="Agent + SL/TP (live-realistic)",
                                intrabar="optimistic (TP first)", source="Best (pre-overfit)", **widgets)
        form = dict(form_defaults(), model="bt_meta", csv=str(self.csv), m1_csv=str(m1), mode="agent_sltp",
                    intrabar="optimistic", source="best", **widgets)
        out = build_backtest_command(form)
        self.assertEqual(out["cmd"], ctk)
        self.assertEqual((out["max_hold"], out["max_hold_source"]), (17, "model meta"))
        self.assertIn("--m1_csv", ctk)

    def test_web_backtest_rejects_what_ctk_rejects(self):
        from webgui.backtest import build_backtest_command, form_defaults
        from webgui.commands import FormError
        for key, raw in (("start", "100%"), ("window", ""), ("conf", "nan"), ("swap_long", "bad"), ("max_pos", "1.5"), ("mc", "-1")):
            with self.assertRaises(FormError) as cm:
                build_backtest_command(dict(form_defaults(), model="bt_probe", csv=str(self.csv), **{key: raw}))
            self.assertEqual(cm.exception.field, key)

    def test_ctk_report_text_equals_backtest_report(self):
        """The CTk report panel and the web report come from the same lines (stored meta copied in)."""
        import backtest_report
        src = sorted(Path(ROOT, "artifacts", "models").glob("*/backtests/*_live_bt.meta.json"))
        self.assertTrue(src, "needs at least one stored live-backtest meta in artifacts/models")
        meta = json.loads(src[0].read_text(encoding="utf-8-sig"))
        dst = paths.backtest_meta_path("bt_report_probe")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(meta), encoding="utf-8")
        self.app._load_backtest_report("bt_report_probe")
        self.assertEqual(self.app.bt_report_box.get("1.0", "end-1c"), "\n".join(backtest_report.report_lines(meta)))
        cards = backtest_report.stat_cards(meta)
        self.assertEqual(self.app.stat_pf.value_label.cget("text"), cards["profit_factor"][0])
        self.assertEqual(self.app.stat_ret.value_label.cget("text"), cards["return"][0])

    # ---- web GUI parity (phase 4): pipeline / walk-forward / fine-tune
    def web_pipe_form(self, **over):
        from webgui.pipeline import pipeline_defaults
        f = pipeline_defaults()
        f.update(csv=str(self.csv), bt_csv=str(self.csv), model_name="ui_pipe", steps="32", window="2", conf="0",
                 train_pct="80", risk="0.01", max_pos="1", sl="2", tp="4", bt_start="auto",
                 lr="0.0002", clip="0.15", ent="0.02", nsteps="16", nepochs="1", batch="8", gamma="0.97", gae="0.9",
                 vf="0.6", max_hold="12", ep_len="32", net_arch="8", use_m1=bool(self.app.pipe_use_m1.get()))
        f.update(over)
        return f

    def ctk_pipeline(self):
        """CTk Run full pipeline -> (review cmd, context, hparams handed to the worker, stage commands)."""
        with patch.object(self.app, "_review_training_command") as review, \
                patch.object(self.app, "_launch_pipeline") as launch:
            self.app._run_full_pipeline()
            cmd, callback = review.call_args.args
            context = review.call_args.kwargs["context"]
            prepared = prepare(cmd)
            self.app._resolve_pipeline_preview(prepared["recipe"], context)
            frozen = cmd + ["--expected_recipe_sha256", prepared["recipe_sha256"]]
            callback(frozen)                                    # still inside the patch: nothing launches
            args = launch.call_args.args
        stages = []
        with patch.object(self.app, "_pipeline_run_cmd", side_effect=lambda c, *a: stages.append(list(c))), \
                patch.object(self.app, "_pipeline_log"), \
                patch.object(self.app, "after", side_effect=lambda delay, fn: fn()), \
                patch.object(self.app, "_finish_pipeline"):
            self.app._pipeline_worker(*args)
        return cmd, context, prepared, frozen, stages

    def web_pipeline(self, form, prepared):
        import backtest_report
        import pipeline_shared
        from webgui.pipeline import build_pipeline, pipeline_stages
        plan = build_pipeline(form, self.root)
        frame = rl_app.validate_frame(pd.read_csv(self.root / plan["bt_csv"]), plan["bt_csv"])
        pipeline_shared.resolve_backtest(prepared["recipe"], plan["context"], frame,
                                         lambda c: backtest_report.find_matching_m1(c, self.root))
        return plan, pipeline_stages(plan, list(plan["train_cmd"]) + ["--expected_recipe_sha256", prepared["recipe_sha256"]])

    def test_web_pipeline_equals_ctk_defaults(self):
        cmd, context, prepared, frozen, ctk_stages = self.ctk_pipeline()
        plan, web_stages = self.web_pipeline(self.web_pipe_form(), prepared)
        self.assertEqual(plan["train_cmd"], cmd)
        self.assertEqual(plan["context"], context)
        self.assertEqual([c for _n, c in web_stages], ctk_stages)
        self.assertEqual([n for n, _c in web_stages], ["Train PPO", "Backtest", "Chart"])
        self.assertEqual(ctk_stages[0], frozen)

    def test_web_pipeline_equals_ctk_external_and_json(self):
        external = self.root / "later.csv"
        frame = data(120)
        frame.timestamp += pd.Timedelta(days=200)
        frame.to_csv(external, index=False)
        app = self.app
        app.pipe_bt_csv.set(str(external))
        set_entry(app.pipe_reward_overrides, '{"trade_penalty":0.013}')
        set_entry(app.pipe_reward_formula, "trade_closed_pnl * 30")
        set_entry(app.pipe_action_params, '{"trail_atr_mult":2.7}')
        set_entry(app.pipe_bt_start, "0.25")
        set_entry(app.pipe_conf, "0.4")
        app.pipe_bt_mode.set("Agent + SL/TP")
        app.pipe_action_profile.set(rl_app.ACTION_PROFILES["manage_6"]["label"])
        try:
            cmd, context, prepared, frozen, ctk_stages = self.ctk_pipeline()
            plan, web_stages = self.web_pipeline(self.web_pipe_form(
                bt_csv=str(external), reward_overrides='{"trade_penalty":0.013}', reward_formula="trade_closed_pnl * 30",
                action_params='{"trail_atr_mult":2.7}', bt_start="0.25", conf="0.4", bt_mode="agent_sltp",
                action_profile="manage_6"), prepared)
        finally:
            app.pipe_bt_mode.set("Pure Agent")
            app.pipe_action_profile.set(rl_app.ACTION_PROFILES["basic_4"]["label"])
        self.assertEqual(plan["train_cmd"], cmd)
        self.assertIn("--eval_csv", cmd)
        self.assertEqual(plan["context"], context)
        self.assertEqual([c for _n, c in web_stages], ctk_stages)

    def test_web_pipeline_rejects_what_ctk_rejects(self):
        from webgui.commands import FormError
        from webgui.pipeline import build_pipeline
        for key, raw in (("steps", "1.5"), ("conf", "2"), ("bt_start", "100%"), ("batch", "64"), ("net_arch", "8,x"),
                         ("reward_overrides", "{bad"), ("action_params", '{"nope":1}'), ("model_name", "///")):
            with self.assertRaises(FormError, msg=key):
                build_pipeline(self.web_pipe_form(**{key: raw}), self.root)

    def test_web_walkforward_equals_ctk_after_copy_from_train(self):
        from webgui.pipeline import build_wf_command, wf_defaults, wf_from_train
        app = self.app
        app.train_reward_json_overrides = {"trade_penalty": .012}
        app._apply_train_reward_overrides_to_controls({"trade_penalty": .012})
        app._wf_copy_from_train()
        app.wf_csv.set(str(self.csv))
        for k, v in (("windows", "3"), ("steps", "64"), ("name", "wf_parity")):
            set_entry(getattr(app, "wf_" + k), v)
        with patch.object(app, "_start_runner") as runner:
            app._run_walkforward()
        ctk = runner.call_args.args[0]
        from reward_profiles import REWARD_PROFILES
        copied = wf_from_train(self.web_form(reward_values=dict(REWARD_PROFILES["balanced"]["params"], trade_penalty=.012)))
        form = dict(wf_defaults(), csv=str(self.csv), windows="3", steps="64", name="wf_parity",
                    **{k: v for k, v in copied.items() if k != "hint"})
        self.assertEqual(build_wf_command(form)["cmd"], ctk)

    def test_web_finetune_equals_ctk(self):
        from webgui.commands import FormError
        from webgui.pipeline import build_ft_command, ft_defaults
        app = self.app
        app.ft_base.set("base")
        app.ft_old.set(str(self.csv))
        app.ft_new.set(str(self.csv))
        app.ft_mode.set("replay (all old+new)")
        for k, v in (("mix", "25%"), ("lr", "5e-5"), ("steps", "1,000"), ("name", "base_ft")):
            set_entry(getattr(app, "ft_" + k), v)
        try:
            with patch.object(app, "_start_runner") as runner:
                app._run_finetune()
        finally:
            app.ft_mode.set("mixed (old+new ⭐)")
        form = dict(ft_defaults(), base="base", old_csv=str(self.csv), new_csv=str(self.csv), mode="replay",
                    mix="25%", lr="5e-5", steps="1,000", name="base_ft")
        self.assertEqual(build_ft_command(form)["cmd"], runner.call_args.args[0])
        with self.assertRaises(FormError):
            build_ft_command(dict(form, name="BASE"))                  # same name as the base, case-insensitive

    # ---- web GUI parity (phase 5): analyze / regime check
    def test_web_analyze_and_regime_commands_equal_ctk(self):
        import regime_tools
        from webgui.api import Api
        app = self.app
        app.show_page("analyze")                                  # pages are built lazily
        app.an_model.set("probe")
        app.an_csv.set(str(self.csv))
        with patch.object(app, "_start_runner") as runner:
            app._run_analyze()
        api = Api(self.root, python=sys.executable)
        with patch.object(api, "_start_page_run", side_effect=lambda page, cmd, info, display=None: {"ok": True, "cmd": cmd}) as start:
            self.assertEqual(api.start_analyze({"model": "probe", "csv": str(self.csv)})["cmd"], runner.call_args.args[0])
        app.regime_csv_path = str(self.csv)
        for method, entries in (("hmm", {"hmm_states": "4"}), ("kmeans", {"km_k": "5", "km_win": "30"}), ("pelt", {"pelt_pen": "12.5"})):
            app.regime_method.set(method)
            for k, v in entries.items():
                set_entry(getattr(app, "regime_" + k), v)
            with patch.object(app, "_start_runner") as runner:
                app._run_regime_detection()
            params = {"hmm_states": app.regime_hmm_states.get(), "km_k": app.regime_km_k.get(),
                      "km_win": app.regime_km_win.get(), "pelt_pen": app.regime_pelt_pen.get()}
            self.assertEqual(regime_tools.detect_command(str(self.csv), method, params), runner.call_args.args[0], method)
        app.regime_method.set("hmm")
        with self.assertRaises(regime_tools.RegimeError):
            regime_tools.detect_command(str(self.csv), "pelt", {"pelt_pen": "0"})

    def test_web_gemini_command_masks_the_key_like_ctk(self):
        import regime_tools
        app = self.app
        app.regime_csv_path = str(self.csv)
        key = "AIzaCONTRACTkey000000000WXYZ"
        logged = []
        with patch.object(rl_app, "_load_api_keys", return_value={"gemini": key}), \
                patch.object(app, "_start_runner") as runner, \
                patch.object(app, "_log", side_effect=lambda w, text, tag="info": logged.append(text)):
            app._refresh_events_with_gemini()
        cmd, masked = regime_tools.gemini_command(str(self.csv), app.regime_ev_symbol.get(), app.regime_ev_topk.get(), key)
        self.assertEqual(cmd, runner.call_args.args[0])
        self.assertEqual(logged[-1], "$ " + masked)
        self.assertNotIn(key, masked)
        self.assertIn(key[:8] + "..." + key[-4:], masked)

    def test_web_regime_cutoff_writes_the_same_file_as_ctk(self):
        from webgui.api import Api
        app = self.app
        src = self.root / "regime_src.csv"
        frame = data(400)
        frame.to_csv(src, index=False)
        src.with_suffix(".params.json").write_text('{"p": 1}')
        date = str(frame.timestamp.iloc[250].date())
        for item in app.regime_tree.get_children():
            app.regime_tree.delete(item)
        app.regime_tree.insert("", "end", iid="b0", values=(date, "—", "—"))
        app.regime_tree.selection_set("b0")
        app.regime_csv_path = str(src)
        app._regime_results_csv = str(src)
        with patch.object(rl_app.messagebox, "askyesno", return_value=True), \
                patch.object(app, "show_page"), patch.object(app, "_set_train_csv") as chosen:
            app._use_regime_cutoff()
        out = self.root / f"regime_src_from_{date}.csv"
        ctk_bytes, ctk_params = out.read_bytes(), out.with_suffix(".params.json").read_bytes()
        self.assertEqual(chosen.call_args.args[0], str(out))
        out.unlink(); out.with_suffix(".params.json").unlink()
        r = Api(self.root).regime_cutoff(str(src), date, str(src))
        self.assertTrue(r["ok"], r)
        self.assertEqual(Path(r["path"]), out)
        self.assertEqual(out.read_bytes(), ctk_bytes)
        self.assertEqual(out.with_suffix(".params.json").read_bytes(), ctk_params)
        stale = Api(self.root).regime_cutoff(str(src), date, str(self.csv))
        self.assertEqual(stale.get("title"), "Stale breakpoint")

    def test_pipeline_external_eval_and_overrides_metadata(self):
        external = self.root/"later.csv"
        frame = data(120)
        frame.timestamp += pd.Timedelta(days=200)
        frame.to_csv(external, index=False)
        self.app.pipe_bt_csv.set(str(external))
        set_entry(self.app.pipe_reward_overrides, '{"trade_penalty":0.013}')
        set_entry(self.app.pipe_reward_formula, "trade_closed_pnl * 30")
        set_entry(self.app.pipe_action_params, '{"trail_atr_mult":2.7}')
        with patch.object(self.app, "_review_training_command") as review, \
                patch.object(self.app, "_launch_pipeline") as launch:
            self.app._run_full_pipeline()
            launch.assert_not_called()
            self.assertEqual(review.call_count, 1)
            cmd, callback = review.call_args.args
            prepared = prepare(cmd)
            context = review.call_args.kwargs["context"]
            self.app._resolve_pipeline_preview(prepared["recipe"], context)
            self.assertEqual(context["resolved_backtest"]["start"], .5)
            self.assertEqual(context["resolved_backtest"]["rows"], 60)
            frozen = cmd + ["--expected_recipe_sha256", prepared["recipe_sha256"]]
            callback(frozen)
            hparams = launch.call_args.args[-1]
            self.assertEqual(hparams["confirmed_train_cmd"], frozen)
        meta = self.run_confirmed(cmd, prepared)
        self.assertEqual(meta["eval_csv"], str(external))
        self.assertEqual(meta["hyperparameters"]["reward_formula"], "trade_closed_pnl * 30")
        self.assertEqual(meta["hyperparameters"]["reward_profile_config"]["trade_penalty"], .013)

        launched = []
        def run_stage(command, *unused):
            launched.append(command)
            wrapper = ("import sys,runpy; from pathlib import Path; import artifact_paths; "
                       f"artifact_paths.MODELS_DIR=Path({str(self.models)!r}); "
                       f"sys.argv={command[1:]!r}; runpy.run_path({command[1]!r},run_name='__main__')")
            done = subprocess.run([sys.executable, "-c", wrapper], cwd=ROOT,
                                  env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            self.assertEqual(done.returncode, 0, (done.stdout+done.stderr)[-4000:])
        with patch.object(self.app, "_pipeline_run_cmd", side_effect=run_stage), \
                patch.object(self.app, "_pipeline_log"), \
                patch.object(self.app, "after", side_effect=lambda delay, fn: fn()), \
                patch.object(self.app, "_finish_pipeline") as finish:
            self.app._pipeline_worker(str(self.csv), str(external), False, False,
                                     prepared["args"].name, 32, 2, 0, "pure_agent", .8, hparams)
        self.assertEqual(launched[0], frozen)
        self.assertEqual(len(launched), 3)
        self.assertTrue(finish.call_args.args[0], finish.call_args)
        self.assertEqual(float(launched[1][launched[1].index("--start")+1]), .5)

    def test_changed_data_or_json_requires_new_confirmation(self):
        cmd = self.capture_train()
        prepared = prepare(cmd)
        self.csv.with_suffix(".params.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "changed after confirmation"):
            prepare(cmd + ["--expected_recipe_sha256", prepared["recipe_sha256"]])
        self.assertFalse(self.models.exists())

    def test_invalid_values_block_train_pipeline_and_previews(self):
        with patch.object(rl_app.messagebox, "showerror"), patch.object(rl_app.messagebox, "showwarning"), \
                patch.object(self.app, "_review_training_command") as review:
            for raw in ("", "nan", "120", "-1", "bad", "0"):
                set_entry(self.app.train_pct, raw)
                self.app._start_training()
                self.app._update_train_split_hint()
                self.assertIsNone(self.app._train_effective_rows())
                set_entry(self.app.pipe_train_pct, raw)
                self.app._run_full_pipeline()
                self.app._update_pipeline_split_hint()
            review.assert_not_called()

    def test_confirmation_cancel_and_confirm(self):
        cmd = self.capture_train()
        prepared = prepare(cmd)
        calls = []
        self.app._show_training_confirmation(cmd, prepared, calls.append)
        dialog = self.app._training_confirmation
        buttons = [c for child in dialog.winfo_children() for c in child.winfo_children()
                   if isinstance(c, rl_app.ctk.CTkButton)]
        next(b for b in buttons if b.cget("text") == "Cancel").invoke()
        self.assertEqual(calls, [])
        self.app._show_training_confirmation(cmd, prepared, calls.append)
        dialog = self.app._training_confirmation
        buttons = [c for child in dialog.winfo_children() for c in child.winfo_children()
                   if isinstance(c, rl_app.ctk.CTkButton)]
        next(b for b in buttons if b.cget("text") == "Start training").invoke()
        self.assertEqual(calls[0][-2:], ["--expected_recipe_sha256", prepared["recipe_sha256"]])

    def test_all_preset_profiles_and_full_train_summary(self):
        for reward_key, reward in rl_app.REWARD_PROFILES.items():
            self.app.train_reward_profile.set(reward["label"])
            self.app._apply_train_reward_profile_defaults()
            for action_key, action in rl_app.ACTION_PROFILES.items():
                with self.subTest(reward=reward_key, action=action_key):
                    self.app.train_action_profile.set(action["label"])
                    self.app._apply_train_action_profile_defaults()
                    recipe = prepare(self.capture_train())["recipe"]
                    self.assertEqual(recipe["settings"]["reward_profile"], reward_key)
                    self.assertEqual(recipe["settings"]["action_profile"], action_key)
        set_entry(self.app.train_pct, "100%")
        self.app.train_reward.set("mtm")
        prepared = prepare(self.capture_train())
        self.assertIsNone(prepared["recipe"]["test_period"])
        self.assertEqual(prepared["recipe"]["test_rows"], 0)
        self.assertIn("no unseen Test", prepared["recipe"]["evaluation_role"])

    def test_invalid_inputs_block_other_pages_without_fallback(self):
        app = self.app
        app.bt_model.set("probe")
        app.bt_csv.set(str(self.csv))
        app.ft_base.set("base")
        app.ft_old.set(str(self.csv))
        app.ft_new.set(str(self.csv))
        app.wf_csv.set(str(self.csv))
        app.tool_feat_csv.set(str(self.csv))
        app.regime_csv_path = str(self.csv)
        app.regime_method.set("hmm")
        cases = [(app.bt_start, "100%", app._run_backtest),
                 (app.bt_window, "", app._run_backtest),
                 (app.bt_conf, "nan", app._run_backtest),
                 (app.bt_swap_long, "bad", app._run_backtest),
                 (app.ft_mix, "100%", app._run_finetune),
                 (app.ft_lr, "inf", app._run_finetune),
                 (app.wf_lr, "", app._run_walkforward),
                 (app.wf_nsteps, "3.5", app._run_walkforward),
                 (app.tool_feat_threshold, "nan", app._show_correlation),
                 (app.tool_feat_train_pct, "101%", app._clean_features),
                 (app.regime_hmm_states, "2.5", app._run_regime_detection)]
        with patch.object(rl_app.messagebox, "showerror") as error, \
                patch.object(rl_app.messagebox, "showwarning") as warning, \
                patch.object(app, "_start_runner") as runner, \
                patch.object(rl_app.threading, "Thread") as worker:
            for entry, raw, action in cases:
                previous = entry.get()
                set_entry(entry, raw)
                count = error.call_count + warning.call_count
                action()
                self.assertGreater(error.call_count + warning.call_count, count, action.__name__)
                set_entry(entry, previous)
            runner.assert_not_called()
            worker.assert_not_called()
        set_entry(app.train_lr, "")
        app._wf_copy_from_train()
        self.assertEqual(app.wf_lr.get(), "")

    def test_walkforward_copies_resolved_json_values_into_arguments(self):
        self.app.train_reward_json_overrides = {"trade_penalty": .012}
        self.app._apply_train_reward_overrides_to_controls({"trade_penalty": .012})
        self.app.train_action_profile.set(rl_app.ACTION_PROFILES["manage_6"]["label"])
        self.app._apply_train_action_profile_defaults()
        self.app.train_action_json_params = {"trail_atr_mult": 3.5}
        self.app._apply_train_action_params_to_controls({"trail_atr_mult": 3.5})
        self.app._wf_copy_from_train()
        self.app.wf_csv.set(str(self.csv))
        with patch.object(self.app, "_start_runner") as runner:
            self.app._run_walkforward()
        cmd = runner.call_args.args[0]
        self.assertEqual(json.loads(cmd[cmd.index("--reward_overrides")+1])["trade_penalty"], .012)
        self.assertEqual(json.loads(cmd[cmd.index("--action_params")+1])["trail_atr_mult"], 3.5)
        self.assertEqual(cmd[cmd.index("--action_profile")+1], "manage_6")
        train = self.capture_train()
        for flag in ("--window", "--max_hold", "--learning_rate", "--clip_range", "--ent_coef",
                     "--n_steps", "--n_epochs", "--batch_size", "--gamma", "--gae_lambda", "--vf_coef"):
            self.assertEqual(float(cmd[cmd.index(flag)+1]), float(train[train.index(flag)+1]))

    def test_custom_action_list_survives_walkforward_copy_and_cli(self):
        action = self.root / "five_actions.json"
        action.write_text(json.dumps({"schema": "metafxclub.action_profile.v1", "base_profile": "manage_6",
                                     "actions": ["hold", "buy", "sell", "close", "move_sl_breakeven"]}))
        with patch.object(rl_app.filedialog, "askopenfilename", return_value=str(action)):
            self.app._load_train_action_json()
        self.assertEqual(len(prepare(self.capture_train())["recipe"]["action_config"]["actions"]), 5)
        self.app._wf_copy_from_train()
        self.app.wf_csv.set(str(self.csv))
        set_entry(self.app.wf_steps, 32)
        set_entry(self.app.wf_windows, 2)
        set_entry(self.app.wf_name, "wf_contract")
        with patch.object(self.app, "_start_runner") as runner:
            self.app._run_walkforward()
        cmd = runner.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--action_profile_json")+1], str(action))
        done = subprocess.run([sys.executable, str(ROOT / cmd[1]), *cmd[2:]], cwd=self.root,
                              env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(done.returncode, 0, (done.stdout+done.stderr)[-4000:])
        meta = json.loads((self.root / "wf_contract_preprocessing.json").read_text())
        self.assertEqual(meta["recipe"]["action_profile"], "custom")
        self.assertEqual(meta["recipe"]["action_profile_json"], str(action))
        for flag in ("window", "max_hold", "learning_rate", "n_steps", "batch_size"):
            self.assertEqual(float(meta["recipe"][flag]), float(cmd[cmd.index("--"+flag)+1]))
        self.app._wf_recipe_changed("action")
        self.assertEqual(self.app.wf_action_profile_json_value, "")

    def test_reward_action_entries_do_not_clamp_invalid_values(self):
        reward = self.app.train_reward_controls["trade_penalty"]["entry"]
        action = self.app.train_action_controls["trail_atr_period"]["entry"]
        original = reward.get()
        with patch.object(rl_app.messagebox, "showerror"):
            for raw in ("nan", "999999", ""):
                set_entry(reward, raw)
                self.assertFalse(self.app._on_reward_entry_change("trade_penalty"))
                self.assertEqual(reward.get(), raw)
            for raw in ("2.5", "999999", "inf"):
                set_entry(action, raw)
                self.assertFalse(self.app._on_action_entry_change("trail_atr_period"))
                self.assertEqual(action.get(), raw)
                self.assertIsNone(self.app._collect_train_action_params())
        set_entry(reward, original)
        self.assertTrue(self.app._on_reward_entry_change("trade_penalty"))
        set_entry(action, "14")
        self.assertTrue(self.app._on_action_entry_change("trail_atr_period"))

    def test_live_visual_flag_reaches_cli_without_changing_the_recipe(self):
        self.app.train_viz_var.set(False)
        plain = self.capture_train()
        self.app.train_viz_var.set(True)
        try:
            viz = self.capture_train()
        finally:
            self.app.train_viz_var.set(False)
        self.assertNotIn("--viz", plain)
        self.assertEqual(viz[-1], "--viz")
        self.assertEqual(viz[:-1], plain)
        # snapshots are read-only: the confirmed recipe fingerprint must not change
        self.assertEqual(prepare(plain)["recipe_sha256"], prepare(viz)["recipe_sha256"])

    def test_corr_threshold_field_reaches_recipe_and_walkforward(self):
        default = self.capture_train()
        self.assertEqual(float(default[default.index("--corr_threshold") + 1]), 1.0)
        set_entry(self.app.train_corr, "0.95")
        try:
            cmd = self.capture_train()
            self.assertEqual(cmd[cmd.index("--corr_threshold") + 1], "0.95")
            self.assertEqual(prepare(cmd)["recipe"]["settings"]["corr_threshold"], 0.95)
            self.app._wf_copy_from_train()
            self.app.wf_csv.set(str(self.csv))
            with patch.object(self.app, "_start_runner") as runner:
                self.app._run_walkforward()
            wf = runner.call_args.args[0]
            self.assertEqual(wf[wf.index("--corr_threshold") + 1], "0.95")
            set_entry(self.app.wf_corr, "")
            with patch.object(self.app, "_start_runner") as runner:
                self.app._run_walkforward()
            self.assertNotIn("--corr_threshold", runner.call_args.args[0])
        finally:
            set_entry(self.app.train_corr, "1.0")

    def test_corr_threshold_rejects_out_of_range(self):
        with patch.object(rl_app.messagebox, "showerror") as err, \
                patch.object(self.app, "_review_training_command") as review:
            try:
                for raw in ("", "0", "-0.1", "1.5", "nan", "bad"):
                    set_entry(self.app.train_corr, raw)
                    self.app._start_training()
            finally:
                set_entry(self.app.train_corr, "1.0")
            review.assert_not_called()
            self.assertEqual(err.call_count, 6)

    def test_viz_snapshot_lines_open_one_window_per_run(self):
        script = self.root / "tools" / "viz" / "live_viewer.py"
        script.parent.mkdir(parents=True)
        script.write_text("")
        viz_dir = self.root / "models" / "m" / "runs" / ("a" * 32) / "viz"
        viz_dir.mkdir(parents=True)
        with patch.object(rl_app.subprocess, "Popen") as popen:
            popen.return_value.poll.return_value = None
            self.app._viz_run_dir = None
            self.app._handle_log_line(f"VIZ_SNAPSHOT 0 10000 {viz_dir / 'snap_000.json'}", page="train")
            self.app._handle_log_line(f"  12%|#   | VIZ_SNAPSHOT 1 20000 {viz_dir / 'snap_001.json'}", page="train")
            self.assertEqual(popen.call_count, 1)
            args = popen.call_args.args[0]
            self.assertEqual(args[1:], [str(script), "--run_dir", str(viz_dir), "--mode", "monitor"])
            self.assertIn("eval 2 @ 20,000", self.app.train_viz_status.cget("text"))
            self.app.train_viz_mode.set("Cinematic")
            try:
                self.app._open_live_visual()          # reopen the current run in Cinematic
            finally:
                self.app.train_viz_mode.set("Monitor (verdicts)")
            self.assertEqual(popen.call_args.args[0][-2:], ["--mode", "cinematic"])
            self.app.train_viz_mode.set("Dashboard (lowest CPU)")
            try:
                self.app._open_live_visual()
            finally:
                self.app.train_viz_mode.set("Monitor (verdicts)")
            self.assertEqual(popen.call_args.args[0][-2:], ["--mode", "dashboard"])
            self.app.train_viz_mode.set("Monitor (verdicts)")
            try:
                self.app._open_live_visual()
            finally:
                self.app.train_viz_mode.set("Monitor (verdicts)")
            self.assertEqual(popen.call_args.args[0][-2:], ["--mode", "monitor"])
            self.app._close_viz_window()
            self.assertGreaterEqual(popen.return_value.terminate.call_count, 1)
        self.assertFalse(self.app._viz_window_running())

    def test_viz_rollout_line_opens_the_window_before_the_first_eval(self):
        script = self.root / "tools" / "viz" / "live_viewer.py"
        script.parent.mkdir(parents=True)
        script.write_text("")
        viz_dir = self.root / "models" / "m" / "runs" / ("b" * 32) / "viz"
        viz_dir.mkdir(parents=True)
        with patch.object(rl_app.subprocess, "Popen") as popen:
            popen.return_value.poll.return_value = None
            self.app._viz_run_dir = None
            self.app._handle_log_line(f"VIZ_ROLLOUT 0 2048 {viz_dir / 'rollout_latest.json'}", page="train")
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(popen.call_args.args[0][1:4], [str(script), "--run_dir", str(viz_dir)])
            self.assertIn("training rollout 1 @ 2,048", self.app.train_viz_status.cget("text"))
            self.app._handle_log_line(f"VIZ_SNAPSHOT 0 10000 {viz_dir / 'snap_000.json'}", page="train")
            self.assertEqual(popen.call_count, 1)                 # same run: same window
            self.assertIn("eval 1 @ 10,000", self.app.train_viz_status.cget("text"))
            self.app._close_viz_window()


if __name__ == "__main__":
    unittest.main()
