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


if __name__ == "__main__":
    unittest.main()
