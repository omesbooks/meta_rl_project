import unittest
import threading
import json
import tempfile
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import rl_app
import backtest_live
import export_to_onnx
import gemini_labeler
import regime_compare
from trading_env import TradingEnv


class Batch1GuiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = rl_app.RLTradingStudio()
        for page in ("tools", "train", "backtest", "walkfwd", "finetune"):
            cls.app.show_page(page)
        cls.app._is_process_busy = lambda: False

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def test_collector_and_export_autofill_follow_source(self):
        app = self.app
        app.tool_collector_out.delete(0, "end")
        app._collector_out_auto = None
        app._on_collector_src_change("eurusd_h4.csv")
        self.assertEqual(app.tool_collector_out.get(), "training_data_eurusd_h4.csv")
        app._on_collector_src_change("gbpusd_h4.csv")
        self.assertEqual(app.tool_collector_out.get(), "training_data_gbpusd_h4.csv")
        app.tool_collector_out.delete(0, "end")
        app.tool_collector_out.insert(0, "my_custom.csv")
        app._on_collector_src_change("xauusd_h4.csv")
        self.assertEqual(app.tool_collector_out.get(), "my_custom.csv")

        app.tool_export_name.delete(0, "end")
        app._export_name_autofilled = None
        app._on_export_model_change("model_a")
        app._on_export_model_change("model_b")
        self.assertEqual(app.tool_export_name.get(), "model_b")
        app.tool_export_name.delete(0, "end")
        app.tool_export_name.insert(0, "custom_deploy")
        app._export_name_autofilled = None
        app._on_export_model_change("model_c")
        self.assertEqual(app.tool_export_name.get(), "custom_deploy")

    def test_split_sidecar_forwarding_helper(self):
        src = Path("source.csv")
        src_params = Path("source.params.json")
        for name in ("source_train.csv", "source_test.csv"):
            dst = Path(name)
            with patch.object(
                    self.app, "_find_csv_params_sidecar", return_value=src_params), \
                    patch("shutil.copy") as copy_file:
                copied, error = self.app._copy_csv_params_sidecar(src, dst)
            self.assertIsNone(error)
            self.assertEqual(copied, dst.with_suffix(".params.json"))
            copy_file.assert_called_once_with(src_params, copied)

    def test_formula_json_is_omitted_when_developer_mode_is_off(self):
        app = self.app
        app.train_csv_path = "uj_h4_dataset.csv"
        app.train_name.delete(0, "end")
        app.train_name.insert(0, "codex_batch1_formula_probe")
        app.train_reward_profile_json_path = "developer_formula_example.json"
        app.train_reward_profile_json_has_formula = True
        app.reward_formula_enabled.set(False)
        captured = {}
        app._start_runner = lambda cmd, page=None: captured.setdefault("cmd", cmd) or True
        app._start_training()
        cmd = captured["cmd"]
        self.assertNotIn("--reward_profile_json", cmd)
        self.assertNotIn("--reward_formula", cmd)

    def test_backtest_uses_model_max_hold_and_resets_stale_m1(self):
        app = self.app
        app.bt_m1_csv.set("(none)")
        app._bt_m1_auto_value = None
        app.bt_csv.set("uj_h4_extra_features.csv")
        app._auto_select_bt_m1()
        self.assertTrue(app.bt_m1_csv.get().endswith("uj_h4_extra_features_m1.csv"))
        app.bt_csv.set("data_au_h4_dataset_clean.csv")
        app._auto_select_bt_m1()
        self.assertEqual(app.bt_m1_csv.get(), "(none)")

        app.bt_model.set("UJ_BE_Test")
        app.bt_csv.set("uj_h4_dataset.csv")
        captured = {}
        app._start_runner = lambda cmd, page=None: captured.setdefault("cmd", cmd) or True
        app._run_backtest()
        cmd = captured["cmd"]
        idx = cmd.index("--max_hold")
        self.assertEqual(cmd[idx + 1], "40")

    def test_walkforward_name_and_four_level_verdict(self):
        app = self.app
        app.wf_csv.set("uj_h4_dataset.csv")
        app.wf_name.delete(0, "end")
        app.wf_name.insert(0, "wf:v2")
        captured = {}
        app._start_runner = lambda cmd, page=None: captured.setdefault("cmd", cmd) or True
        app._run_walkforward()
        self.assertEqual(app.wf_name.get(), "wf_v2")
        self.assertIn("wf_v2", captured["cmd"])

        app._handle_log_line(
            "🟡 MOSTLY ROBUST — 4/5 windows มี PF>1 → ใช้ระวัง", page="walkfwd")
        self.assertEqual(app.wf_verdict_text.cget("text"), "MOSTLY ROBUST")
        self.assertEqual(app.wf_verdict_icon.cget("text"), "🟡")

    def test_finetune_name_collision_is_blocked(self):
        app = self.app
        app.ft_base.set("UJ_BE_Test")
        app.ft_old.set("uj_h4_dataset.csv")
        app.ft_new.set("uj_h4_dataset.csv")
        app.ft_name.delete(0, "end")
        app.ft_name.insert(0, "UJ_BE_Test")
        called = []
        app._start_runner = lambda *args, **kwargs: called.append(args)
        with patch.object(rl_app.messagebox, "showwarning") as warning:
            app._run_finetune()
        warning.assert_called_once()
        self.assertFalse(called)

    def test_finetune_percent_mix_is_normalized(self):
        app = self.app
        app.ft_base.set("UJ_BE_Test")
        app.ft_old.set("uj_h4_dataset.csv")
        app.ft_new.set("uj_h4_dataset.csv")
        app.ft_name.delete(0, "end")
        app.ft_name.insert(0, "UJ_BE_Test_pct")
        app.ft_mix.delete(0, "end")
        app.ft_mix.insert(0, "30")
        captured = {}
        app._start_runner = lambda cmd, page=None: captured.setdefault("cmd", cmd) or True
        app._run_finetune()
        cmd = captured["cmd"]
        self.assertEqual(cmd[cmd.index("--mix_ratio") + 1], "0.3")

    def test_invalid_reward_entry_reverts_only_that_field(self):
        app = self.app
        keys = list(app.train_reward_controls)
        target, untouched = keys[0], keys[1]
        untouched_entry = app.train_reward_controls[untouched]["entry"]
        untouched_entry.delete(0, "end")
        untouched_entry.insert(0, "0.123")
        target_entry = app.train_reward_controls[target]["entry"]
        target_entry.delete(0, "end")
        target_entry.insert(0, "bad")
        with patch.object(rl_app.messagebox, "showerror"):
            self.assertFalse(app._on_reward_entry_change(target))
        self.assertEqual(untouched_entry.get(), "0.123")

    def test_integer_parser_accepts_display_formats_before_run(self):
        entry = self.app.train_steps
        cases = {
            "300,000": 300000,
            "2e5": 200000,
            "": 123,
            "50000": 50000,
        }
        for raw, expected in cases.items():
            entry.delete(0, "end")
            entry.insert(0, raw)
            self.assertEqual(
                self.app._parse_int_field(entry, 123, "Steps"), expected)
        entry.delete(0, "end")
        entry.insert(0, "3.5")
        with patch.object(rl_app.messagebox, "showwarning"):
            self.assertIsNone(self.app._parse_int_field(entry, 123, "Steps"))

    def test_pipeline_stop_uses_local_process_snapshot(self):
        app = self.app

        class RacingProc:
            def __init__(self):
                self.terminated = False

            def poll(self):
                app.pipeline_proc = None
                return None

            def terminate(self):
                self.terminated = True

        proc = RacingProc()
        app.pipeline_proc = proc
        app._pipeline_log = lambda *_args, **_kwargs: None
        app._stop_pipeline()
        self.assertTrue(proc.terminated)

    def test_close_stops_children_before_destroy(self):
        app = self.app

        class FakeRunner:
            def __init__(self):
                self.stopped = False

            def is_running(self):
                return True

            def stop(self):
                self.stopped = True

        class FakeProc:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None if not self.terminated else 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

        original_runner = app.runner
        runner = FakeRunner()
        proc = FakeProc()
        app.runner = runner
        app.pipeline_running = True
        app.pipeline_proc = proc
        try:
            with patch.object(rl_app.messagebox, "askyesno", return_value=True), \
                    patch.object(app, "destroy") as destroy:
                app._on_close()
            self.assertTrue(runner.stopped)
            self.assertTrue(proc.terminated)
            destroy.assert_called_once()
        finally:
            app.runner = original_runner
            app.pipeline_running = False
            app.pipeline_proc = None


class TradingEnvClampTests(unittest.TestCase):
    def test_oversized_episode_is_clamped_to_dataframe(self):
        rows = 40
        df = pd.DataFrame({
            "close": np.linspace(1.0, 1.1, rows),
            "high": np.linspace(1.01, 1.11, rows),
            "low": np.linspace(0.99, 1.09, rows),
            "feature": np.linspace(-1.0, 1.0, rows),
        })
        env = TradingEnv(df, ["feature"], window_size=10, max_steps=2000)
        self.assertEqual(env.max_steps, 28)
        _, _ = env.reset(seed=7)
        done = False
        count = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            count += 1
            self.assertLess(count, rows)


class ProcessRunnerRaceTests(unittest.TestCase):
    def test_start_guard_and_stop_during_spawn(self):
        gate = threading.Event()

        class FakeStdout:
            def readline(self):
                return ""

        class FakeProc:
            def __init__(self):
                self.stdout = FakeStdout()
                self.returncode = 0
                self.terminated = False

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.terminated = True

            def wait(self, timeout=None):
                return self.returncode

        fake = FakeProc()

        def delayed_popen(*_args, **_kwargs):
            gate.wait(2)
            return fake

        runner = rl_app.ProcessRunner()
        with patch.object(rl_app.subprocess, "Popen", side_effect=delayed_popen):
            self.assertTrue(runner.start(["fake-command"]))
            self.assertTrue(runner.is_running())
            self.assertFalse(runner.start(["second-command"]))
            runner.stop()
            gate.set()
            runner.thread.join(timeout=3)
        self.assertFalse(runner.thread.is_alive())
        self.assertTrue(fake.terminated)
        self.assertFalse(runner.is_running())


class BatchThreeFourPureTests(unittest.TestCase):
    def test_failed_meta_preserves_complete_result(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model_live_bt.meta.json"
            path.write_text(json.dumps({"status": "complete", "result": {"total_trades": 5}}))
            target = backtest_live._write_failed_meta(path, {"status": "failed"})
            self.assertEqual(json.loads(path.read_text())["status"], "complete")
            self.assertNotEqual(target, path)
            self.assertEqual(json.loads(target.read_text())["status"], "failed")

    def test_deploy_name_sanitizer_produces_mql_identifier(self):
        self.assertEqual(export_to_onnx.sanitize_deploy_name("2024 uj-v1.2"),
                         "m_2024_uj_v1_2")
        self.assertEqual(export_to_onnx.sanitize_deploy_name("à¹‚à¸¡à¹€à¸”à¸¥"), "")

    def test_duplicate_event_names_survive_reader(self):
        payload = {"events": [
            {"event": "COVID", "date": "2020-03-12"},
            {"event": "COVID", "date": "2020-04-20"},
        ]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            events = gemini_labeler.load_known_events(path)
        self.assertEqual(len(events), 2)

    def test_regime_html_inlines_payload(self):
        daily = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "close": [1.0, 1.1],
        })
        result = {
            "daily": daily, "breaks": ["2020-01-02"], "method": "pelt",
            "score": 0, "n_breaks": 1, "time_sec": 0.01, "n_bars": 2,
            "csv": "source.csv", "info": {},
        }
        with tempfile.TemporaryDirectory() as td:
            html_path = Path(td) / "chart.html"
            regime_compare.write_single_chart(result, html_path)
            html = html_path.read_text(encoding="utf-8")
        self.assertIn("const d = {", html)
        self.assertNotIn("fetch('regime_single_data.json')", html)

    def test_ruptures_breakpoint_maps_to_first_new_regime_row(self):
        class FakeAlgo:
            def fit(self, _signal):
                return self

            def predict(self, **_kwargs):
                return [2, 4]

        fake_rpt = types.SimpleNamespace(
            Pelt=lambda **_kwargs: FakeAlgo(),
            Binseg=lambda **_kwargs: FakeAlgo(),
        )
        daily = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=4),
            "close": [1.0, 1.0, 2.0, 2.0],
        })
        with patch.dict(sys.modules, {"ruptures": fake_rpt}):
            self.assertEqual(regime_compare.method_pelt(daily)[0], daily.date.iloc[2])
            self.assertEqual(regime_compare.method_binseg(daily)[0], daily.date.iloc[2])


if __name__ == "__main__":
    unittest.main()
