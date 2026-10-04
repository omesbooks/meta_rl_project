import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import artifact_paths as ap
from action_profiles import get_action_profile
from trading_env import TradingEnv, SegmentedTradingEnv
from training_data import (validate_frame, fit_preprocessing, apply_normalization,
                           split_training_data, check_selection_scope, walkforward_source)


def data(n=120):
    x = np.arange(n, dtype=float)
    p = 100 + np.sin(x / 5)
    return pd.DataFrame(dict(timestamp=pd.date_range("2024-01-01", periods=n, freq="4h"),
                             open=p, high=p+1, low=p-1, close=p,
                             f=np.sin(x/7), g=np.cos(x/9), constant=np.ones(n)))


class DataTests(unittest.TestCase):
    def test_test_data_does_not_change_preprocessing(self):
        original, changed = data(), data()
        changed.loc[96:, ["f", "g"]] = 1e8
        a, validation, test, _ = split_training_data(original, .8, 2)
        b, _, _, _ = split_training_data(changed, .8, 2)
        fa, na, ia = fit_preprocessing(a, ["f", "g", "constant"], .9)
        fb, nb, ib = fit_preprocessing(b, ["f", "g", "constant"], .9)
        self.assertEqual((fa, ia), (fb, ib))
        pd.testing.assert_frame_equal(na, nb)
        self.assertNotIn("constant", fa)
        self.assertLess(a.timestamp.max(), validation.timestamp.min())
        self.assertLess(validation.timestamp.max(), test.timestamp.min())

    def test_full_train_has_no_fake_test(self):
        _, validation, test, _ = split_training_data(data(), 1, 2)
        self.assertIsNone(validation)
        self.assertIsNone(test)

    def test_overlapping_eval_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly later"):
            split_training_data(data(), .8, 2, data().tail(30))

    def test_bad_prices_features_timestamps_rejected(self):
        for col, value in (("timestamp", "bad"), ("close", 0), ("f", np.inf), ("f", np.nan)):
            with self.subTest(col=col, value=value):
                frame = data()
                if col == "timestamp":
                    frame[col] = frame[col].astype(str)
                frame.loc[5, col] = value
                with self.assertRaises(ValueError):
                    validate_frame(frame)
        frame = data()
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_frame(frame)

    def test_missing_feature_is_not_filled(self):
        features, norm, _ = fit_preprocessing(data(), ["f", "g"])
        with self.assertRaisesRegex(ValueError, "missing features"):
            apply_normalization(data().drop(columns="g"), features, norm)

    def test_selection_scope_and_wf_raw_source(self):
        with tempfile.TemporaryDirectory() as td:
            original, clean = Path(td)/"raw.csv", Path(td)/"raw_clean.csv"
            data().to_csv(original, index=False)
            data().drop(columns="g").to_csv(clean, index=False)
            with self.assertRaisesRegex(ValueError, "provenance"):
                check_selection_scope(clean, data().head(60))
            clean.with_suffix(".features.json").write_text(json.dumps({
                "source_csv": str(original), "threshold": .9,
                "fit_end": data().timestamp.iloc[80].isoformat()}))
            with self.assertRaisesRegex(ValueError, "later"):
                check_selection_scope(clean, data().head(60))
            raw, threshold = walkforward_source(clean)
            self.assertIn("g", raw)
            self.assertEqual(threshold, .9)


class RewardTests(unittest.TestCase):
    def test_known_return_and_drawdown(self):
        frame = data()
        frame.loc[:, "close"] = 100.
        frame.loc[2, "close"] = 110.
        frame.loc[3, "close"] = 90.
        env = TradingEnv(frame, ["f"], window_size=2, max_steps=3,
                         random_start=False, spread_pct=0, commission=0)
        env.reset()
        env.step(1)
        env.step(0)
        env.step(0)
        stats = env.get_stats()
        self.assertAlmostEqual(stats["return"], -.1)
        self.assertAlmostEqual(stats["max_dd"], .9 / 1.1 - 1)
        self.assertAlmostEqual(env._curve[-1], env.equity)

    def test_liquidation_and_close_match(self):
        for mode, formula in (("realized", ""), ("mtm", ""), ("realized", "trade_closed_pnl * 50")):
            results = []
            for action in (0, 3):
                env = TradingEnv(data(), ["f"], window_size=2, max_steps=2,
                                 random_start=False, reward_mode=mode, reward_formula=formula)
                env.reset(seed=7)
                env.step(1)
                _, reward, term, trunc, _ = env.step(action)
                results.append((reward, env.equity))
                self.assertTrue(term)
                self.assertFalse(trunc)
                self.assertEqual(env._curve[-1], env.equity)
                self.assertEqual(len(env.trades), 1)
            np.testing.assert_allclose(results[0], results[1])

    def test_flat_close_cannot_evade_penalty(self):
        rewards = []
        for action in (0, 3):
            env = TradingEnv(data(), ["f"], window_size=2, max_steps=8)
            env.reset(seed=7)
            rewards.append(env.step(action)[1])
        self.assertEqual(rewards[0], rewards[1])
        self.assertLess(rewards[0], 0)

    def test_stop_or_max_hold_closes_only_once_at_boundary(self):
        for reason in ("managed_sl", "max_hold"):
            env = TradingEnv(data(), ["f"], window_size=2, max_steps=2,
                             max_hold_bars=1 if reason == "max_hold" else 30,
                             action_profile="manage_6")
            env.reset(seed=7)
            env.step(1)
            if reason == "managed_sl":
                env.stop_price = float(env._closes[env.t]) + .01
            env.step(0)
            self.assertEqual(len(env.trades), 1)
            self.assertEqual(env.trades[0]["reason"], reason)

    def test_segment_episodes_never_cross_sources(self):
        old, new = data(40), data(45)
        new.timestamp += pd.Timedelta(days=100)
        env = SegmentedTradingEnv([old, new], ["f"], window_size=2, max_steps=200)
        for seed in range(6):
            env.reset(seed=seed)
            current = env.envs[env.active]
            start, end = current.df.timestamp.iloc[[0, -1]]
            done = False
            while not done:
                self.assertTrue(start <= current.df.timestamp.iloc[current.t] <= end)
                _, _, term, trunc, _ = env.step(1)
                done = term or trunc
            self.assertEqual(current.position, 0)
        self.assertTrue(all(env.segment_steps))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PPO("MlpPolicy", TradingEnv(data(), ["f"], window_size=2),
                        n_steps=8, batch_size=4, policy_kwargs={"net_arch": [8]}, device="cpu")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(ap, "MODELS_DIR", Path(tmp.name))
        p.start()
        self.addCleanup(p.stop)
        ap._READ_OUTPUTS.clear()
        self.addCleanup(ap._READ_OUTPUTS.clear)

    def write(self, run, params=False):
        self.model.save(run.path/"demo.zip")
        pd.DataFrame({"mean": [0.], "std": [1.]}, index=["f"]).to_csv(run.path/"demo_norm.csv")
        (run.path/"demo.train.json").write_text(json.dumps({"features": ["f"], "hyperparameters": {
            "algo": "ppo", "window": 2, "action_profile_config": get_action_profile("basic_4")[1]}}))
        if params:
            (run.path/"demo.params.json").write_text("{}")

    def test_abort_keeps_previous_artifacts(self):
        home = ap.ensure_model_dirs("demo")
        (home/"demo.zip").write_bytes(b"old")
        (home/"demo_norm.csv").write_bytes(b"old norm")
        with self.assertRaises(KeyboardInterrupt):
            with ap.ArtifactRun("demo") as run:
                self.write(run)
                raise KeyboardInterrupt()
        self.assertEqual(ap.final_model_path("demo").read_bytes(), b"old")
        self.assertEqual(ap.norm_path("demo").read_bytes(), b"old norm")
        self.assertEqual(json.loads((run.path/"run.json").read_text())["status"], "cancelled")

    def test_no_stale_params_or_reports(self):
        with ap.ArtifactRun("demo") as first:
            self.write(first, True)
            (first.path/"backtests"/"demo_live_bt_trades.csv").write_text("old result")
            first.publish(self.model)
        with ap.ArtifactRun("demo") as second:
            self.write(second)
            second.publish(self.model)
        self.assertEqual(ap.model_dir("demo"), second.path)
        self.assertIsNone(ap.find_params_path("demo"))
        self.assertIsNone(ap.find_trades_path("demo"))
        self.assertIn("demo", ap.model_names_from_artifacts())

    def test_pointer_swap_failure_preserves_old_generation(self):
        with ap.ArtifactRun("demo") as first:
            self.write(first)
            first.publish(self.model)
        replace = ap.os.replace
        def fail(src, dst):
            if Path(dst).name == "current.json":
                raise PermissionError("simulated publication failure")
            return replace(src, dst)
        with self.assertRaises(PermissionError):
            with ap.ArtifactRun("demo") as second:
                self.write(second)
                with patch.object(ap.os, "replace", side_effect=fail):
                    second.publish(self.model)
        self.assertEqual(ap.model_dir("demo"), first.path)

    def test_wrong_norm_is_not_published(self):
        with self.assertRaises(ValueError):
            with ap.ArtifactRun("demo") as run:
                self.write(run)
                pd.DataFrame({"mean": [0.], "std": [1.]}, index=["wrong"]).to_csv(run.path/"demo_norm.csv")
                run.publish(self.model)
        self.assertFalse((run.home/"current.json").exists())

    def test_pinned_reader_uses_same_generation(self):
        with ap.ArtifactRun("demo") as first:
            self.write(first)
            first.publish(self.model)
        ap.pin_model_generation("demo")
        with ap.ArtifactRun("demo") as second:
            self.write(second)
            second.publish(self.model)
        self.assertEqual(ap.model_dir("demo"), first.path)

    def test_finetune_learning_rate_schedule(self):
        model = self.model
        model.learning_rate = 1e-4
        model._setup_lr_schedule()
        model.set_logger(configure(format_strings=[]))
        for progress in (1., .5):
            model._current_progress_remaining = progress
            model._update_learning_rate(model.policy.optimizer)
            self.assertEqual(model.policy.optimizer.param_groups[0]["lr"], 1e-4)

    def test_process_cancel_and_restart_recovery(self):
        with ap.ArtifactRun("demo") as run:
            ap.mark_process_runs(run.state["pid"], "cancelled")
            self.assertEqual(json.loads((run.path/"run.json").read_text())["status"], "cancelled")
        run.state.update(status="running", pid=2147483647)
        ap._atomic_json(run.path/"run.json", run.state)
        self.assertEqual(ap.recover_interrupted_runs(), [str(run.path/"run.json")])
        self.assertEqual(json.loads((run.path/"run.json").read_text())["status"], "failed")

    def test_corrupt_weights_are_not_published(self):
        with self.assertRaises(Exception):
            with ap.ArtifactRun("demo") as run:
                self.write(run)
                (run.path/"demo.zip").write_bytes(b"broken")
                run.publish(self.model)
        self.assertFalse((run.home/"current.json").exists())


class GuiDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rl_app
        cls.module = rl_app
        cls.app = rl_app.RLTradingStudio()
        cls.app.withdraw()
        for page in ("tools", "train", "pipeline", "backtest", "walkfwd", "finetune", "models"):
            cls.app.show_page(page)

    @classmethod
    def tearDownClass(cls):
        for callback in cls.app.tk.call("after", "info"):
            cls.app.tk.call("after", "cancel", callback)
        cls.app.destroy()

    def test_clean_uses_train_slice_and_writes_provenance(self):
        frame = data()
        frame["h"] = np.arange(len(frame))
        frame["i"] = np.sin(np.arange(len(frame))/3)
        frame["j"] = np.cos(np.arange(len(frame))/4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame.to_csv(root/"raw.csv", index=False)
            with patch.object(self.module, "WORK_DIR", root), \
                    patch.object(self.app, "_log"), patch.object(self.app, "_tools_file_notice"):
                self.app._clean_features_worker("raw.csv", 1., .5)
            provenance = json.loads((root/"raw_clean.features.json").read_text())
            self.assertEqual(provenance["fit_rows"], 60)
            self.assertEqual(provenance["fit_end"], frame.timestamp.iloc[59].isoformat())
            clean = pd.read_csv(root/"raw_clean.csv")
            self.assertEqual(len(clean), len(frame))
            self.assertNotIn("constant", clean)

    def test_gui_shows_three_splits_and_ppo_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"raw.csv"
            data().to_csv(path, index=False)
            self.app.train_csv_path = str(path)
            self.app.train_pct.delete(0, "end")
            self.app.train_pct.insert(0, "80")
            self.app._update_train_split_hint()
            text = self.app.train_split_hint.cget("text")
            self.assertIn("Validation", text)
            self.assertIn("Test", text)
            self.assertEqual(self.app.train_algo.cget("values"), ["PPO (recommended)"])

    def test_gui_rejects_invalid_ppo_settings(self):
        with patch.object(self.module.messagebox, "showerror"):
            self.assertFalse(self.app._validate_ppo_fields({"learning_rate": "nan"}))
            self.assertFalse(self.app._validate_ppo_fields({"n_steps": 1, "batch_size": 1}))
            self.assertFalse(self.app._validate_ppo_fields({"n_steps": 8, "batch_size": 16}))
            self.assertTrue(self.app._validate_ppo_fields({"n_steps": 16, "batch_size": 8}))


if __name__ == "__main__":
    unittest.main()
