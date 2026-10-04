"""Isolated integration checks; no user model or dataset is overwritten."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = {"checks": [], "mt5_verified": False, "full_length_training": False}
    with tempfile.TemporaryDirectory(prefix="rl_flow_") as td:
        work = Path(td)
        x = np.arange(600, dtype=float)
        close = 150 + .8 * np.sin(x / 9) + .003 * x
        df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=len(x), freq="4h"),
                           "open": close, "high": close + .2, "low": close - .2,
                           "close": close, "rsi_mean": 50 + 25 * np.sin(x / 9),
                           "ret_1": pd.Series(close).pct_change().fillna(0)})
        source = work / "source.csv"
        df["unused_constant"] = 0.
        df.to_csv(source, index=False)
        reordered = work / "reordered.csv"
        df[df.columns[::-1]].to_csv(reordered, index=False)
        source.with_suffix(".params.json").write_text("{}")
        old_csv, new_csv = work / "old.csv", work / "new.csv"
        df.iloc[:300].to_csv(old_csv, index=False)
        df.iloc[300:450].to_csv(new_csv, index=False)

        def run(label, script, *arguments, expected=0):
            wrapper = (
                "import sys,os,runpy; from pathlib import Path; "
                f"sys.path.insert(0,{str(ROOT)!r}); import artifact_paths; "
                f"artifact_paths.MODELS_DIR=Path({str(work / 'models')!r}); "
                f"os.chdir({str(work)!r}); "
                f"sys.argv={[script, *map(str, arguments)]!r}; "
                f"runpy.run_path({str(ROOT / script)!r},run_name='__main__')"
            )
            started = time.monotonic()
            env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            done = subprocess.run([sys.executable, "-c", wrapper], cwd=ROOT, env=env,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
            passed = done.returncode == expected if expected == 0 else done.returncode != 0
            entry = dict(check=label, passed=passed, returncode=done.returncode,
                         seconds=round(time.monotonic() - started, 2),
                         output=(done.stdout + done.stderr)[-12000:])
            report["checks"].append(entry)
            Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{label}: {'PASS' if passed else 'FAIL'} ({entry['seconds']}s)", flush=True)
            if not passed:
                raise RuntimeError(entry["output"])

        common = ["--window", 2, "--steps", 64, "--n_steps", 16, "--batch_size", 8,
                  "--n_epochs", 1, "--ep_len", 16, "--net_arch", "8", "--mc_eval", 0,
                  "--reward_overrides", '{"trade_penalty":0.012}']
        for profile in ("basic_4", "manage_6"):
            run("train_" + profile, "rl_train.py", source, "--name", profile,
                "--action_profile", profile, "--train_pct", .8, *common)
            pointer = json.loads((work / "models" / profile / "current.json").read_text())
            active = work / "models" / profile / "runs" / pointer["run_id"]
            meta = json.loads((active / f"{profile}.train.json").read_text())
            assert meta["status"] == "completed" and meta["test_is_unseen"]
            assert meta["validation_period"]["end"] < meta["eval_period"]["start"]
        active_before = (work / "models" / "basic_4" / "current.json").read_bytes()
        run("invalid_retrain_preserves_model", "rl_train.py", source, "--name", "basic_4",
            "--n_steps", 1, "--batch_size", 1, expected=1)
        assert (work / "models" / "basic_4" / "current.json").read_bytes() == active_before
        run("train_100_percent", "rl_train.py", source, "--name", "full", "--train_pct", 1, *common)
        full_home = work / "models" / "full"
        full_run = json.loads((full_home / "current.json").read_text())["run_id"]
        full_meta = json.loads((full_home / "runs" / full_run / "full.train.json").read_text())
        assert full_meta["test_is_unseen"] is False and full_meta["validation_rows"] == 0
        run("backtest", "backtest_live.py", "basic_4", reordered, "--mode", "pure_agent",
            "--start", .9, "--conf", 0, "--mc", 0, "--random_baseline", 0)
        run("backtest_chart", "backtest_chart.py", "basic_4", source)
        run("analyze_saved_feature_order", "rl_analyze.py", "basic_4", reordered, "--start", .9)
        base_home = work / "models" / "manage_6"
        base_hashes = {str(p.relative_to(base_home)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in base_home.rglob("*") if p.is_file()}
        run("fine_tune", "rl_finetune.py", "manage_6", "--old_csv", old_csv,
            "--new_csv", new_csv, "--steps", 32, "--ep_len", 16, "--name", "fine", "--lr", .0001)
        fine_home = work / "models" / "fine"
        fine_run = json.loads((fine_home / "current.json").read_text())["run_id"]
        fine_meta = json.loads((fine_home / "runs" / fine_run / "fine.train.json").read_text())
        assert fine_meta["hyperparameters"]["reward_profile_config"]["trade_penalty"] == .012
        assert len(fine_meta["segment_training_steps"]) == 2
        assert base_hashes == {str(p.relative_to(base_home)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in base_home.rglob("*") if p.is_file()}
        run("walk_forward", "rl_walkforward.py", source, "--windows", 2, "--steps", 32,
            "--window", 2, "--n_steps", 16, "--batch_size", 8, "--n_epochs", 1,
            "--net_arch", "8", "--ep_len", 16, "--name", "wf",
            "--reward_overrides", '{"trade_penalty":0.012}')
        wf = json.loads((work / "wf_preprocessing.json").read_text())
        assert len(wf["windows"]) == 2
        run("export_onnx", "export_to_onnx.py", "basic_4", "--output_dir", work / "export")

        sys.path.insert(0, str(ROOT))
        import artifact_paths
        import onnxruntime as ort
        import torch
        from stable_baselines3 import PPO
        from export_to_onnx import PolicyWrapper
        artifact_paths.MODELS_DIR = work / "models"
        model = PPO.load(artifact_paths.final_model_path("basic_4"), device="cpu")
        fine_model = PPO.load(artifact_paths.final_model_path("fine"), device="cpu")
        assert fine_model.lr_schedule(.5) == .0001
        assert fine_model.policy.optimizer.param_groups[0]["lr"] == .0001
        session = ort.InferenceSession(str(work / "export" / "Files" / "basic_4.onnx"), providers=["CPUExecutionProvider"])
        norm = pd.read_csv(artifact_paths.norm_path("basic_4"), index_col=0)
        from trading_env import TradingEnv
        from training_data import apply_normalization
        normalized = apply_normalization(df, list(norm.index), norm)
        env = TradingEnv(normalized, list(norm.index), window_size=2, random_start=False)
        obs, _ = env.reset()
        largest = 0.
        for i in range(50):
            batch = obs[None, :]
            with torch.no_grad():
                reference = PolicyWrapper(model.policy)(torch.from_numpy(batch)).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: batch})[0]
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
            largest = max(largest, float(np.abs(actual-reference).max()))
            obs, _, _, _, _ = env.step(int(reference.argmax()))
        report["checks"].append(dict(check="onnx_parity_50_observations", passed=True, max_abs_error=largest))
        report["passed"] = all(c["passed"] for c in report["checks"])
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Integration checks complete", flush=True)


if __name__ == "__main__":
    main()
