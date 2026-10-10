"""Phase 3.1 — observation-size ablation runner.

Trains the same PPO recipe under several (window, corr_threshold) arms, then
runs the training diagnosis and an out-of-sample backtest (best checkpoint,
Test slice only) for each arm, and collects everything into one table.

Each arm = one model named <prefix>_<arm>_w<window>_t<threshold>. Arms run
sequentially; a failed step is recorded and the next arm still runs.

Usage (from project root, inside the venv):
    python tools/analysis/obs_ablation.py                       # arms A,B,C,D @ 200k steps
    python tools/analysis/obs_ablation.py --arms A,B --steps 400000
    python tools/analysis/obs_ablation.py --dry_run             # print commands only
    python tools/analysis/obs_ablation.py --steps 3000 --arms A,D --prefix smoke --tag smoke
    python tools/analysis/obs_ablation.py --arms D --seeds 0,1,2          # 3 seeds per arm
    python tools/analysis/obs_ablation.py --arms D --seeds 0,1,2 --train_start 2019-01-01 \
        --prefix reg2019 --tag regime_2019   # regime locality: train from 2019, SAME Val/Test rows

Outputs:
    docs/ablation_<tag>.json / .md      results table
    artifacts/ablation/<tag>/<model>.*.log   full stdout of each step
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[2]
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

from artifact_paths import train_meta_path, backtest_meta_path  # noqa: E402

# (window, corr_threshold) per arm. Threshold 1.0 = drop constants only.
ARMS = {
    "A": (30, 1.00),   # baseline: full feature set, window 30
    "B": (30, 0.95),   # pruned features, same window
    "C": (15, 1.00),   # full features, half window
    "D": (15, 0.95),   # both reductions
    "E": (30, 0.90),   # optional: harder prune
}

# One recipe for every arm. net_arch is pinned on purpose: "auto" would pick a
# smaller network for window < 20 and confound the comparison.
RECIPE = {
    "learning_rate": "5e-5", "clip_range": "0.1", "ent_coef": "0.02",
    "n_steps": "4096", "n_epochs": "15", "batch_size": "128",
    "gamma": "0.99", "gae_lambda": "0.97", "vf_coef": "0.7",
    "net_arch": "256,128,64", "ep_len": "2000",
    "reward_mode": "realized", "reward_profile": "balanced",
    "action_profile": "basic_4",
}


def run_step(cmd, log_path, label):
    """Run one subprocess, tee stdout to a log file, return (rc, seconds)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              env=env, cwd=WORK_DIR, text=True, encoding="utf-8")
    secs = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
    print(f"    [{label}] {status} in {secs/60:.1f} min -> {log_path.name}", flush=True)
    return proc.returncode, secs


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _frac_for_row(n, k):
    """Shortest fraction f with int(n * f) == k — the split formula used by
    training_data.split_training_data and backtest_live --start."""
    for d in range(2, 12):
        f = round((k + 0.5) / n, d)
        if int(n * f) == k:
            return f
    raise ValueError(f"no fraction reproduces row {k} of {n}")


def trim_for_train_start(csv, train_start, train_pct):
    """Drop rows before `train_start` for TRAINING while keeping exactly the
    Validation/Test rows the untrimmed split would give.

    rl_train splits at int(n * train_pct) and halves the holdout; backtest
    --start cuts at int(n * start). After dropping the first `first` rows the
    same absolute rows sit at (row - first), so both fractions are recomputed
    to land there. Returns (trimmed_csv, train_pct, oos_start, info).
    """
    import shutil
    import pandas as pd
    src = Path(csv)
    df = pd.read_csv(src)
    n = len(df)
    split = int(n * train_pct)
    test0 = split + (n - split) // 2
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    mask = (ts >= pd.Timestamp(train_start)).to_numpy()
    if not mask.any():
        raise ValueError(f"no rows on/after {train_start} in {src.name}")
    first = int(mask.argmax())
    if first == 0:
        raise ValueError(f"{train_start} is not after the first row — nothing to trim")
    if first >= split:
        raise ValueError(f"{train_start} (row {first}) is past the train split (row {split})")
    out = df.iloc[first:].reset_index(drop=True)
    m = len(out)
    new_pct = _frac_for_row(m, split - first)
    new_oos = _frac_for_row(m, test0 - first)
    dst = src.with_name(f"{src.stem}_from_{train_start}.csv")
    out.to_csv(dst, index=False)
    sidecar = src.with_suffix(".params.json")
    if sidecar.exists():
        shutil.copy(sidecar, dst.with_suffix(".params.json"))
    t = out["timestamp"]
    info = {
        "source_csv": str(src), "train_start": train_start, "rows_dropped": first,
        "rows": m, "train_rows": split - first, "train_pct": new_pct, "oos_start": new_oos,
        "train_span": f"{t.iloc[0]} -> {t.iloc[split - first - 1]}",
        "validation_span": f"{t.iloc[split - first]} -> {t.iloc[test0 - first - 1]}",
        "test_span": f"{t.iloc[test0 - first]} -> {t.iloc[-1]}",
        "untrimmed_split_rows": [split, test0],
    }
    return str(dst), new_pct, new_oos, info


def _trades_pnl(name):
    """Per-trade net pnl from the stored backtest trades CSV, or None."""
    import pandas as pd
    meta = Path(backtest_meta_path(name))
    candidates = [meta.with_name(f"{name}_live_bt_trades.csv"),
                  *sorted(meta.parent.glob("*trades*.csv"))]
    for p in candidates:
        if p.is_file():
            df = pd.read_csv(p)
            for col in ("pnl_dollars", "net_pnl", "pnl", "pnl_pct", "profit"):
                if col in df.columns:
                    return df[col].to_numpy(dtype=float)
    return None


def summarize_seeds(rows):
    """Per-arm aggregate over seeds: PF spread, pooled trades, pooled PF."""
    import numpy as np
    out = []
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm and r.get("oos_pf") is not None]
        if len(rs) < 2:
            continue
        pfs = [r["oos_pf"] for r in rs]
        los = [r["oos_pf_ci95_lo"] for r in rs if r.get("oos_pf_ci95_lo") is not None]
        rets = [r["oos_return"] for r in rs if r.get("oos_return") is not None]
        pnls = [p for p in (_trades_pnl(r["model"]) for r in rs) if p is not None]
        pooled_pf = pooled_trades = None
        if len(pnls) == len(rs):
            allp = np.concatenate(pnls)
            gp, gl = float(allp[allp > 0].sum()), float(-allp[allp < 0].sum())
            pooled_pf = (gp / gl) if gl > 0 else None
            pooled_trades = int(len(allp))
        out.append({
            "arm": arm, "n_seeds": len(rs), "seeds": [r.get("seed") for r in rs],
            "pf_mean": float(np.mean(pfs)), "pf_min": float(min(pfs)), "pf_max": float(max(pfs)),
            "pf_lo95_min": (min(los) if los else None),
            "seeds_pf_lo95_gt1": sum(1 for v in los if v > 1.0),
            "return_mean": (float(np.mean(rets)) if rets else None),
            "pooled_trades": pooled_trades, "pooled_pf": pooled_pf,
        })
    return out


def collect(name, train_secs, steps):
    meta = read_json(train_meta_path(name))
    bt = read_json(backtest_meta_path(name)).get("result", {}) or {}
    hp = meta.get("hyperparameters", {}) or {}
    diag = meta.get("train_diagnosis", {}) or {}
    boot = bt.get("bootstrap") or {}
    pf_ci = (boot.get("profit_factor") or {}).get("ci95") or [None, None]
    dd_conf = boot.get("max_dd_conf") or {}
    feature_count = meta.get("feature_count")
    window = hp.get("window")
    return {
        "model": name,
        "window": window,
        "corr_threshold": ((meta.get("preprocessing") or {}).get("threshold")),
        "features": feature_count,
        "obs_dim": (window * feature_count + 3) if (window and feature_count) else None,
        "train_minutes": round(train_secs / 60, 1),
        "steps_per_sec": round(steps / train_secs, 1) if train_secs else None,
        "eval_peak_step": diag.get("eval_peak_step"),
        "eval_peak": diag.get("eval_peak_value"),
        "explained_variance": diag.get("explained_variance_end"),
        "approx_kl": diag.get("approx_kl_late_mean"),
        "diag_bad": sum(1 for f in diag.get("findings", []) if f.get("level") == "bad"),
        "oos_trades": bt.get("total_trades"),
        "oos_pf": bt.get("profit_factor"),
        "oos_pf_ci95_lo": pf_ci[0],
        "oos_pf_ci95_hi": pf_ci[1],
        "oos_return": bt.get("return_pct"),
        "oos_max_dd": bt.get("max_drawdown"),
        "oos_dd95_worst": dd_conf.get("0.95"),
        "oos_sortino": bt.get("sortino"),
        "bootstrap_verdict": boot.get("verdict"),
    }


def fmt(v, kind="num"):
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v*100:.1f}%"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:.2f}"


def write_markdown(rows, path, args, trim_info=None):
    lines = [f"# Observation ablation — {args.tag}", "",
             f"- date: {datetime.now():%Y-%m-%d %H:%M}",
             f"- csv: `{args.csv}` · steps/arm: {args.steps:,} · train_pct: {args.train_pct} · "
             f"OOS backtest: best checkpoint, `--start {args.oos_start}`, conf 0"
             + (f" · seeds: {args.seeds}" if args.seeds else ""),
             f"- recipe: " + ", ".join(f"{k}={v}" for k, v in RECIPE.items())]
    if trim_info:
        lines.append(f"- train_start {trim_info['train_start']}: dropped {trim_info['rows_dropped']:,} rows "
                     f"of `{Path(trim_info['source_csv']).name}` · train {trim_info['train_rows']:,} rows "
                     f"({trim_info['train_span']}) · validation {trim_info['validation_span']} · "
                     f"test {trim_info['test_span']} (same Val/Test rows as the untrimmed split)")
    lines += ["",
              "| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | "
              "OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        seed = "—" if r.get("seed") is None else str(r["seed"])
        if r.get("features") is None:
            # training itself failed — nothing to show but the reason
            lines.append(f"| {r['arm']} | {seed} | — | — | — | — | — | — | — | — | — | — | — | "
                         f"{r.get('error', 'no data')} | — | — | — | — |")
            continue
        peak = (f"{fmt(r['eval_peak'])} @ {fmt(r['eval_peak_step'], 'int')}"
                if r.get("eval_peak") is not None else "—")
        ci = (f"[{fmt(r['oos_pf_ci95_lo'])}, {fmt(r['oos_pf_ci95_hi'])}]"
              if r.get("oos_pf_ci95_lo") is not None else "—")
        if r.get("error"):
            ci = r["error"]
        elif r.get("model_source", "best") != "best":
            ci += f" ({r['model_source']})"
        lines.append(
            f"| {r['arm']} | {seed} | {r['window']} | {r['corr_threshold']} | {fmt(r['features'],'int')} | "
            f"{fmt(r['obs_dim'],'int')} | {r['train_minutes']} | {fmt(r['steps_per_sec'])} | {peak} | "
            f"{fmt(r['explained_variance'])} | {fmt(r['approx_kl'])} | {fmt(r['oos_trades'],'int')} | "
            f"{fmt(r['oos_pf'])} | {ci} | {fmt(r['oos_return'],'pct')} | {fmt(r['oos_max_dd'],'pct')} | "
            f"{fmt(r['oos_dd95_worst'],'pct')} | {fmt(r['oos_sortino'])} |")
    lines += ["", "Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF "
              "point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than "
              "0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least "
              "1.5x faster. Single runs — rerun the winner and A once before trusting a close call."]
    summary = summarize_seeds(rows)
    if summary:
        lines += ["", "## Across seeds", "",
                  "| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | "
                  "mean return | pooled trades | pooled PF |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for sm in summary:
            lines.append(f"| {sm['arm']} | {sm['n_seeds']} | {fmt(sm['pf_mean'])} | "
                         f"{fmt(sm['pf_min'])}–{fmt(sm['pf_max'])} | {fmt(sm['pf_lo95_min'])} | "
                         f"{sm['seeds_pf_lo95_gt1']}/{sm['n_seeds']} | {fmt(sm['return_mean'], 'pct')} | "
                         f"{fmt(sm['pooled_trades'], 'int')} | {fmt(sm['pooled_pf'])} |")
        lines += ["", "Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades "
                  "(seeds share the same Test rows, so this is one sample with more trades, not "
                  "independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on "
                  "most seeds, not just a pooled point estimate above 1."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Phase 3.1 observation-size ablation")
    ap.add_argument("--csv", default="usdjpy_h4_2025_base.csv")
    ap.add_argument("--arms", default="A,B,C,D", help="comma list from " + ",".join(ARMS))
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--train_pct", type=float, default=0.70)
    ap.add_argument("--oos_start", type=float, default=0.85,
                    help="backtest start fraction = Test slice (Validation = train_pct..oos_start)")
    ap.add_argument("--max_hold", type=int, default=35)
    ap.add_argument("--mc_eval", type=int, default=0, help="quick-eval MC runs in rl_train (0 = off)")
    ap.add_argument("--mc", type=int, default=1000, help="order-shuffle MC in the OOS backtest")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seeds", default="",
                    help="comma list, e.g. 0,1,2 -> one model per (arm, seed), named <name>_s<seed>; "
                         "empty = one unseeded run per arm")
    ap.add_argument("--train_start", default="",
                    help="YYYY-MM-DD: drop rows before this date for TRAINING only; the Validation/Test "
                         "rows stay exactly those of the untrimmed --train_pct split (writes "
                         "<csv>_from_<date>.csv and recomputes the fractions)")
    ap.add_argument("--prefix", default="abl")
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arm(s): {unknown}")
    if not Path(args.csv).is_file():
        ap.error(f"csv not found: {args.csv}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds else [None]
    trim_info = None
    if args.train_start:
        try:
            args.csv, args.train_pct, args.oos_start, trim_info = trim_for_train_start(
                args.csv, args.train_start, args.train_pct)
        except ValueError as exc:
            ap.error(str(exc))
        print(f"[trim] {Path(trim_info['source_csv']).name} -> {args.csv}: dropped "
              f"{trim_info['rows_dropped']:,} rows; train {trim_info['train_rows']:,} rows "
              f"({trim_info['train_span']}); validation {trim_info['validation_span']}; "
              f"test {trim_info['test_span']}; effective --train_pct {args.train_pct} "
              f"--oos_start {args.oos_start}")

    py = sys.executable
    log_dir = WORK_DIR / "artifacts" / "ablation" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"=== obs ablation [{args.tag}] arms={arms} seeds={seeds} steps={args.steps:,} csv={args.csv}")

    for arm, seed in [(a, sd) for a in arms for sd in seeds]:
        window, thr = ARMS[arm]
        name = f"{args.prefix}_{arm}_w{window}_t{int(round(thr * 100)):03d}"
        if seed is not None:
            name += f"_s{seed}"
        train_cmd = [py, "rl_train.py", args.csv, "--name", name,
                     "--steps", str(args.steps), "--window", str(window),
                     "--corr_threshold", str(thr), "--train_pct", str(args.train_pct),
                     "--max_hold", str(args.max_hold), "--mc_eval", str(args.mc_eval)]
        if seed is not None:
            train_cmd += ["--seed", str(seed)]
        for k, v in RECIPE.items():
            train_cmd += [f"--{k}", v]
        diag_cmd = [py, "train_diagnose.py", name]
        bt_cmd = [py, "backtest_live.py", name, args.csv, "--conf", "0",
                  "--mode", "pure_agent", "--source", "best",
                  "--start", str(args.oos_start), "--max_hold", str(args.max_hold),
                  "--mc", str(args.mc), "--bootstrap", str(args.bootstrap),
                  "--random_baseline", "0"]

        print(f"\n--- arm {arm} seed {seed}: window={window} corr_threshold={thr} -> {name}")
        if args.dry_run:
            for c in (train_cmd, diag_cmd, bt_cmd):
                print("   ", " ".join(c))
            continue

        rc, train_secs = run_step(train_cmd, log_dir / f"{name}.train.log", "train")
        if rc != 0:
            rows.append({"arm": arm, "seed": seed, "model": name, "error": f"train failed rc={rc}"})
            continue
        run_step(diag_cmd, log_dir / f"{name}.diag.log", "diag")
        rc, _ = run_step(bt_cmd, log_dir / f"{name}.backtest.log", "backtest")
        source_used = "best"
        if rc != 0 and "model not found" in (log_dir / f"{name}.backtest.log").read_text(
                encoding="utf-8", errors="replace"):
            # No best checkpoint (eval never fired — too few steps). Fall back
            # to the final model so the arm still gets an OOS number, and say so.
            print("    [backtest] no best checkpoint -> retrying with --source final", flush=True)
            bt_final = [("final" if c == "best" else c) for c in bt_cmd]
            rc, _ = run_step(bt_final, log_dir / f"{name}.backtest.log", "backtest(final)")
            source_used = "final (no best checkpoint)"
        row = {"arm": arm, "seed": seed, **collect(name, train_secs, args.steps),
               "model_source": source_used}
        if rc != 0:
            row["error"] = f"backtest failed rc={rc}"
        rows.append(row)
        # persist after every arm so a crash mid-way loses nothing
        (WORK_DIR / "docs" / f"ablation_{args.tag}.json").write_text(
            json.dumps({"args": vars(args), "recipe": RECIPE, "trim": trim_info, "rows": rows,
                        "across_seeds": summarize_seeds(rows)}, indent=2),
            encoding="utf-8")
        write_markdown(rows, WORK_DIR / "docs" / f"ablation_{args.tag}.md", args, trim_info)

    if args.dry_run:
        return 0
    print(f"\n=== done: docs/ablation_{args.tag}.md")
    for r in rows:
        label = r['arm'] if r.get('seed') is None else f"{r['arm']} s{r['seed']}"
        if "error" in r:
            print(f"  {label}: {r['error']}")
        else:
            print(f"  {label}: obs {r['obs_dim']}  {r['train_minutes']} min  "
                  f"PF {fmt(r['oos_pf'])} CI [{fmt(r['oos_pf_ci95_lo'])}, {fmt(r['oos_pf_ci95_hi'])}]  "
                  f"DD {fmt(r['oos_max_dd'], 'pct')}  EV {fmt(r['explained_variance'])}")
    for sm in summarize_seeds(rows):
        print(f"  {sm['arm']} across {sm['n_seeds']} seeds: PF mean {fmt(sm['pf_mean'])} "
              f"[{fmt(sm['pf_min'])}, {fmt(sm['pf_max'])}]  pooled PF {fmt(sm['pooled_pf'])} "
              f"on {fmt(sm['pooled_trades'], 'int')} trades  CI-lo>1: "
              f"{sm['seeds_pf_lo95_gt1']}/{sm['n_seeds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
