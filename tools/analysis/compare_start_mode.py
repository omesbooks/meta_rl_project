"""Compare two obs_ablation tags that differ only in rl_train --start_mode (fit vs uniform).

    python tools/analysis/compare_start_mode.py --fit startfit --uniform startuni [--out docs/start_mode_<date>.md]

Per seed: Train coverage from <run>/episodes.json (share of bars played, last-10% coverage), the
best validation eval reward (evaluations.npz), and the OOS backtest (PF, return, trades, DD) that
obs_ablation already collected. Then paired differences across seeds (uniform - fit) with a sign
count and a bootstrap CI of the mean difference. Skeptical by design: 5 seeds is a small sample.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact_paths import model_dir  # noqa: E402


def per_seed(tag):
    doc = json.loads((ROOT / "docs" / f"ablation_{tag}.json").read_text(encoding="utf-8"))
    out = {}
    for r in doc["rows"]:
        if "error" in r:
            continue
        run = model_dir(r["model"])
        ep = json.loads((run / "episodes.json").read_text(encoding="utf-8")) if (run / "episodes.json").exists() else {}
        s = ep.get("summary", {})
        ev = run / "logs" / "evaluations.npz"
        best_val = None
        if ev.exists():
            z = np.load(ev)
            best_val = float(np.max(z["results"].mean(axis=1)))
        out[r["seed"]] = {"model": r["model"], "covered": s.get("covered"), "last10": s.get("last10"),
                          "first10": s.get("first10"), "passes": s.get("passes"), "episodes": s.get("episodes"),
                          "best_val": best_val, "pf": r.get("oos_pf"), "ret": r.get("oos_return"),
                          "trades": r.get("oos_trades"), "dd": r.get("oos_max_dd"),
                          "pf_lo": r.get("oos_pf_ci95_lo"), "pf_hi": r.get("oos_pf_ci95_hi")}
    return doc["args"], out


def boot_ci(d, n=5000, seed=0):
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], float)
    if len(d) < 2:
        return None
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f(v, d=2, pct=False):
    if v is None or not np.isfinite(v):
        return "–"
    return f"{100 * v:.1f}%" if pct else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="startfit"); ap.add_argument("--uniform", default="startuni")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    args_f, F = per_seed(a.fit)
    args_u, U = per_seed(a.uniform)
    seeds = sorted(set(F) & set(U))
    lines = [f"# start_mode comparison: fit vs uniform ({date.today()})", "",
             f"csv `{args_f['csv']}` · arm {args_f['arms']} · {args_f['steps']:,} steps · seeds {seeds} · "
             f"ep_len 2000 · OOS = last {100 * (1 - args_f['oos_start']):.0f}% (backtest, best checkpoint)", "",
             "**fit** (default): an episode starts where a full episode fits, so the last ep_len bars of Train are only "
             "reached as episode tails. **uniform**: any bar can start an episode; a late start plays a shorter one.", "",
             "## Per seed", "",
             "| seed | mode | Train covered | last 10% | plays/bar | episodes | best val reward | OOS PF | PF CI95 | OOS return | trades | max DD |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for s in seeds:
        for mode, R in (("fit", F[s]), ("uniform", U[s])):
            lines.append(f"| {s} | {mode} | {f(R['covered'], pct=True)} | {f(R['last10'], pct=True)} | {f(R['passes'])} | {R['episodes'] or '–'} | "
                         f"{f(R['best_val'])} | {f(R['pf'])} | [{f(R['pf_lo'])}, {f(R['pf_hi'])}] | {f(R['ret'], 2)}% | {R['trades'] or '–'} | {f(R['dd'], pct=True)} |")
    lines += ["", "## Paired differences (uniform − fit), same seed", "",
              "| metric | mean diff | bootstrap CI95 | uniform better in |", "|---|---:|---|---:|"]
    verdict = []
    for key, label, higher in (("covered", "Train covered", True), ("last10", "last-10% coverage", True),
                               ("best_val", "best val reward", True), ("pf", "OOS PF", True),
                               ("ret", "OOS return %", True), ("dd", "OOS max DD (lower better)", False)):
        d = [(U[s][key] - F[s][key]) if (U[s][key] is not None and F[s][key] is not None) else None for s in seeds]
        ci = boot_ci(d)
        wins = sum(1 for x in d if x is not None and ((x > 0) if higher else (x < 0)))
        n = sum(1 for x in d if x is not None)
        pct = key in ("covered", "last10", "dd")
        lines.append(f"| {label} | {f(ci[0], pct=pct) if ci else '–'} | {'[' + f(ci[1], pct=pct) + ', ' + f(ci[2], pct=pct) + ']' if ci else '–'} | {wins}/{n} |")
        if ci and key in ("pf", "ret", "best_val"):
            verdict.append((label, ci, wins, n))
    lines += ["", "## Reading it", ""]
    for label, ci, wins, n in verdict:
        clear = (ci[1] > 0) or (ci[2] < 0)
        lines.append(f"- {label}: uniform better in {wins}/{n} seeds, mean diff {f(ci[0])} (CI [{f(ci[1])}, {f(ci[2])}]) → "
                     + ("**CI excludes 0** — a real difference at this sample size" if clear else "CI includes 0 — not separable from seed noise with these seeds"))
    lines += ["", "Coverage differences are mechanical (they follow from the sampling rule); only the OOS/validation rows say whether it matters for trading. "
              "5 seeds × 1 OOS window is a small sample: treat a CI that includes 0 as 'no evidence', not 'no effect'.", ""]
    text = "\n".join(lines)
    out = Path(a.out) if a.out else ROOT / "docs" / f"start_mode_{date.today().isoformat()}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
