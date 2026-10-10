"""Signal-level feature screen — run BEFORE putting features into RL.

Question: does any feature, on its own or combined linearly, predict the
direction of future returns in a way that is STABLE across time?

For every feature and horizon h (bars), on the same 70/15/15 split the RL
experiments use (Train / Validation / Test, time-ordered):
  - IC         Spearman rank correlation of feature[t] with log(close[t+h]/close[t])
               over the WHOLE slice
  - z          IC / moving-block-bootstrap SE on Train (blocks of --block bars,
               handles overlapping returns and persistent features)
  - vol IC     Spearman with |return| over h (predicts how much, not which way)
  - drift      median |z-score| of the feature on Val/Test using Train mean/std.
               > 2 = non-stationary (price levels, cumulative OBV/PVT...): its IC
               is biased negative even on a random walk (finite-sample unit-root
               bias) and the RL net sees values it never saw in training.
Stable = stationary AND |z| >= 3 on Train AND same-sign IC on Validation AND Test.

DO NOT rank-correlate inside short windows (e.g. 250-bar blocks): for
persistent features that is biased toward negative IC — a shuffled random
walk produces block t-stats of -4 (RSI 14) to -14 (EMA-20 level), the same as
the real data (verified 2026-10-10, docs/feature_screen_usdjpy_h4_2025_base.md).
Also: cross-sectional persistence (do Train ICs predict Validation ICs across
features?) and a ridge model on all features (alpha picked on Validation,
reported once on Test).

Usage (repo root):
    python tools/analysis/feature_screen.py usdjpy_h4_2025_base.csv
    python tools/analysis/feature_screen.py xauusd_h4_2025_v2.csv --drop "_div_|^div_|^near_|^range_pos_(100|250|1500)$"
Writes docs/feature_screen_<csv stem>.md and .csv
"""
import argparse
import io
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training_data import feature_columns, read_csv_cached  # noqa: E402


def rank_cols(a):
    """Column-wise average ranks of a 2-D array (NaN-free)."""
    return pd.DataFrame(a).rank(axis=0).to_numpy()


def spearman_many(X, y):
    """Spearman IC of every column of X with y (rows with NaN y dropped)."""
    ok = ~np.isnan(y)
    if ok.sum() < 10:
        return np.full(X.shape[1], np.nan)
    rx = rank_cols(X[ok])
    ry = pd.Series(y[ok]).rank().to_numpy()
    rx = rx - rx.mean(0)
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry[:, None]).sum(0) / den


def boot_se(X, y, block, n_boot=200, seed=11):
    """Moving-block bootstrap SE of the whole-slice Spearman IC (per column)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    nb = max(1, n // block)
    starts = np.arange(0, n - block + 1)
    ics = []
    for _ in range(n_boot):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(block)).ravel()
        ics.append(spearman_many(X[idx], y[idx]))
    return np.nanstd(np.array(ics), axis=0, ddof=1)


def ridge_ic(Xtr, ytr, Xva, yva, Xte, yte, alphas=(1, 10, 100, 1000, 10000)):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    Z = lambda X: np.clip((X - mu) / sd, -6, 6)
    A, B, C = Z(Xtr), Z(Xva), Z(Xte)
    ok = ~np.isnan(ytr)
    A, yt = A[ok], ytr[ok]
    ym, ys = yt.mean(), yt.std()
    yt = (yt - ym) / ys
    best = None
    for a in alphas:
        w = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ yt)
        ic_tr = spearman_many((A @ w)[:, None], ytr[ok])[0]
        ic_va = spearman_many((B @ w)[:, None], yva)[0]
        if best is None or ic_va > best[2]:
            best = (a, ic_tr, ic_va, w)
    a, ic_tr, ic_va, w = best
    ic_te = spearman_many((C @ w)[:, None], yte)[0]
    return a, ic_tr, ic_va, ic_te


def main():
    ap = argparse.ArgumentParser(description="Signal-level feature screen")
    ap.add_argument("csv")
    ap.add_argument("--train_pct", type=float, default=0.70)
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--block", type=int, default=250, help="block length (bars) of the bootstrap SE")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--drop", default="", help="regex of feature names to exclude")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    df = read_csv_cached(args.csv)
    feats = feature_columns(df)
    if args.drop:
        feats = [f for f in feats if not re.search(args.drop, f)]
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = (np.nanstd(X, axis=0) > 0) & np.isfinite(X).all(0)
    dropped_const = [f for f, k in zip(feats, keep) if not k]
    feats = [f for f, k in zip(feats, keep) if k]
    X = X[:, keep]
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    split = int(n * args.train_pct)
    test0 = split + (n - split) // 2
    parts = {"train": slice(0, split), "val": slice(split, test0), "test": slice(test0, n)}
    H = [int(h) for h in args.horizons.split(",")]
    ts = df["timestamp"].astype(str)
    print(f"[screen] {Path(args.csv).name}: {n:,} rows, {len(feats)} features "
          f"(constant dropped: {dropped_const or 'none'})")
    for k, sl in parts.items():
        print(f"   {k:5s} {ts.iloc[sl].iloc[0]} -> {ts.iloc[sl].iloc[-1]} ({sl.stop - sl.start:,} bars)")

    res = pd.DataFrame({"feature": feats})
    tr = X[parts["train"]]
    mu, sdv = tr.mean(0), tr.std(0)
    sdv[sdv < 1e-12] = 1.0
    zv = np.abs(np.median((X[split:] - mu) / sdv, axis=0))
    res["drift_z"] = zv
    res["ar1"] = [np.corrcoef(tr[:-1, j], tr[1:, j])[0, 1] for j in range(tr.shape[1])]
    stationary = res["drift_z"] < 2
    print(f"   non-stationary (drift_z >= 2, excluded from 'stable' and ridge): "
          f"{', '.join(res.loc[~stationary, 'feature']) or 'none'}")
    summary = {}
    for h in H:
        fwd = np.full(n, np.nan)
        fwd[:-h] = np.log(close[h:] / close[:-h])
        ic_tr = spearman_many(X[parts["train"]], fwd[parts["train"]])
        se = boot_se(X[parts["train"]], fwd[parts["train"]], args.block, args.boot)
        res[f"ic_train_h{h}"] = ic_tr
        res[f"z_train_h{h}"] = ic_tr / se
        res[f"ic_val_h{h}"] = spearman_many(X[parts["val"]], fwd[parts["val"]])
        res[f"ic_test_h{h}"] = spearman_many(X[parts["test"]], fwd[parts["test"]])
        res[f"volic_train_h{h}"] = spearman_many(X[parts["train"]], np.abs(fwd[parts["train"]]))
        res[f"volic_val_h{h}"] = spearman_many(X[parts["val"]], np.abs(fwd[parts["val"]]))
        tt, iv, it = res[f"z_train_h{h}"], res[f"ic_val_h{h}"], res[f"ic_test_h{h}"]
        stable = stationary & (tt.abs() >= 3) & (np.sign(iv) == np.sign(tt)) & (np.sign(it) == np.sign(tt))
        res[f"stable_h{h}"] = stable
        st = res[stationary]
        pers_val = st[f"ic_train_h{h}"].corr(st[f"ic_val_h{h}"], method="spearman")
        pers_test = st[f"ic_train_h{h}"].corr(st[f"ic_test_h{h}"], method="spearman")
        sign_agree = float(np.mean(np.sign(st[f"ic_train_h{h}"]) == np.sign(st[f"ic_val_h{h}"])))
        cols = np.flatnonzero(stationary.to_numpy())
        a, ic_tr, ic_va, ic_te = ridge_ic(X[parts["train"]][:, cols], fwd[parts["train"]], X[parts["val"]][:, cols],
                                          fwd[parts["val"]], X[parts["test"]][:, cols], fwd[parts["test"]])
        from scipy import stats
        n_st = int(stationary.sum())
        exp2 = n_st * 2 * stats.norm.sf(2)
        exp3 = n_st * 2 * stats.norm.sf(3)
        tts = tt[stationary]
        k = n_st
        summary[h] = dict(blocks=k, n2=int((tts.abs() >= 2).sum()), n3=int((tts.abs() >= 3).sum()),
                          exp2=exp2, exp3=exp3, stable=int(stable.sum()), pers_val=pers_val,
                          pers_test=pers_test, sign_agree=sign_agree, ridge=(a, ic_tr, ic_va, ic_te),
                          max_abs_ic_val=float(res[f"ic_val_h{h}"].abs().max()),
                          med_volic=float(res[f"volic_train_h{h}"].abs().median()),
                          max_volic=float(res[f"volic_train_h{h}"].abs().max()))
        s = summary[h]
        print(f"\n[h={h}] stationary features {k} · |z|>=2: {s['n2']} (chance ~{exp2:.1f}) · |z|>=3: {s['n3']} "
              f"(chance ~{exp3:.1f}) · stable: {s['stable']}")
        print(f"       persistence Train->Val {pers_val:+.2f}, Train->Test {pers_test:+.2f}, "
              f"sign agreement Train/Val {sign_agree:.0%}")
        print(f"       ridge (alpha {a}): IC train {ic_tr:+.3f}  val {ic_va:+.3f}  test {ic_te:+.3f}")

    stem = Path(args.csv).stem
    out_csv = ROOT / "docs" / f"feature_screen_{stem}.csv"
    res.to_csv(out_csv, index=False, float_format="%.5f")

    hm = H[len(H) // 2]
    st_res = res[stationary]
    top = st_res.reindex(st_res[f"z_train_h{hm}"].abs().sort_values(ascending=False).index).head(args.top)
    L = [f"# Feature screen — `{Path(args.csv).name}`", "",
         f"{n:,} rows · {len(feats)} features · split {args.train_pct:.0%}/"
         f"{(1 - args.train_pct) / 2:.0%}/{(1 - args.train_pct) / 2:.0%} · IC over whole slices · "
         f"z = Train IC / moving-block bootstrap SE ({args.block}-bar blocks, {args.boot} resamples)",
         f"- Train {ts.iloc[0]} → {ts.iloc[split - 1]} · Validation → {ts.iloc[test0 - 1]} · Test → {ts.iloc[-1]}",
         f"- constant features dropped: {', '.join(dropped_const) or 'none'}",
         f"- non-stationary (drift_z ≥ 2, excluded from stable/ridge): "
         f"{', '.join(res.loc[~stationary, 'feature']) or 'none'}", "",
         "## Summary per horizon", "",
         "| horizon | |z|≥2 (chance) | |z|≥3 (chance) | stable | persistence Train→Val | Train→Test | sign agree Train/Val | ridge IC train / val / test | median |vol IC| |",
         "|---:|---|---|---:|---:|---:|---:|---|---:|"]
    for h, s in summary.items():
        a, i1, i2, i3 = s["ridge"]
        L.append(f"| {h} | {s['n2']} ({s['exp2']:.1f}) | {s['n3']} ({s['exp3']:.1f}) | {s['stable']} | "
                 f"{s['pers_val']:+.2f} | {s['pers_test']:+.2f} | {s['sign_agree']:.0%} | "
                 f"{i1:+.3f} / {i2:+.3f} / {i3:+.3f} (α {a}) | {s['med_volic']:.2f} |")
    L += ["", f"## Top {args.top} by Train stability (h={hm})", "",
          f"| feature | IC train | z | IC val | IC test | stable | vol IC train |",
          "|---|---:|---:|---:|---:|:-:|---:|"]
    for _, r in top.iterrows():
        L.append(f"| {r['feature']} | {r[f'ic_train_h{hm}']:+.3f} | {r[f'z_train_h{hm}']:+.1f} | "
                 f"{r[f'ic_val_h{hm}']:+.3f} | {r[f'ic_test_h{hm}']:+.3f} | {'✓' if r[f'stable_h{hm}'] else ''} | "
                 f"{r[f'volic_train_h{hm}']:+.3f} |")
    stable_any = sorted({f for h in H for f in res.loc[res[f'stable_h{h}'], 'feature']})
    L += ["", f"Stable in at least one horizon: {', '.join(stable_any) if stable_any else 'none'}",
          "", "Full table: `" + out_csv.name + "`"]
    out_md = out_csv.with_suffix(".md")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[screen] stable in any horizon: {stable_any or 'none'}")
    print(f"[screen] -> {out_md.relative_to(ROOT)}, {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
