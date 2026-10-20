# start_mode comparison: fit vs uniform (2026-10-20)

csv `usdjpy_h4_2025_base_from_2019-01-01.csv` · arm C · 60,000 steps · seeds [0, 1, 2, 3, 4] · ep_len 2000 · OOS = last 15% (backtest, best checkpoint)

**fit** (default): an episode starts where a full episode fits, so the last ep_len bars of Train are only reached as episode tails. **uniform**: any bar can start an episode; a late start plays a shorter one.

## Per seed

| seed | mode | Train covered | last 10% | plays/bar | episodes | best val reward | OOS PF | PF CI95 | OOS return | trades | max DD |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | fit | 97.6% | 78.2% | 8.06 | 31 | 4.32 | 1.06 | [0.69, 1.63] | 0.02% | 165 | -11.1% |
| 0 | uniform | 99.7% | 99.9% | 8.06 | 36 | -1.72 | 0.67 | [0.46, 1.02] | -0.11% | 217 | -13.2% |
| 1 | fit | 94.3% | 63.2% | 8.06 | 31 | -0.37 | 0.82 | [0.52, 1.34] | -0.07% | 152 | -10.5% |
| 1 | uniform | 97.2% | 99.9% | 8.06 | 36 | -5.58 | 1.00 | [0.68, 1.51] | -0.00% | 163 | -10.0% |
| 2 | fit | 95.4% | 94.5% | 8.06 | 31 | 3.37 | 0.77 | [0.51, 1.20] | -0.10% | 178 | -14.9% |
| 2 | uniform | 94.5% | 99.9% | 8.06 | 37 | 0.28 | 0.90 | [0.57, 1.41] | -0.04% | 159 | -8.3% |
| 3 | fit | 89.3% | 17.1% | 8.06 | 31 | -1.39 | 0.78 | [0.52, 1.12] | -0.09% | 204 | -17.3% |
| 3 | uniform | 96.7% | 99.9% | 8.06 | 34 | 0.02 | 0.79 | [0.45, 1.30] | -0.06% | 120 | -10.4% |
| 4 | fit | 94.3% | 88.1% | 8.06 | 31 | 4.57 | 0.65 | [0.42, 1.01] | -0.14% | 158 | -14.8% |
| 4 | uniform | 93.9% | 99.9% | 8.06 | 41 | 0.27 | 0.65 | [0.42, 0.97] | -0.15% | 195 | -17.8% |

## Paired differences (uniform − fit), same seed

| metric | mean diff | bootstrap CI95 | uniform better in |
|---|---:|---|---:|
| Train covered | 2.2% | [-0.1%, 4.8%] | 3/5 |
| last-10% coverage | 31.7% | [11.2%, 58.1%] | 5/5 |
| best val reward | -3.44 | [-5.36, -0.98] | 1/5 |
| OOS PF | -0.02 | [-0.21, 0.13] | 3/5 |
| OOS return % | 0.00 | [-0.07, 0.06] | 3/5 |
| OOS max DD (lower better) | 1.8% | [-1.9%, 5.5%] | 2/5 |

## Reading it

- best val reward: uniform better in 1/5 seeds, mean diff -3.44 (CI [-5.36, -0.98]) → **CI excludes 0** — a real difference at this sample size
- OOS PF: uniform better in 3/5 seeds, mean diff -0.02 (CI [-0.21, 0.13]) → CI includes 0 — not separable from seed noise with these seeds
- OOS return %: uniform better in 3/5 seeds, mean diff 0.00 (CI [-0.07, 0.06]) → CI includes 0 — not separable from seed noise with these seeds

Coverage differences are mechanical (they follow from the sampling rule); only the OOS/validation rows say whether it matters for trading. 5 seeds × 1 OOS window is a small sample: treat a CI that includes 0 as 'no evidence', not 'no effect'.
