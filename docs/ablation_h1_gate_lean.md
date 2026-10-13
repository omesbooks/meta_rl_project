# Observation ablation — h1_gate_lean

- date: 2026-10-13 19:03
- csv: `usdjpy_tds_h1_lean.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 29 | 438 | 18.2 | 182.80 | 0.86 @ 90,000 | -0.22 | 0.01 | 112 | 1.77 | [1.02, 3.13] | 4.6% | -0.8% | -2.3% | 0.41 |
| C | 1 | 15 | 1.0 | 29 | 438 | 12.9 | 258.30 | 0.87 @ 20,000 | -0.55 | 0.01 | 32 | 1.35 | [0.59, 3.17] | 1.0% | -1.0% | -2.2% | 0.11 |
| C | 2 | 15 | 1.0 | 29 | 438 | 13.5 | 246.10 | 0.04 @ 90,000 | -0.56 | 0.01 | 67 | 0.93 | [0.50, 1.96] | -0.5% | -3.7% | -4.9% | -0.04 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 3 | 1.35 | 0.93–1.77 | 0.50 | 1/3 | 1.7% | 211 | 1.31 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
