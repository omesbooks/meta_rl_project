# Observation ablation — h1_base

- date: 2026-10-12 20:41
- csv: `usdjpy_tds_h1_base.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 28 | 423 | 20.8 | 160.10 | -6.54 @ 20,000 | -0.15 | 0.01 | 1,253 | 1.03 | [0.86, 1.23] | 2.2% | -6.9% | -18.5% | 0.21 |
| C | 1 | 15 | 1.0 | 28 | 423 | 21.6 | 154.50 | 1.14 @ 30,000 | -0.34 | 0.01 | 1,111 | 1.04 | [0.85, 1.26] | 3.4% | -6.7% | -17.4% | 0.28 |
| C | 2 | 15 | 1.0 | 28 | 423 | 26.7 | 124.90 | -2.41 @ 70,000 | -0.32 | 0.01 | 1,236 | 1.10 | [0.90, 1.31] | 9.7% | -9.9% | -14.1% | 0.71 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 3 | 1.06 | 1.03–1.10 | 0.85 | 0/3 | 5.1% | 3,600 | 1.05 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
