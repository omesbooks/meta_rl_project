# Observation ablation — h1_srb

- date: 2026-10-12 20:42
- csv: `usdjpy_tds_h1_srb.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 41 | 618 | 21.2 | 157.60 | -6.93 @ 100,000 | -0.31 | 0.01 | 1,500 | 0.92 | [0.77, 1.08] | -9.7% | -14.6% | -25.8% | -0.76 |
| C | 1 | 15 | 1.0 | 41 | 618 | 21.8 | 152.70 | 7.22 @ 190,000 | -0.31 | 0.01 | 1,180 | 0.98 | [0.83, 1.17] | -2.5% | -9.2% | -20.9% | -0.15 |
| C | 2 | 15 | 1.0 | 41 | 618 | 27.7 | 120.50 | -0.80 @ 200,000 | -0.32 | 0.01 | 1,365 | 0.79 | [0.65, 0.96] | -23.5% | -27.5% | -37.0% | -1.85 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 3 | 0.90 | 0.79–0.98 | 0.65 | 0/3 | -11.9% | 4,045 | 0.89 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
