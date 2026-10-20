# Observation ablation — startfit

- date: 2026-10-20 15:45
- csv: `usdjpy_h4_2025_base_from_2019-01-01.csv` · steps/arm: 60,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2,3,4
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 180 | 2,703 | 23.6 | 42.40 | 4.32 @ 20,000 | -0.16 | 0.01 | 165 | 1.06 | [0.69, 1.63] | 2.0% | -11.1% | -15.3% | 0.26 |
| C | 1 | 15 | 1.0 | 180 | 2,703 | 8.8 | 114.20 | -0.37 @ 40,000 | -0.18 | 0.01 | 152 | 0.82 | [0.52, 1.34] | -6.7% | -10.5% | -20.1% | -0.74 |
| C | 2 | 15 | 1.0 | 180 | 2,703 | 5.7 | 174.60 | 3.37 @ 40,000 | -0.11 | 0.01 | 178 | 0.77 | [0.51, 1.20] | -9.7% | -14.9% | -22.5% | -1.12 |
| C | 3 | 15 | 1.0 | 180 | 2,703 | 6.0 | 165.50 | -1.39 @ 40,000 | -0.17 | 0.01 | 204 | 0.78 | [0.52, 1.12] | -9.3% | -17.3% | -21.7% | -1.06 |
| C | 4 | 15 | 1.0 | 180 | 2,703 | 6.5 | 152.90 | 4.57 @ 50,000 | -0.16 | 0.01 | 158 | 0.65 | [0.42, 1.01] | -14.3% | -14.8% | -26.6% | -1.69 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 5 | 0.82 | 0.65–1.06 | 0.42 | 0/5 | -7.6% | 857 | 0.81 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
