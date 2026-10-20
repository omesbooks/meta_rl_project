# Observation ablation — startuni

- date: 2026-10-20 15:47
- csv: `usdjpy_h4_2025_base_from_2019-01-01.csv` · steps/arm: 60,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2,3,4
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 180 | 2,703 | 23.1 | 43.30 | -1.72 @ 20,000 | -0.21 | 0.01 | 217 | 0.67 | [0.46, 1.02] | -11.2% | -13.2% | -20.5% | -1.61 |
| C | 1 | 15 | 1.0 | 180 | 2,703 | 5.9 | 169.40 | -5.58 @ 50,000 | -0.04 | 0.01 | 163 | 1.00 | [0.68, 1.51] | -0.2% | -10.0% | -16.7% | 0.02 |
| C | 2 | 15 | 1.0 | 180 | 2,703 | 5.8 | 173.30 | 0.28 @ 30,000 | -0.28 | 0.01 | 159 | 0.90 | [0.57, 1.41] | -3.5% | -8.3% | -17.2% | -0.40 |
| C | 3 | 15 | 1.0 | 180 | 2,703 | 6.2 | 161.40 | 0.02 @ 20,000 | -0.10 | 0.01 | 120 | 0.79 | [0.45, 1.30] | -6.5% | -10.4% | -18.8% | -0.70 |
| C | 4 | 15 | 1.0 | 180 | 2,703 | 5.8 | 173.40 | 0.27 @ 40,000 | -0.11 | 0.01 | 195 | 0.65 | [0.42, 0.97] | -14.9% | -17.8% | -26.2% | -1.84 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 5 | 0.80 | 0.65–1.00 | 0.42 | 0/5 | -7.3% | 854 | 0.79 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
