# Observation ablation — regime_full

- date: 2026-10-10 18:53
- csv: `usdjpy_h4_2025_base.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| D | 0 | 15 | 0.95 | 85 | 1,278 | 19.9 | 167.50 | 2.39 @ 30,000 | 0.04 | 0.02 | 345 | 0.83 | [0.59, 1.16] | -12.8% | -21.1% | -30.7% | -0.96 |
| D | 1 | 15 | 0.95 | 85 | 1,278 | 18.0 | 185.70 | 5.84 @ 80,000 | -0.04 | 0.02 | 71 | 0.94 | [0.47, 1.75] | -2.7% | -13.6% | -23.2% | -0.12 |
| D | 2 | 15 | 0.95 | 85 | 1,278 | 18.1 | 183.70 | -2.55 @ 90,000 | 0.15 | 0.02 | 81 | 0.96 | [0.55, 1.79] | -2.2% | -12.5% | -24.3% | -0.07 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| D | 3 | 0.91 | 0.83–0.96 | 0.47 | 0/3 | -5.9% | 497 | 0.89 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
