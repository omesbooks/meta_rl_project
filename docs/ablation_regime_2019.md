# Observation ablation — regime_2019

- date: 2026-10-10 18:53
- csv: `usdjpy_h4_2025_base_from_2019-01-01.csv` · steps/arm: 200,000 · train_pct: 0.5286 · OOS backtest: best checkpoint, `--start 0.7643`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4
- train_start 2019-01-01: dropped 6,233 rows of `usdjpy_h4_2025_base.csv` · train 5,767 rows (2019.01.02 00:00 -> 2022.09.13 00:00) · validation 2022.09.13 04:00 -> 2024.05.07 00:00 · test 2024.05.07 04:00 -> 2025.12.30 16:00 (same Val/Test rows as the untrimmed split)

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| D | 0 | 15 | 0.95 | 82 | 1,233 | 19.9 | 167.90 | -6.08 @ 200,000 | 0.55 | 0.01 | 154 | 1.05 | [0.69, 1.62] | 0.9% | -4.4% | -7.2% | 0.09 |
| D | 1 | 15 | 0.95 | 82 | 1,233 | 17.8 | 186.90 | 0.28 @ 50,000 | 0.36 | 0.02 | 95 | 0.90 | [0.49, 1.48] | -6.3% | -15.9% | -30.6% | -0.32 |
| D | 2 | 15 | 0.95 | 82 | 1,233 | 17.8 | 187.20 | -3.50 @ 40,000 | 0.49 | 0.01 | 315 | 0.56 | [0.39, 0.79] | -35.1% | -37.8% | -50.3% | -2.64 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| D | 3 | 0.84 | 0.56–1.05 | 0.39 | 0/3 | -13.5% | 564 | 0.73 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
