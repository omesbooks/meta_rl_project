# Observation ablation — 20261009_run1

- date: 2026-10-09 19:32
- csv: `usdjpy_h4_2025_base.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| A | 30 | 1.0 | 181 | 5,433 | 22.5 | 148.10 | 6.03 @ 90,000 | -0.06 | 0.04 | 213 | 0.85 | [0.55, 1.33] | -7.5% | -16.0% | -23.4% | -0.43 |
| B | 30 | 0.95 | 85 | 2,553 | 12.5 | 266.60 | 9.11 @ 200,000 | -0.22 | 0.03 | 196 | 0.74 | [0.48, 1.09] | -18.0% | -25.8% | -36.4% | -1.20 |
| C | 15 | 1.0 | 181 | 2,718 | 13.5 | 247.60 | 4.33 @ 80,000 | 0.09 | 0.03 | 81 | 0.63 | [0.35, 1.20] | -17.8% | -22.2% | -35.8% | -1.11 |
| D | 15 | 0.95 | 85 | 1,278 | 11.5 | 290.70 | 0.25 @ 140,000 | 0.15 | 0.02 | 149 | 1.00 | [0.62, 1.62] | -0.8% | -10.8% | -22.8% | 0.01 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.
