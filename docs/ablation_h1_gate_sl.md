# Observation ablation — h1_gate_sl

- date: 2026-10-13 16:56
- csv: `usdjpy_tds_h1_srb.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 41 | 618 | 33.4 | 99.70 | -0.14 @ 40,000 | -0.30 | 0.02 | 77 | 1.09 | [0.63, 2.01] | 0.6% | -2.1% | -3.7% | 0.05 |
| C | 1 | 15 | 1.0 | 41 | 618 | 22.8 | 146.40 | 1.18 @ 90,000 | -0.43 | 0.01 | 63 | 1.38 | [0.74, 2.80] | 1.9% | -1.6% | -2.5% | 0.14 |
| C | 2 | 15 | 1.0 | 41 | 618 | 17.8 | 187.40 | 0.11 @ 140,000 | -0.91 | 0.01 | 52 | 1.97 | [0.98, 4.13] | 2.8% | -1.4% | -1.5% | 0.25 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 3 | 1.48 | 1.09–1.97 | 0.63 | 0/3 | 1.7% | 192 | 1.35 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
