# Observation ablation — h1_gate

- date: 2026-10-13 16:46
- csv: `usdjpy_tds_h1_srb.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 41 | 618 | 28.0 | 119.10 | 0.27 @ 10,000 | -0.38 | 0.01 | 47 | 2.35 | [1.03, 6.26] | 4.0% | -1.3% | -1.7% | 0.33 |
| C | 1 | 15 | 1.0 | 41 | 618 | 20.0 | 166.50 | 0.54 @ 130,000 | -1.00 | 0.01 | 41 | 1.71 | [0.71, 3.76] | 1.6% | -0.8% | -1.5% | 0.14 |
| C | 2 | 15 | 1.0 | 41 | 618 | 17.2 | 194.20 | 2.46 @ 20,000 | -0.38 | 0.01 | 101 | 2.12 | [1.22, 3.92] | 7.8% | -1.5% | -2.4% | 0.55 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 3 | 2.06 | 1.71–2.35 | 0.71 | 2/3 | 4.4% | 189 | 2.09 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
