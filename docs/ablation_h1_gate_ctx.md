# Observation ablation — h1_gate_ctx

- date: 2026-10-13 17:52
- csv: `usdjpy_tds_h1_srb_ctx.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 0,1,2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C | 0 | 15 | 1.0 | 63 | 948 | 14.5 | 230.20 | -0.66 @ 10,000 | -0.30 | 0.01 | 85 | 1.36 | [0.78, 2.50] | 2.9% | -2.1% | -4.0% | 0.24 |
| C | 1 | 15 | 1.0 | 63 | 948 | 15.9 | 209.80 | 0.58 @ 200,000 | -0.82 | 0.01 | 60 | 1.00 | [0.49, 2.08] | -0.0% | -2.0% | -4.7% | -0.00 |
| C | 2 | 15 | 1.0 | 63 | 948 | 15.1 | 221.40 | 1.00 @ 200,000 | -0.70 | 0.01 | 59 | 0.53 | [0.25, 1.10] | -3.7% | -4.0% | -8.0% | -0.25 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.

## Across seeds

| arm | seeds | PF mean | PF min–max | min PF CI lo | seeds w/ CI lo > 1 | mean return | pooled trades | pooled PF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| C | 3 | 0.96 | 0.53–1.36 | 0.25 | 0/3 | -0.3% | 204 | 0.96 |

Pooled PF = sum of winning pnl / sum of losing pnl over every seed's Test trades (seeds share the same Test rows, so this is one sample with more trades, not independent evidence). An edge claim needs the PF 95% CI lower bound > 1.0 on most seeds, not just a pooled point estimate above 1.
