# Observation ablation — 31b_v2_s2

- date: 2026-10-10 19:20
- csv: `usdjpy_h4_2025_v2.csv` · steps/arm: 200,000 · train_pct: 0.7 · OOS backtest: best checkpoint, `--start 0.85`, conf 0 · seeds: 2
- recipe: learning_rate=5e-5, clip_range=0.1, ent_coef=0.02, n_steps=4096, n_epochs=15, batch_size=128, gamma=0.99, gae_lambda=0.97, vf_coef=0.7, net_arch=256,128,64, ep_len=2000, reward_mode=realized, reward_profile=balanced, action_profile=basic_4

| arm | seed | window | thr | feats | obs | train min | steps/s | eval peak @ | EV | KL | OOS trades | PF | PF 95% CI | return | max DD | DD 95% worst | sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| D | 2 | 15 | 0.95 | 103 | 1,548 | 24.0 | 138.60 | -9.09 @ 90,000 | -0.01 | 0.02 | 242 | 0.68 | [0.46, 1.01] | -21.6% | -27.3% | -37.1% | -1.53 |

Decision rule (non-inferiority vs arm A): accept a smaller arm when its OOS PF point estimate lies inside A's PF 95% CI, its PF 95% lower bound is not more than 0.10 below A's, its max DD is not more than 2 pp worse, and it trains at least 1.5x faster. Single runs — rerun the winner and A once before trusting a close call.
