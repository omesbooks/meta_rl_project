# ML4T Reference — Index

Mirror of chapter READMEs from **Stefan Jansen, _Machine Learning for Algorithmic
Trading_ (2nd Edition, Packt, 2020)** — companion repo
[stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading).

Each file here is the chapter README copied verbatim (overview + linked notebooks
+ key concepts). Code, notebooks, and data are NOT mirrored — clone the source
repo for those.

---

## Most relevant to this project (PPO forex/gold RL)

Read these first when you need theory grounding for a specific task.

| Chapter | Most useful for |
|---|---|
| **05_strategy_evaluation** | Backtest pitfalls, performance metrics (Sharpe, PF, drawdown, IR), pyfolio. Read when interpreting `backtest_live.py` results. |
| **06_machine_learning_process** | Train/test split discipline, cross-validation that doesn't leak, bias-variance trade-off. Read before tuning anything. |
| **08_ml4t_workflow** | End-to-end ML4T pipeline (Zipline/Quantopian style). Mirror of our own pipeline structure. |
| **09_time_series_models** | Stationarity tests, ARIMA, structural breaks. Useful for understanding the regime-shift issue in our GBP/USD data. |
| **20_autoencoders_for_conditional_risk_factors** | Latent-factor/regime extraction with autoencoders. Relevant to the Regime Check page (HMM/PELT alternatives). |
| **22_deep_reinforcement_learning** ⭐ | RL for trading — value, policy, actor-critic, Q-learning, OpenAI baselines, building a custom env. This is our chapter. |
| **24_alpha_factor_library** | Pre-built factor implementations — ideas for extending `RL_Indicators.mqh` and feature columns. |

## Background / occasionally useful

| Chapter | Relevance |
|---|---|
| 01_machine_learning_for_trading | Book overview, scope, what ML does well/poorly in markets |
| 02_market_and_fundamental_data | Data sources, exchange feeds, tick vs bar data |
| 03_alternative_data | News, satellite, sentiment data — out of scope for this project |
| 04_alpha_factor_research | Factor research workflow (Alphalens) — relevant when adding features |
| 07_linear_models | OLS, regularization, logistic regression baselines |
| 10_bayesian_machine_learning | Uncertainty quantification, PyMC — useful for confidence calibration |
| 11_decision_trees_random_forests | Tree models — baseline to compare against PPO |
| 12_gradient_boosting_machines | XGBoost/LightGBM/CatBoost — strong supervised baseline |
| 13_unsupervised_learning | Clustering, PCA — ties into K-Means regime detection |

## Not directly relevant

NLP-heavy (14–16), CNN (18), generative (21), text-data (14) are listed in
`00_main_README.md` for completeness but unlikely to come up for our forex RL
work. Skip unless explicitly needed.

## Setup reference

- `installation.md` — Python/conda setup from the source repo. Not used directly
  (we already have our own `requirements.txt`), but useful to compare which
  versions Jansen tested.

---

## How to use this with the rest of the project

1. **Quick lookup** — grep a topic across all files:
   ```bash
   grep -rn -i "stationarity" reference/ml4t/
   grep -rn -i "sharpe" reference/ml4t/
   ```
2. **Read one chapter** — open the relevant `<NN>_<topic>.md` directly.
3. **Don't load the whole folder** into an AI context. Files vary 80–520 lines;
   pick by topic.

## Related project docs

- `reference/MIT-Quant-Bible.md` — math foundations (probability, statistics,
  least squares). Lighter and more general than ML4T.
- `AGENTS.md` — codebase guide. The "Reference Documents" section links here.
- `graphify-out/GRAPH_REPORT.md` — code dependency graph for this project.
