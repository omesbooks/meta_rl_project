# AGENTS.md — Meta RL Trading Project

> Guide for AI agents (Claude Code, Codex, Cursor) working on this codebase.
> Last updated: 2026-05-26
> For a full dependency map see `graphify-out/GRAPH_REPORT.md` (534 nodes, 927 edges).

---

## 1. What This Is

End-to-end **Reinforcement Learning (PPO)** trading system for forex/gold (EUR/USD, XAUUSD).
Pipeline: MT5 historical data → train RL agent → backtest → deploy to MetaTrader 5.

The active system is **`rl_app.py`** (class `RLTradingStudio`) — a CustomTkinter GUI
that orchestrates the whole workflow through subprocess calls to CLI scripts.

> **Legacy note:** `app.py` (`TrainerApp`) is an older **PyCaret supervised** trainer.
> It is NOT part of the RL workflow. Ignore it unless explicitly asked.

### Two deployment paths (both valid)
1. **ONNX → MT5 EA** — `export_to_onnx.py` converts the PPO `.zip` to `.onnx`, embeds it
   in a generated `.mq5` Expert Advisor (via `#resource`), runs inside MT5 / Strategy Tester.
2. **Python live bot** — `live_trader.py` runs a loop using the MetaTrader5 Python API
   directly, logging trades to SQLite. No MQL5 compile needed.

---

## 2. Environment

| Item | Value |
|------|-------|
| Python interpreter | `.venv/Scripts/python.exe` (Windows) — **ALWAYS use this**, not bare `python` |
| Launch GUI | `run_rl_app.bat` (auto-creates venv + installs requirements) |
| Requirements | `requirements.txt` (stable-baselines3, gymnasium, torch, customtkinter, onnx, MetaTrader5) |
| Working dir | All scripts assume CWD = project root (`WORK_DIR = Path(__file__).parent`) |
| OS | Windows (paths, `.bat` launcher, MT5 integration) |

---

## 3. Key Commands

| Task | Command |
|------|---------|
| Launch GUI | `run_rl_app.bat` or `.venv/Scripts/python.exe rl_app.py` |
| Train PPO | `python rl_train.py <csv> --steps N --window 10 --name <model> [--eval_csv <csv>]` |
| Backtest (live logic) | `python backtest_live.py <model> <csv> --conf 0.85 --window 10 --mode pure_agent` |
| Backtest chart | `python backtest_chart.py <model> <csv> --limit 5000` |
| Walk-forward | `python rl_walkforward.py <csv> --windows 5 --steps 50000` |
| Export to ONNX | `python export_to_onnx.py <model> [--name <deploy>]` |
| Fine-tune | `python rl_finetune.py <base> --old_csv X --new_csv Y --steps 50000 --name <new>` |
| Relabel targets | `python relabel.py <csv> --mode quantile` |
| Feature engineering | `python feature_engineer.py <csv> [--target_tf D1]` |
| Pull MT5 data | `python pull_mt5_data.py --start 2010-01-01 --end 2020-12-31 --name <out>` |
| Quarterly cycle | `python quarterly_update.py [--auto-deploy] [--dry-run]` |
| Live trading | `python live_trader.py [--demo|--live|--paper]` |

---

## 4. File Map (RL system)

### Core engine
| File | Role |
|------|------|
| `rl_app.py` | **Main GUI** (`RLTradingStudio`, ~5000 lines). Pages: Train, Pipeline, Backtest, Walk-forward, Fine-tune, Analyze, Models, Tools, Settings. Drives everything via subprocess. |
| `trading_env.py` | Gymnasium env (`TradingEnv`). 4 actions, V4 reward, random episode start. The single source of truth for env behavior. |
| `rl_train.py` | PPO trainer CLI. Loads CSV → normalize → split → train → save `.zip` + `_norm.csv`. |
| `backtest_live.py` | Production-grade backtest (`run_backtest_live`, `SimAccount`). Matches `live_trader.py` logic exactly. Confidence filter + 3-layer risk. |
| `export_to_onnx.py` | PPO `.zip` → `.onnx` + `_config.mqh` + `_EA.mq5` (from template). `PolicyWrapper` adds softmax. |

### Validation & tuning
| File | Role |
|------|------|
| `rl_walkforward.py` | 5-window rolling train/test. Robustness gate (PF > 1.0 every window). |
| `rl_finetune.py` | Smart fine-tune: mix old(30%) + new(70%), lower LR (1e-4), ~50k steps. Prevents catastrophic forgetting. |
| `grid_search.py` | Sweep `conf × atr_sl × atr_tp` over `backtest_live.py`, report best PF. |
| `rl_backtest.py` | Quick env-based backtest + equity curve. |
| `rl_backtest_filtered.py` | Backtest that skips low-confidence trades (simulate selective entry). |
| `rl_analyze.py` / `analyze_confidence.py` | Confidence → accuracy analysis (is there a usable threshold?). |

### Data layer
| File | Role |
|------|------|
| `pull_mt5_data.py` | Pull bars from MT5 + compute features + label → CSV. |
| `mt5_connector.py` | MT5 API wrapper (`MT5Connector`): connect, get_rates, send_buy/sell, close_all. Default symbol `XAUUSDm`. |
| `features.py` | Python feature engine (`calc_features`), mirrors `DataCollector_v2.mq4`. Used by `live_trader.py`. |
| `feature_engineer.py` | Phase A features: multi-TF, volatility regime, range/trend → `<input>_enriched.csv`. |
| `relabel.py` | Re-label `future_return` → target (quantile / fixed / binary). |
| `add_lag_features.py` | Add lag features (memory) for supervised models. |
| `fix_csv_header.py` | Repair DataCollector CSV (header + BOM). |

### Deployment & orchestration
| File | Role |
|------|------|
| `live_trader.py` | Production Python live bot. MT5 API loop + SQLite (`live_trades.db`). Edit CONFIG block at top. |
| `quarterly_update.py` | Full cycle orchestrator: pull → features → fine-tune → walk-forward → decision gate → (optional) deploy → notify. |

### MQL5 (mt5_files/MQL5/)
| File | Role |
|------|------|
| `Indicators/CandlePatterns.mq5` | 10 candlestick patterns indicator (exposed via iCustom). |
| `Experts/DataCollector_v4.mq5` | Export bars + indicators + 10 candle features → CSV (67 cols). |
| `Experts/ML_RL_Trader_template.mq5` | **TEMPLATE** — placeholders filled by `export_to_onnx.py`. Do not hand-edit for a specific model. |
| `Include/RL_Indicators.mqh` | Feature library: 75-feature master list, dynamic feature mapping, iCustom auto-load. Shared by all models. |
| `Experts/*_EA.mq5`, `Include/*_config.mqh` | **Generated per model** by export script. |

---

## 5. Core Concepts

### Actions (Discrete(4) — `trading_env.py`)
`0 = Hold` · `1 = Buy` (open long) · `2 = Sell` (open short) · `3 = Close` (close position)

### V4 Reward (the "Honest" reward)
```
+ Net-PnL × 50         on close          (real $ dominates)
+ 0.01 bonus           if net pnl > 0.5% (meaningful win)
+ unrealized × 0.1                        (small direction hint)
- give-back penalty -0.001                (gave back half of peak)
- trade penalty     -0.005 on open        (discourage over-trading)
- hold idle penalty -0.0001 / bar         (while flat)
- time-decay        -0.0002               (held > 70% of max_hold)
→ Break-even win-rate ~53% (matches real-$ break-even)
```

### Confidence filter (the key to profitability)
Backtest & live only execute Buy/Sell when `confidence ≥ threshold` (default 0.85).
**Close is never filtered.** Raising the threshold turned V10 from PF 0.95 → PF 1.24.
`confidence = softmax(policy_logits)[argmax]`.

### Steps rule of thumb
`total_timesteps ≈ train_rows × 5–10`. Monitor `ep_rew_mean`: rising→keep going,
flat for 10–50k steps→stop, train PF≫test PF→overfit.

### Train/test discipline
- **Time-based split only** (no shuffle) — random split causes look-ahead bias.
- **Normalize using TRAIN stats only**, saved to `<model>_norm.csv`.
- Feature selection (correlation) must use TRAIN data only.

---

## 6. DO NOT TOUCH (generated / templates)

| Pattern | Why |
|---------|-----|
| `mt5_files/MQL5/Experts/*_template.mq5` | Template with placeholders; edited only by `export_to_onnx.py`. |
| `*_config.mqh` | Auto-generated per model (feature list + norm mean/std). Regenerate via export, never hand-edit. |
| `*_EA.mq5` (non-template) | Generated from template (gitignored per deployment). |
| `*.onnx`, `*.zip`, `*_norm.csv`, `*_logs/`, `*_best/` | Model artifacts (gitignored). |
| `*_live_bt_trades.csv`, `*_equity.png`, `*_chart.html` | Backtest outputs (gitignored). |
| `branding_config.json` | Per-user GUI branding override (gitignored). |
| `graphify-out/` | Knowledge-graph cache. |

---

## 7. Conventions

- **UTF-8 stdout**: every CLI script starts with
  `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")`.
  Subprocesses are launched with `env["PYTHONIOENCODING"]="utf-8"`. Keep this — Windows cp1252 breaks Thai/emoji.
- **Exit codes matter**: the GUI Pipeline checks `returncode`. On error, use `sys.exit(1)`
  (not bare `return`). On "no trades", still write an empty trades CSV so the chart stage succeeds.
- **argparse help**: escape literal `%` as `%%` or `--help` crashes.
- **GUI busy check**: use `self._is_process_busy()` (covers both `runner` and `pipeline_running`),
  not `self.runner.is_running()` alone.
- **Lazy page build**: GUI pages are built on first open (`_page_builders`). Widgets like
  `self.pipe_csv` do NOT exist until that page is opened — don't assume at init.
- **File listings are cached**: use `_list_csv_files()`, `_list_model_names()`,
  `_collect_model_rows()` (signature-based, mtime+size). Don't add raw `WORK_DIR.glob()` in hot paths.
- **Dropdowns**: `ScrollableOptionMenu` for file/model lists (searchable); `CTkOptionMenu` for short static lists.

---

## 8. Pipeline (rl_app.py "Pipeline" page)

Stages run as subprocess, each gated on exit code:
```
relabel (optional) → train (rl_train.py) → backtest (backtest_live.py) → chart (backtest_chart.py)
```
Live PPO metrics (6 health pills) are parsed from SB3 stdout:
`ep_rew_mean, approx_kl, clip_fraction, explained_variance, entropy_loss, value_loss`
(see `METRIC_INFO` + `classify_metric()` for good/warn/bad thresholds).

---

## 9. Gotchas (bugs we've actually hit)

| Symptom | Cause / Fix |
|---------|-------------|
| `--help` crashes with format error | unescaped `%` in argparse help → use `%%` |
| Pipeline fails after a bad model | backtest had 0 trades → must save empty CSV before return |
| Two processes write same files | page action skipped `pipeline_running` → use `_is_process_busy()` |
| relabel "succeeds" but no output | error path used `return` not `sys.exit(1)` |
| ONNX error 5019 in MT5 Tester | must `#resource`-embed ONNX, use `OnnxCreateFromBuffer` |
| ".onnx" splits into ".onnx.data" | consolidate with `save_model(save_as_external_data=False)` |
| EA "array out of range" | feature-count mismatch → rely on dynamic mapping in `RL_Indicators.mqh` |
| Thai/emoji garbled in logs | missing UTF-8 stdout wrapper |

---

## 10. Reference Documents

- `README.md` — project overview + architecture diagram.
- `PRODUCTION_README.md` — full 5-step production deployment guide (pull → train → WF → gate → deploy).
- `mt5_files/README_ONNX_Setup.md` — MT5 ONNX setup.
- `mt5_files/MQL5/Experts/README_DataCollector_v4.md` — data collector usage.
- `mt5_files/MQL5/Indicators/README_CandlePatterns.md` — candle pattern reference.
- `graphify-out/GRAPH_REPORT.md` — knowledge graph (god nodes, communities, gaps).

### Knowledge Base (quant theory)
- `reference/MIT-Quant-Bible.md` — MIT Sloan Quant Bible (converted from PDF, 51 pages).
  General quant-finance reference. Sections most relevant to this project:
  - **§2 Probability** (expected value, variance, covariance, correlation) → feature analysis
  - **§3 Stats** (LLN/CLT, confidence intervals) → confidence-threshold reasoning
  - **§4 Data Science** (least squares, regressions, dimensionality reduction) → feature engineering
  - §5–7 (market making, interview question banks) → background only, not project-specific.

  > **How to read it:** the file is large (~169 KB / 2,500 lines). Do NOT load the whole
  > file into context. Instead grep for the topic and read only that range:
  > - Page anchors: each page starts with `<!-- page N -->` (use the TOC on page 2 to map section → page).
  > - Section headers: search the numbered titles, e.g. `2.5 Covariance and Correlation`,
  >   `3.2 Confidence Intervals`, `4.4 Dimensionality Reduction`.
  > - Typical flow: `grep -n "Covariance"` → note the line → `Read` with `offset`/`limit` around it.

- `reference/ml4t/` — chapter READMEs from Stefan Jansen, *Machine Learning for
  Algorithmic Trading* (2nd ed., 2020). 24 chapter overviews + main book TOC +
  install guide. Start with `reference/ml4t/INDEX.md` for which chapter to read.
  Highest-relevance chapters for this project:
  - **22_deep_reinforcement_learning** ⭐ — RL for trading (our chapter)
  - **05_strategy_evaluation** — backtest pitfalls, Sharpe/PF/DD interpretation
  - **08_ml4t_workflow** — end-to-end pipeline structure
  - **09_time_series_models** — stationarity tests, regime drift, ARIMA
  - **20_autoencoders_for_conditional_risk_factors** — latent regime extraction
  - **24_alpha_factor_library** — pre-built feature ideas for `RL_Indicators.mqh`

  > **How to read it:** files are 80–520 lines each, grep across all to find a topic
  > then open the matching chapter:
  > `grep -rn -i "stationarity" reference/ml4t/`
  > Don't paste the whole folder into context — pick by topic.
