# AGENTS.md — Meta RL Trading Project

> Guide for AI agents (Claude Code, Codex, Cursor) working on this codebase.
> Last updated: 2026-07-06
> For a full dependency map see `graphify-out/GRAPH_REPORT.md`.
>
> **Repo was reorganized (2026-06/07):** supporting scripts moved under `tools/`,
> old PyCaret/MQL files under `legacy/`, generated model+backtest output under
> `artifacts/models/<name>/`, and docs under `docs/`. RL actions and rewards are
> now **configurable profiles** (see §4 "Action & Reward profiles" and §5). Paths
> below reflect the new layout.

---

## 1. What This Is

End-to-end **Reinforcement Learning (PPO)** trading system for forex/gold. Datasets in
active use: EUR/USD, GBP/USD (GU), USD/JPY (UJ), AUD/USD (AU) — mostly H4 — plus XAUUSD.
Pipeline: MT5 historical data → train RL agent → backtest → deploy to MetaTrader 5.

The active system is **`rl_app.py`** (class `RLTradingStudio`) — a CustomTkinter GUI
that orchestrates the whole workflow through subprocess calls to CLI scripts.

> **Legacy note:** `legacy/pycaret/app.py` (`TrainerApp`) is an older **PyCaret supervised**
> trainer. It is NOT part of the RL workflow. Everything under `legacy/` (old PyCaret
> scripts + old MQL collectors/EAs) is kept for reference only — ignore unless asked.

### Two deployment paths (both valid)
1. **ONNX → MT5 EA** — `export_to_onnx.py` converts the PPO `.zip` to `.onnx`, embeds it
   in a generated `.mq5` Expert Advisor (via `#resource`), runs inside MT5 / Strategy Tester.
2. **Python live bot** — `tools/mt5/live_trader.py` runs a loop using the MetaTrader5 Python
   API directly, logging trades to SQLite. No MQL5 compile needed.

---

## 2. Environment

| Item | Value |
|------|-------|
| Python interpreter | `.venv/Scripts/python.exe` (Windows) — **ALWAYS use this**, not bare `python` |
| Launch GUI | `run_rl_app.bat` (auto-creates venv + checks `requirements.txt` before launching) |
| Requirements | `requirements.txt` (RL stack, GUI, ONNX, regime detection, Gemini labeler, MT5 Python API, doc/slide generators) |
| Working dir | All scripts assume CWD = project root (`WORK_DIR = Path(__file__).parent`) |
| OS | Windows (paths, `.bat` launcher, MT5 integration) |

---

## 3. Key Commands

| Task | Command |
|------|---------|
| Launch GUI | `run_rl_app.bat` or `.venv/Scripts/python.exe rl_app.py` |
| Train PPO | `python rl_train.py <csv> --steps N --window 10 --name <model> [--reward_profile balanced] [--action_profile basic_4] [--mc_eval 1000 --mc_skip_frac 0.10] [--eval_csv <csv>]` |
| Backtest (live logic) | `python backtest_live.py <model> <csv> --conf 0 --window 10 --mode pure_agent [--intrabar {pessimistic,optimistic}] [--stop_slippage 0.0001] [--m1_csv <m1.csv>] [--swap_long -0.00005 --swap_short -0.00003] [--random_baseline 20] [--mc 1000]` |
| Backtest chart | `python backtest_chart.py <model> <csv> --limit 5000` |
| Walk-forward | `python rl_walkforward.py <csv> --windows 5 --steps 50000` |
| Export to ONNX | `python export_to_onnx.py <model> [--name <deploy>]` |
| Fine-tune | `python rl_finetune.py <base> --old_csv X --new_csv Y --steps 50000 --name <new>` |
| Relabel targets | `python relabel.py <csv> --mode quantile` |
| Feature engineering | `python tools/data/feature_engineer.py <csv> [--target_tf D1]` |
| Pull MT5 data | `python tools/mt5/pull_mt5_data.py --start 2010-01-01 --end 2020-12-31 --name <out>` |
| Quarterly cycle | `python tools/automation/quarterly_update.py [--auto-deploy] [--dry-run]` |
| Live trading | `python tools/mt5/live_trader.py [--demo|--live|--paper]` |
| Regime detection (single method) | `python regime_compare.py <csv> --method {hmm,kmeans,pelt} [--n-states N] [--k K] [--penalty P]` |
| Regime detection (compare 6 methods) | `python regime_compare.py <csv> --method all` |
| Auto-label price shocks with Gemini | `python gemini_labeler.py <csv> --symbol GBPUSD --top-k 15 --api-key $env:GEMINI_API_KEY` |

> Scripts under `tools/` use `Path(__file__)`-relative or repo-root-relative imports;
> run them from the repo root (CWD = project root) so `import action_profiles` etc. resolve.

---

## 4. File Map (RL system)

### Repo layout (post-reorg)
```
<root>/                RL engine + profiles kept at root (imported by tools/ and rl_app)
  action_profiles.py   reward_profiles.py  reward_formula.py  artifact_paths.py
  rl_train.py  trading_env.py  backtest_live.py  backtest_chart.py
  rl_walkforward.py  rl_finetune.py  export_to_onnx.py  relabel.py  rl_analyze.py
  regime_compare.py  gemini_labeler.py  build_training_from_collector.py
tools/
  analysis/    grid_search, rl_backtest, rl_backtest_filtered, analyze_confidence, compare_features
  automation/  quarterly_update, trigger_finetune
  data/        feature_engineer, fix_csv_header
  mt5/         mt5_connector, features, live_trader, pull_mt5_data
  generators/  pptx/docx slide+doc generators
legacy/
  pycaret/     app.py, backtest.py, predict.py, add_lag_features.py, signal_server.py, run.bat
  mql/         old MQL4/5 collectors + GA/RL bridge EAs
artifacts/models/<name>/   generated model + backtest output (see §6)
reward_profile_configs/    example reward-profile JSON presets
mt5_files/MQL5/            indicators, DataCollector_RL, EA template, RL_Indicators.mqh
docs/                      metafxclub studio guide/ (flow) + explainers/ (background)
reference/                 MIT-Quant-Bible.md, ml4t/  (quant theory)
```

### Core engine (root)
| File | Role |
|------|------|
| `rl_app.py` | **Main GUI** (`RLTradingStudio`, ~8000 lines). Pages: Train, Pipeline, Backtest, Walk-forward, Fine-tune, Analyze, **Regime Check**, Models, Tools, Settings. Drives everything via subprocess. Train page exposes **Reward Profile** + **Action Profile** pickers. |
| `trading_env.py` | Gymnasium env (`TradingEnv`). Action space + reward are **profile-driven**: `action_profile` (default `basic_4`) sets `spaces.Discrete(len(actions))`; `reward_profile` (default `balanced`) sets the reward weights. Imports `get_reward_profile` + `get_action_profile`. Single source of truth for env behavior. |
| `rl_train.py` | PPO trainer CLI. Loads CSV → time-sorted split → **train-only** z-score normalize → train → saves under `artifacts/models/<name>/` (`.zip` + `_norm.csv` + `.params.json` + `.train.json` run metadata) via `artifact_paths`. Accepts `--train_pct`, `--reward_profile`/`--reward_overrides`/`--reward_profile_json`/`--reward_formula`, `--action_profile`/`--action_params`/`--action_profile_json`. Forwards the input CSV's `.params.json` sidecar. |
| `backtest_live.py` | Production-grade backtest (`run_backtest_live`, `SimAccount`). Matches `tools/mt5/live_trader.py` logic. Confidence filter + 3-layer risk. Window auto-detected from `model.observation_space.shape`. Defaults: `--conf 0`, max 1 position. **Execution realism:** `--intrabar {pessimistic,optimistic}` (SL-first vs TP-first when both fall inside one bar), `--stop_slippage` (adverse fill on SL stops only), `--m1_csv` (`M1Resolver` replays M1 bars inside ambiguous main-TF bars to read the true SL/TP order; falls back to the assumption with per-run counters). See §5 "Execution realism". |
| `export_to_onnx.py` | PPO `.zip` → `.onnx` + `_config.mqh` + `_EA.mq5` (from template). Embeds `<model>.params.json` → emits `RL_ApplyDataCollectorConfig()` in the `.mqh` so the EA reproduces the collector's exact indicator periods. `PolicyWrapper` adds softmax. |

### Action & Reward profiles (root) — configurable RL behavior
| File | Role |
|------|------|
| `action_profiles.py` | Presets for **what the model may do**. `ACTION_PROFILES`: `basic_4` (Hold/Buy/Sell/Close, 4 actions) and `manage_6` (+ Break-Even / Trailing, 6 actions). Changing the profile changes the **model output dimension** → a model trained under one profile CANNOT be reused under another. Consumed by `trading_env`, `backtest_live`, and the EA export. JSON presets loadable via `load_action_profile_json` (schema `metafxclub.action_profile.v1`). |
| `reward_profiles.py` | Presets for **how the agent is rewarded during training** (does not affect execution after training). `REWARD_PROFILES`: `balanced` (the V4 honest reward — default), `anti_overtrade`, `low_drawdown`, `trend_follower`, `scalper`. Each is a weight dict (`close_pnl_mult`, `trade_penalty`, `giveback_*`, `time_decay_*`, …). Overridable per-run and via `reward_profile_configs/*.json`. |
| `reward_formula.py` | **Developer mode**: lets a reward profile carry a custom formula string, parsed by a safe AST evaluator (whitelisted vars + `abs/min/max/round/sqrt/log/exp/tanh/clip/sign`). `DEFAULT_REWARD_FORMULA` reproduces the balanced reward. `validate_reward_formula` rejects unsafe/unknown tokens. |
| `artifact_paths.py` | Central path helpers for `artifacts/models/<name>/` (`model_dir`, `backtests_dir`, `logs_dir`, `best_dir`, `final_model_path`, `ensure_model_dirs`). Legacy root-level artifacts stay discoverable so old runs still load. |
| `reward_profile_configs/` | Example JSON presets: `balanced_custom_1.json`, `anti_overtrade_example.json`, `developer_formula_example.json` (+ README). |

### Validation & tuning
| File | Role |
|------|------|
| `rl_walkforward.py` (root) | 5-window rolling train/test. Robustness gate (PF > 1.0 every window). Validates the training *recipe*, not a saved model file (retrains per window). Accepts the full Train recipe (reward mode/profile/overrides/formula, PPO hyperparams, max hold, net_arch) — GUI has a "⧉ Copy settings from Train" button so WF validates the SAME recipe you train with. |
| `rl_finetune.py` (root) | Smart fine-tune: mix old(30%) + new(70%), lower LR (1e-4), ~50k steps. Prevents catastrophic forgetting. |
| `rl_analyze.py` (root) | Confidence → accuracy analysis (is there a usable threshold?). |
| `tools/analysis/grid_search.py` | Sweep `conf × atr_sl × atr_tp` over `backtest_live.py`, report best PF. |
| `tools/analysis/rl_backtest.py` | Quick env-based backtest + equity curve. |
| `tools/analysis/rl_backtest_filtered.py` | Backtest that skips low-confidence trades (simulate selective entry). |
| `tools/analysis/analyze_confidence.py` | Older confidence-vs-accuracy analysis. |
| `tools/analysis/compare_features.py` | Diff MT5 live features vs training CSV per feature (parity debugging). |

### Regime detection & event labeling (new)
| File | Role |
|------|------|
| `regime_compare.py` | Six structural-break detection methods on a price CSV: **HMM 3-state**, **K-Means rolling** features, **PELT** (changepoint), **Bai-Perron**, **BinSeg**, **Rolling t-test**. CLI flags: `--method {hmm,kmeans,pelt,all}` plus per-method params. Writes `regime_single.html` (chart with breakpoints) and `regime_single_data.json` (used by the GUI table). `KNOWN_EVENTS` auto-loads from `known_events.json` if present, else falls back to 3 hardcoded events (Lehman 2008-09-15, Brexit 2016-06-23, Truss 2022-09-26). |
| `gemini_labeler.py` | **Auto-label price-shock events** so users don't have to maintain `KNOWN_EVENTS` by hand. `detect_shocks()` picks the top-K shock dates by z-score of `\|return\|` over a rolling 60-day baseline; selection iterates *descending z-score* (not by date — that was an early bug that filled the K-slot quota from the earliest years and silently dropped Brexit). `label_one()` then asks **Google Gemini** (gemini-2.5-flash, free tier) what event each date corresponds to. Output is written to `known_events.json` with `last_updated`, `symbol`, `source_csv`, and one entry per event with `event/category/confidence/rationale/source`. Falls back to `Shock @ YYYY-MM` placeholder labels if no API key is set. |
| `known_events.json` (gitignored) | Generated event list. Drop-in replacement for `regime_compare.KNOWN_EVENTS`. Refresh by running `gemini_labeler.py` or the Regime Check page "Refresh events with Gemini" button. |
| `api_keys.json` (gitignored) | User's Gemini API key. Loaded by `_load_api_keys()` in `rl_app.py`. Edit via the Settings page → "API Keys" card. **Never** commit this file. |

### Data layer
| File | Role |
|------|------|
| `build_training_from_collector.py` (root) | Collector CSV → training CSV (adds/forwards the feature set + `.params.json`). First step after importing a DataCollector_RL dump. |
| `relabel.py` (root) | Re-label `future_return` → target (quantile / fixed / binary). Legacy supervised concept; RL does not need targets. |
| `tools/mt5/pull_mt5_data.py` | Pull bars from MT5 + compute features + label → CSV. |
| `tools/mt5/mt5_connector.py` | MT5 API wrapper (`MT5Connector`): connect, get_rates, send_buy/sell, close_all. Default symbol `XAUUSDm`. |
| `tools/mt5/features.py` | Python feature engine (`calc_features`), mirrors the MQL collector. Used by `tools/mt5/live_trader.py`. |
| `tools/data/feature_engineer.py` | Phase A features: multi-TF, volatility regime, range/trend → `<input>_enriched.csv`. |
| `tools/data/fix_csv_header.py` | Repair DataCollector CSV (header + BOM). |
| `legacy/pycaret/add_lag_features.py` | Add lag features (memory) for the old supervised models. |

### Deployment & orchestration
| File | Role |
|------|------|
| `tools/mt5/live_trader.py` | Production Python live bot. MT5 API loop + SQLite (`live_trades.db`). Edit CONFIG block at top. |
| `tools/automation/quarterly_update.py` | Full cycle orchestrator: pull → features → fine-tune → walk-forward → decision gate → (optional) deploy → notify. |
| `tools/automation/trigger_finetune.py` | Trigger-based fine-tune orchestrator (checks data-drift / schedule, kicks off `rl_finetune`). |

### MQL5 (mt5_files/MQL5/)
| File | Role |
|------|------|
| `Indicators/CandlePatterns.mq5` | 10 candlestick patterns indicator (exposed via iCustom). |
| `Experts/DataCollector_RL.mq5` | Export bars + current RL feature stack -> CSV + `.params` sidecar for train/deploy parity. With `InpCollectM1=true` (default) also dumps `<out>_m1.csv` (raw M1 OHLCV over the collected period, chunked `CopyRates`) for `backtest_live.py --m1_csv`; partial M1 coverage is logged, not fatal. **Never put `_m1` in `InpOutFile` yourself** — the suffix is added automatically; a hand-typed `_m1` name makes the GUI treat the feature dataset as M1 data (seen in the field). Data Tools Import has an "Import M1" checkbox (default on), and `_find_matching_m1` walks name ancestry so derived datasets (`_train/_test/_relabeled/_clean/_from_<date>/training_data_`) inherit the original M1 file — M1Resolver slices by timestamp, so nothing is copied. |
| `Experts/ML_RL_Trader_template.mq5` | **TEMPLATE** — placeholders filled by `export_to_onnx.py`. Required for MT5 EA export; do not hand-edit for a specific model. |
| `Include/RL_Indicators.mqh` | Feature library: 75-feature master list, dynamic feature mapping, iCustom auto-load. Shared by all models. |
| `Experts/*_EA.mq5`, `Include/*_config.mqh` | **Generated per model** by export script. |

---

## 5. Core Concepts

### Actions (profile-driven — `action_profiles.py`)
The action space is chosen by an **action profile**, not hardcoded:
- `basic_4` (default): `0 = Hold` · `1 = Buy` · `2 = Sell` · `3 = Close`
- `manage_6`: adds Break-Even and Trailing-stop management actions (6 total)

`trading_env` builds `spaces.Discrete(len(profile["actions"]))`. **Switching profile
changes the model's output dimension** — a model trained on `basic_4` cannot load under
`manage_6` and vice-versa. The profile must match across train → backtest → export → EA.

### Reward profiles (`reward_profiles.py`) — default `balanced` = the V4 "Honest" reward
```
balanced (default V4):
+ Net-PnL × 50         on close          (real $ dominates)
+ 0.01 bonus           if net pnl > 0.5% (meaningful win)
+ unrealized × 0.1                        (small direction hint)
- give-back penalty -0.001                (gave back half of peak)
- trade penalty     -0.005 on open        (discourage over-trading)
- hold idle penalty -0.0001 / bar         (while flat)
- time-decay        -0.0002               (held > 70% of max_hold)
→ Break-even win-rate ~53% (matches real-$ break-even)

Other presets: anti_overtrade, low_drawdown, trend_follower, scalper — each re-weights
the same terms. Developer mode (reward_formula.py) allows a custom safe-expression formula.
The reward profile only affects TRAINING; it does not change backtest/live execution.
```

### Confidence filter (execution gate, not train logic)
Backtest & live only execute Buy/Sell when `confidence ≥ threshold`.
Backtest defaults to `--conf 0` so the first validation run matches train/quick eval as closely as possible.
**Close is never filtered.** Sweep thresholds such as 0.3/0.5/0.7/0.85/0.9 on OOS/WF before using a live-style gate.
`confidence = softmax(policy_logits)[argmax]`.

### Execution realism (intrabar SL/TP ambiguity)
Bar-level OHLC cannot tell which of SL/TP was hit first when both fall inside one
bar's range — a real error source on H4 when SL/TP distances are small vs bar range.
`backtest_live.py` handles it in two layers:
- **L1 bracket:** `--intrabar pessimistic` (SL first, default) vs `optimistic` (TP first)
  bound the true result from both sides; `--stop_slippage` adds adverse fill on SL stops.
  The results block reports `Decided by assumption` share and warns above 10%.
- **L2 M1 replay:** `--m1_csv <file>` makes `M1Resolver` walk the M1 bars inside each
  ambiguous main bar and read the actual order. Unresolvable cases (no M1 coverage,
  both levels in one minute) fall back to the assumption and are counted separately.
  Measured on `rl_uj_h4` @ 0.3/0.3 ATR: assumption share 27.4% → 0.1%, result lands
  between the two brackets (WR 35→49.3→61.5%).
- Trades CSV column `ambiguous` = True only for assumption-decided exits.
- MT5 Strategy Tester (real ticks) stays the final pre-live gate — L1/L2 are for
  fast Python iteration.

Three statistical-validation layers on top (all in `backtest_live.py`):
- **Overnight swap:** `--swap_long`/`--swap_short` (price fraction per rollover,
  negative = cost; Wednesday rollover charges 3x for T+2 weekend). H4 positions
  held to `max_hold=30` cross ~5 nights — a real cost most backtests skip.
  Per-trade `swap_pct` lands in the trades CSV; totals in the report + meta.
- **Random-agent baseline:** `--random_baseline N` reruns the SAME engine
  (`_simulate()` with a random policy, no NN inference) N times. If the model
  doesn't beat ≥95% of random agents, the result comes from the risk rules,
  not model edge. Verdict + percentile in report + meta.
- **Monte Carlo DD:** `--mc N` (default 1000, needs ≥10 trades) shuffles trade
  order → distribution of max drawdown + P(hitting the hard stop). Final
  balance is order-independent (product commutes), so only DD is reported.
  Answers sizing/survival risk, not strategy quality.

MC also runs at **train time**: `rl_train.py --mc_eval N` (default 1000, 0=off)
+ `--mc_skip_frac` (default 0.10) probe the quick-eval trade list — shuffle-DD
plus SQX-style skip-retention (drop N% of trades → profit retention + flip
rate + verdict). Configurable from the Train page ("MC Eval Runs" / "MC Skip
%"); results persist in `<name>.train.json` under `quick_eval_mc`. Train-MC
runs on idealized TradingEnv trades (no costs/SL-TP) → use it for recipe
screening only; the backtest MC is the one that sizes real capital.
The Pipeline's backtest stage passes `--random_baseline 20` and auto-attaches
`--m1_csv` when a companion M1 file exists, matching the Backtest page
defaults. GUI surfaces: Backtest page renders a full report panel (equity PNG
+ settings/realism/baseline/MC/significance/analytics) from the meta JSON, and
the Models page has a **Realism** column (`M1 ✓` / `ok` / `guess N%` / `-`)
showing how each stored result resolved intrabar exits.

### Steps rule of thumb
`total_timesteps ≈ train_rows × 5–10`. Monitor `ep_rew_mean`: rising→keep going,
flat for 10–50k steps→stop, train PF≫test PF→overfit.

### Train/test discipline
- **Time-based split only** (no shuffle) — random split causes look-ahead bias.
- **Normalize using TRAIN stats only**, saved to `<model>_norm.csv`.
- Feature selection (correlation) must use TRAIN data only.

---

## 6. DO NOT TOUCH (generated / templates)

Generated model + backtest output now lives under **`artifacts/models/<name>/`**:
```
artifacts/models/<name>/
  <name>.zip  <name>_norm.csv  <name>.params.json  <name>.train.json
  backtests/  (<name>_backtest_chart.html, _live_bt_equity.png, _live_bt_trades.csv, _live_bt.meta.json)
  logs/       (tensorboard events, evaluations.npz)
  best/       (best_model.zip)
```
A few sample models (`rl_uj_h4`, `rl_au_h4`, `rl_uj_extra`) are committed so the Backtest
page works out-of-the-box; everything else here is gitignored.

| Pattern | Why |
|---------|-----|
| `mt5_files/MQL5/Experts/*_template.mq5` | Template with placeholders; edited only by `export_to_onnx.py`. |
| `mt5_files/MQL5/**/*.ex5` | Compiled MQL5 binaries — rebuilt from the `.mq5`/`.mqh` sources in MetaEditor, don't hand-edit. |
| `*_config.mqh` | Auto-generated per model (feature list + norm mean/std). Regenerate via export, never hand-edit. |
| `*_EA.mq5` (non-template) | Generated from template (gitignored per deployment). |
| `artifacts/models/<name>/` | Generated model + backtest output. Managed by `artifact_paths` — don't hand-edit (except the curated samples). |
| `*.onnx`, `*.zip`, `*_norm.csv`, `logs/`, `best/` | Model artifacts (mostly gitignored). |
| `branding_config.json`, `api_keys.json`, `known_events.json` | Per-user runtime files (gitignored). |
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

## 8b. Regime Check page workflow

End-to-end loop the GUI supports for fixing regime-drift in training data:

```
Browse CSV
  ↓
[Refresh events with Gemini]  (gemini_labeler.py subprocess → known_events.json)
  ↓
Pick method (HMM ⭐ / K-Means / PELT) + params → [Run Detection]
  ↓
regime_compare.py subprocess → regime_single_data.json + regime_single.html
  ↓
GUI auto-populates the breakpoints table from regime_single_data.json
on subprocess success (see _handle_done → _load_regime_results).
Each row shows: date, nearest known event (✓ name / —), days-apart.
  ↓
User selects a breakpoint row → [Use Selected as Train Cutoff]
  ↓
_use_regime_cutoff() filters the CSV (rows where timestamp >= cutoff),
saves as <basename>_from_<YYYY-MM-DD>.csv, copies the .params.json
sidecar to match, then switches to the Train page with the new file
pre-selected via _set_train_csv().
  ↓
User trains PPO on the regime-aligned subset → eval reward improves
(empirically: GBPUSD H4 full-history -38.7% return → post-Brexit cut
should align train and test distributions).
```

Why this exists: the surrounding research showed eval reward kept getting
worse (`-26.9` then `-38.7%`) because the GBPUSD price distribution shifted
~18% between training years (2004–2022) and the test slice (2022–2026).
HMM identified Brexit-2016 and COVID-2020 as the breakpoints; retraining on
the post-Brexit subset removes the distribution shift.

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
| Train page silently overwrites a prior model | `_start_training` now pops an "Overwrite model?" confirm dialog when `<name>.zip` already exists (mirrors what the Pipeline page always did). |
| Train page Dataset card shows ⚠ no .params.json | The chosen CSV has no sidecar in its folder. Use the "Attach .params" button to copy one from anywhere — `_attach_train_params` validates the JSON and copies it to `<csv_basename>.params.json`. |
| Regime page table empty / score 2/3 instead of N/15 | The summary used to hardcode `/3` (the old hardcoded event count). Fixed to use `n_events = len(event_dates)` from `regime_single_data.json`. |
| Gemini run misses Brexit despite z=7.0 being highest | `detect_shocks` used to take top-K*5 candidates, **sort by date**, then iterate front-to-back — once early-year clustered shocks filled K slots, later high-z events were dropped. Now iterates *descending z-score* with a global min-gap check. |
| Gemini API rate-limit errors | Free-tier limit is 15 req/min. `gemini_labeler.py` sleeps `RATE_LIMIT_SLEEP = 4.5s` between calls so a 15-event refresh fits under the limit. |
| Looking up Gemini key in code | Never hardcode. Stored in gitignored `api_keys.json`, accessed via `_load_api_keys()`. The Settings page UI saves/loads it; the masked entry shows `Show` toggle for visibility. |
| Model "size mismatch" / obs or action-dim error on load | The saved model's **action profile** or **window/feature count** differs from the current run. An action profile change (`basic_4`↔`manage_6`) changes the output dim; you must retrain, not reload. Keep the profile consistent across train → backtest → export → EA. |
| Reward change had no effect on backtest | Reward profiles only shape **training**. Backtest/live execution is unaffected — you must retrain to see a reward-profile change. |
| Import error running a `tools/` script | Run from the repo root so root-level imports (`action_profiles`, `artifact_paths`, …) resolve. `cd` into `tools/…` and it breaks. |
| Log/metrics show up on the wrong GUI page | Fixed (`8925489`): the process runner now routes stdout/`done` to the **page that started the run** (owner page), not just `current_page`. |
| Old model artifacts not found after reorg | `artifact_paths` still checks legacy root-level paths as a fallback, so pre-reorg `.zip`/`_norm.csv` at the repo root remain loadable. New runs write under `artifacts/models/<name>/`. |
| Backtest looks great but SL/TP are tight vs bar range | Check the `Execution realism` block: if `Decided by assumption` >10%, the PF/WR partly reflect the intrabar guess, not the market. Bracket with `--intrabar optimistic`, or add `--m1_csv` to resolve with data. Tight-SL results without M1 are not trustworthy. |
| M1 replay silently resolves nothing | Feeds mismatch: M1 must come from the same broker/source as the main CSV (same server-time timestamps). If neither level is touched in the M1 slice, the resolver counts it as `no/mismatched M1 data` and falls back — check that counter in the results block. |
| Reward overrides (sliders) had no effect on training | Fixed 2026-07-11: `rl_train` resolved `--reward_overrides` but only wrote them to meta — the env re-resolved the base profile without them. `TradingEnv` now takes `reward_overrides=` and train/walk-forward pass it through. **Models trained before this fix with non-default sliders actually used base-profile params** (their meta overstates the overrides). `rl_finetune` still has the old behavior (reads `reward_profile_config` from meta but never applies it). |
| Walk-Forward verdict didn't match your recipe | Pre-fix, `rl_walkforward.py` hardcoded reward (`realized`/balanced) and PPO hyperparams. Now it accepts the full Train recipe; use the GUI's "⧉ Copy settings from Train" button before running so WF validates the same recipe. |
| MT5 compile: 25× "undeclared identifier" (`SESSION_ASIA_START`, `EX_TEMA_PERIOD`, …) in `<model>_config.mqh` | Version mismatch: the generated config assigns globals that only exist in the **current** `RL_Indicators.mqh` (extra-features update, 2026-07). The terminal being compiled still has a pre-update copy in `MQL5/Include/`. Fix: copy the bundled `packages/<name>/MQL5/Include/RL_Indicators.mqh` (or `mt5_files/MQL5/Include/RL_Indicators.mqh`) into that terminal's Include folder — always install BOTH files from a package, never just the config. |

---

## 10. Reference Documents

- `README.md` — project overview, install, Codex workflow, sample models.
- `docs/metafxclub studio guide/` — **current user flow guide** (start at `00_reading_order.md`):
  dashboard flow → data prep → quality/regime check → train → backtest → walk-forward →
  custom feature guide → project folder map. HTML deep-dives, one per dashboard step.
- `docs/codex_git_update_prompts.md` — safe "pull + reinstall deps" prompt for Codex.
- `docs/codex_custom_action_prompts.md` — prompt for asking Codex to add/change an Action
  Profile across train/backtest/fine-tune/walk-forward/export/EA consistently.
- `docs/explainers/11_production_readme.md` — full 5-step production deployment guide.
- `docs/metafxclub studio guide/12_monte_carlo_deep_dive.html` — what the MC process is, the two probes (shuffle-DD, skip-retention), and the Train-MC vs Backtest-MC comparison (idealized env trades vs full-cost SimAccount trades; screening vs capital sizing).
- `docs/metafxclub studio guide/13_regime_check_deep_dive.html` — regime detection algorithms (HMM/K-Means/PELT), per-method parameter tables (each method takes a DIFFERENT param set: States vs k+Window vs Penalty), fixed in-code constants (daily resample, 25-of-30-day persistence, ±90d match tolerance), tuning recipes and limits.
- `mt5_files/README_ONNX_Setup.md` — MT5 ONNX setup.
- `mt5_files/MQL5/Indicators/README_CandlePatterns.md` — candle pattern reference.
- `graphify-out/GRAPH_REPORT.md` — knowledge graph (god nodes, communities, gaps). Regenerate with `/graphify` or `/graphify . --update`.
- `graphify-out/graph.html` — interactive Pyvis visualization. Open in any browser, no server needed.

### Background explainers (`docs/explainers/`, open in browser)
- `07_rl_reward_explained.html` — the reward terms + how reward profiles re-weight them.
- `04_data_tools_modules_explained.html` — the Data Tools page modules.
- `06_parity_config_explained.html` — why `.params.json` sidecars exist and how `RL_ApplyDataCollectorConfig` works.
- `05_data_collector_v4_explained.html` — legacy v4 collector; current workflow uses `DataCollector_RL.mq5`.
- `13_class_imbalance_explained.html` — UP/DOWN/FLAT class-balance handling in `relabel.py` (legacy supervised).
- `regime_single.html` (root) — single-method regime result, regenerated on each Regime Check run.

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
