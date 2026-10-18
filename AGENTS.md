# AGENTS.md — Meta RL Trading Project

> Guide for AI agents (Claude Code, Codex, Cursor) working on this codebase.
> Last updated: 2026-10-18
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
| Launch web GUI (preview, migration in progress) | `run_rl_web.bat` or `.venv/Scripts/python.exe rl_web.py [--debug]` — native pywebview window, Clear Liquid theme. all 10 pages (phases 0–5 of `docs/web_gui_migration_plan_2026-10-18.md`); the classic app stays available until phase 6 swaps the launcher. Settings has a Clear Liquid / Deep Glass theme switch (`web_settings.json`, gitignored). |
| Train PPO | `python rl_train.py <csv> --steps N --window 10 --name <model> [--seed N] [--corr_threshold 0.95] [--reward_profile balanced] [--action_profile basic_4] [--mc_eval 1000 --mc_skip_frac 0.10] [--eval_csv <csv>]` |
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
| Verify a freshly collected dataset | `python tools/data/verify_collected_features.py <csv>` — constant columns, divergence parity vs `tools/data/divergence_features.py`, nearness windows, candle mapping. Run before training on any new DataCollector_RL dump. |
| Live 3D training visual (neon network + candlestick replay) | Train page: tick **Live 3D visual** (adds `rl_train.py --viz`) → after every evaluation (10k steps) `tools/viz/snapshot.py` writes `<run>/viz/snap_NNN.json` + `index.json` and prints `VIZ_SNAPSHOT <i> <step> <path>`; the GUI opens `tools/viz/live_viewer.py` (pywebview window, private 127.0.0.1 server) which follows new snapshots and has a timeline + eval-reward sparkline. **Training rollouts (default source while training):** after every PPO rollout (>= 4 s apart, ~71 ms, ~1% overhead) `make_rollout_callback` writes `<run>/viz/rollout_latest.json` + `rollout_meta.json` from the rollout buffer — the SAMPLED (exploring) actions on TRAIN bars, env rewards, GAE advantages, greedy argmax, activations from one batched forward pass — and prints `VIZ_ROLLOUT <seq> <step> <path>` (first + every 10th; the GUI opens the window on the first one, ~2k steps). Viewer `source` menu: Training (live) / Eval replay; training view adds explored-action dots (sampled != greedy), an advantage strip and REWARD/ADVANTAGE/SAMPLED·GREEDY tiles; a new rollout waits for the current replay to end. Dashboard stays eval-based. Details: `docs/viz_training_rollout_2026-10-11.md`. **Monitor mode** (2026-10-18, `tools/viz/monitor.js`, `--mode monitor`, menu 'Monitor (verdicts)'): the answer to 'stop or keep going?' — after EVERY rollout `make_rollout_callback` appends a row to `<run>/viz/monitor.json` (entries per 100 bars, mean hold, exit reasons, action mix + entropy, trade PF/pnl in bp, and on a gated run: signal bars, acceptance, off-gate attempts, plus a fixed-rule counterfactual = enter on every signal bar, exit after 4 bars, no stop, split into the signal bars the agent ACCEPTED vs SKIPPED). The panel shows verdict chips with the numbers and thresholds they rest on (filters the signal / over-trading / exits by timeout / selection helps-hurts = accepted − skipped / exit timing adds-loses = agent − accepted / exploration collapsed / train PF), pooled over the last 20 rollouts, plus sparklines per rollout and the eval table. Event-driven like Dashboard (~1% CPU). Selection and exit-timing verdicts need ≥ 30 accepted and ≥ 30 skipped signal bars — early verdicts flip (seen: 'helps' at 6 rollouts → 'random' at 13). **Monitor is the GUI default mode.** Two timeline cards overlay **sibling seed runs** (`live_viewer.find_peers`: models named `<stem>_s<seed>` next to this one, newest run with a viz folder; served as `/peers.json` + `/peer/<k>/…`): per-evaluation behaviour (eval reward, equity end, trades, win rate, hold, acceptance — `snapshot.summary()` now writes `entries/bars/hold_mean/pnl_bp_mean` and, on gated runs, `signal_bars/accepted/acceptance`; eval captures record the gate per frame) and per-rollout training behaviour (5-rollout means). Runs trained before 2026-10-18 have no monitor rows and older index entries lack the behaviour fields. **Attribution (Level 3, 2026-10-18):** every snapshot carries `attr` = gradient × input of the chosen action's logit per frame (int8, per-frame scaled, `attr_scale`), `input_names` (`feature@-lag`, then position / unrealized_pnl / bars_in_position), `attr_feature` / `attr_lag` shares and `attr_top`; `snapshot.attribution()` is one batched forward/backward. 3D view: 'inputs' menu (attribution default) colours the input layer by what pushed THIS decision and the TOP DRIVERS panel lists the strongest inputs per bar; Monitor: 'WHAT DRIVES THE DECISIONS' (mean top features over the last 20 rollouts) and 'WHICH BARS OF THE WINDOW MATTER' (share per lag — flat ≈ the window length is not used, as seen on a 16k-step near-random policy: ~1/15 per lag). index.json entries carry `top_features`. Linear sanity test: attribution == W[a] · x. **Theme (2026-10-18):** the panel modes (Monitor, Dashboard) use the 'Clear Liquid' glass theme — light aurora background, white translucent cards (`body.clear.mon/.dash` rules in `network3d.html` + `body.clear #mon` rules in `monitor.js`); chart palettes are read from the `--c-*` CSS variables at render time (`themePalette()` in monitor.js / dashboard.js) so they follow the theme. The 3D modes keep the dark scene but their HUD (stats, drivers, control/timeline panels, banner) is 'deep glass' (frosted blue, specular line); `?hud=flat` disables it (measured Lite: glass 21.9% vs flat 26.2% of a core — no cost). `?theme=dark` keeps the old dark panels. Design candidates: `tools/viz/design_demo.html` (A deep glass / B clear liquid / C obsidian mono). Display modes: **Lite** (default while training, ~20–40% of one core depending on machine state: no bloom, 4 synapses, no pulses, 1x pixels, 20 fps), **Cinematic** (~60%+: bloom, pulses, 30 fps) and **Dashboard** (`tools/viz/dashboard.js`, ~1–2%: KPIs, learning curve, equity of every eval, action mix, replay chart, neuron heatmaps, last trades — redrawn only when a snapshot arrives; reads the per-eval `snapshot.summary()` fields in `index.json`) — menu next to the checkbox (`live_viewer.py --mode`) or in the window. **Auto idle**: a live window plays each snapshot once, then stops rendering (~1% CPU) until the next evaluation or mouse input; it also sleeps when minimised (pywebview events) or paused 8 s. Measurements: `docs/viz_display_modes_2026-10-11.md`. **◉ Open Visual** reopens it, or captures a one-off snapshot for models trained without `--viz`. CLI: `python tools/viz/live_viewer.py --model <name> [--browser]`, `python tools/viz/capture_snapshot.py <model> [--slice test]` → `tools/viz/out/<model>_network3d.html` (open with `live_viewer.py --html out/<file>`). PPO/A2C only. Drawn connections are each neuron's top-K by |weight| x std(source) over the captured bars (raw |weight| over-drew the env's unnormalized unrealized-pnl / bars-in-position inputs, which carry large weights but contribute ~nothing). `--viz`/`--viz_frames` are excluded from the recipe hash. Validation equity is a visual, not edge evidence. |
| Screen features BEFORE adding them to RL | `python tools/analysis/feature_screen.py <csv> [--drop <regex>]` → `docs/feature_screen_<stem>.md/.csv`: whole-slice Spearman IC vs forward return (h=1/5/20) on Train/Val/Test, Train z from moving-block bootstrap, drift filter for non-stationary features, ridge on all features. Never rank-correlate inside short windows (biased negative for persistent features — see `docs/feature_screen_2026-10-10.md`). |
| Multi-seed A/B (train + diagnose + Test backtest + bootstrap) | `python tools/analysis/obs_ablation.py --arms D --seeds 0,1,2 [--train_start 2019-01-01] --prefix <p> --tag <t>` → `docs/ablation_<t>.md` (per-seed rows + across-seed pooled PF). `--train_start` trims Train only; Validation/Test rows stay those of the untrimmed split. |
| Dissect a profitable EA before adding its logic as RL features | `python tools/analysis/mt5_report_signals.py <tester report>.html --out docs/<name>` parses an MT5 Strategy Tester HTML report (UTF-16) into deals/orders CSV + a per-signal table (entry, original SL/TP from the Orders table, R, per-leg result in R). `tools/data/srb_features.py` is the Python twin of the Metafxclub SRB v02 breakout EA (H4 swing S/R zones, H1 buffered break + strong candle + EMA bias; `apply_state` adds the day-flag / zone-cooldown / one-position gates) - 1107/1107 report entries reproduced on USDJPY_TDS. `python tools/analysis/srb_event_edge.py usdjpy_tds_h1.csv usdjpy_tds_h4.csv --report_signals docs/srb_rf1_report_signals.csv [--cost_pips 1]` is the CONDITIONAL edge test (forward return after the event vs. three nulls, simple exits in R, cost sensitivity, within-event feature ranks) -> `docs/srb_event_edge_<date>.md`. Event features fire on ~2% of bars, so `feature_screen.py` (unconditional IC) cannot see them - always test events conditionally. Bars for the twin come from the RoboForex MT5 terminal's custom symbol `USDJPY_TDS` via the MetaTrader5 API (`copy_rates_range` in yearly chunks - one 12-year H1 call fails with 'Invalid params'). |
| Gated entries / RL on top of an EA signal | `rl_train.py <csv> --entry_gate <col>`: a buy opens only where the RAW column is > 0, a sell only where it is < 0 (off-gate buys/sells act as hold); recorded in the train meta and auto-applied by `backtest_live.py` (`--entry_gate` overrides; skipped entries are counted). `obs_ablation.py --entry_gate srb_sig [--sl_mode continuous --sl_range 0.5,3.0]` passes it through. Result 2026-10-13 (`docs/srb_rl_gated_2026-10-13.md`): SRB columns as plain features were ignored by PPO (PF ~1.0), gated entries reach PF 1.7–2.4 per seed on Test (pooled 2.09, 189 trades) with the value in RL-timed exits; signal selection skill is inconsistent across seeds. `tools/analysis/srb_gate_diag.py <prefix...>` shows accepted vs skipped signals under the fixed 4-bar rule. |
| Continuous stop-loss head (hybrid PPO policy) | `rl_train.py ... --sl_mode continuous --sl_range 0.5,3.0 [--sl_log_std_init -0.7 --sl_ent_scale 0.1 --atr_period 14]` (PPO only): `hybrid_policy.HybridPolicy` = Categorical head for the trading action + Gaussian head for the stop distance in MT5-style ATR; action is a Box(2) `[index, sl_atr]`, log pi = cat + gauss, deterministic = (argmax, mu). `TradingEnv(sl_mode='continuous', atr_col=...)` sets the stop at entry from the RAW `atr__raw` column and fills it intrabar; `backtest_live` reads the head (`[sl] model stop-loss head`, `sl_atr` in the trades CSV, random baseline draws a stop in range). Use `hybrid_policy.n_discrete_actions(space)` instead of `action_space.n`. ONNX/EA export and the GUI do NOT support the hybrid head yet (`export_to_onnx` refuses it). Measured 2026-10-13: the head learned a near-constant 1.6–1.7 ATR from ~770 events and lowered PF vs. no stop (1.35 vs 2.09 pooled) — a fixed EA stop reproduces it. |
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
| `trading_env.py` | Gymnasium env (`TradingEnv`). Action space + reward are **profile-driven**: `action_profile` (default `basic_4`) sets `spaces.Discrete(len(actions))`; `reward_profile` (default `balanced`) sets the reward weights. Imports `get_reward_profile` + `get_action_profile`. Single source of truth for env behavior. Options: `spread_pct`/`commission` (cost model; the old hard-wired defaults = ~7.5 pips round-trip on USDJPY), `entry_gate` (gated entries), `sl_mode='continuous'` + `atr_col` (hybrid SL head, Box(2) action). |
| `hybrid_policy.py` | SB3 `ActorCriticPolicy` with a Categorical + Gaussian action head (`HybridPolicy`, `HybridDistribution`), `hybrid_action_space`, `split_action`, `n_discrete_actions`, `is_hybrid_space`. Used by `rl_train --sl_mode continuous`; read by `backtest_live` through `policy.heads(obs)` → (action probs, SL mean in ATR, value). |
| `rl_train.py` | PPO trainer CLI. Loads CSV → time-sorted split → **train-only** z-score normalize → train → saves under `artifacts/models/<name>/` (`.zip` + `_norm.csv` + `.params.json` + `.train.json` run metadata) via `artifact_paths`. Accepts `--train_pct`, `--reward_profile`/`--reward_overrides`/`--reward_profile_json`/`--reward_formula`, `--action_profile`/`--action_params`/`--action_profile_json`. Forwards the input CSV's `.params.json` sidecar. |
| `backtest_live.py` | Production-grade backtest (`run_backtest_live`, `SimAccount`). Matches `tools/mt5/live_trader.py` logic. Confidence filter + 3-layer risk. Window auto-detected from `model.observation_space.shape`; omitted `--max_hold` auto-resolves from the selected model's train metadata (fallback 30), and the GUI passes that resolved value explicitly. Defaults: `--conf 0`, max 1 position. **Execution realism:** `--intrabar {pessimistic,optimistic}` (SL-first vs TP-first when both fall inside one bar), `--stop_slippage` (adverse fill on SL stops only), `--m1_csv` (`M1Resolver` replays M1 bars inside ambiguous main-TF bars to read the true SL/TP order; falls back to the assumption with per-run counters). See §5 "Execution realism". |
| `rl_web.py` + `webgui/` | **Web GUI (pywebview)** replacing rl_app.py page by page. `webgui/commands.py` = Train form → `rl_train.py` command with the CTk rules (FormError names the field; `test_ui_training_contract.py::test_web_command_equals_ctk_*` asserts the command equals the CTk one for the same inputs); `webgui/logparse.py` = stdout → events (progress, metric health, VIZ lines); `webgui/runner.py` = one subprocess, ProcessRunner semantics; `webgui/api.py` = `Api` (one method per user action, returns dicts) + `JsBridge` (what the window exposes: ASCII JSON, inf/NaN → null) + `to_json`; `webgui/ui/` = index.html / app.js / theme.css (page only paints; no rules in JS). Review → confirm keeps the CTk two-step (`prepare_training` + `--expected_recipe_sha256`). Extra web-only Train fields: entry gate, costs, seed (appended only when non-default). |
| `model_results.py` | Models-table rows shared by rl_app and the web GUI (`collect_model_rows`, `metrics_from_trades`, `table_values` = the 10 display cells, `params_status`, `equity_path_for_model`, `export_report_pdf` — Figure-based, safe off the main thread). rl_app keeps only the signature cache. `test_webgui.py::test_models_table_equals_the_ctk_rows` is the parity check. |
| `backtest_report.py` + `webgui/backtest.py` | Backtest page shared by rl_app and the web GUI. `backtest_report.py`: the full report text from `<model>_live_bt.meta.json` (`report_lines`, `stat_cards`, `report_rows` = the same lines split for a table), the companion-M1 lookup (`find_matching_m1`), the M1 auto-select rule (`auto_select_m1`: never overrides an explicit choice) and the Skip-% range preview (`skip_preview`); rl_app delegates to all of them. `webgui/backtest.py`: form → `backtest_live.py` / `backtest_chart.py` command with the CTk rules (max_hold from the model's train meta, FormError names the field); `test_ui_training_contract.py::test_web_backtest_command_equals_ctk_*` and `test_ctk_report_text_equals_backtest_report` are the parity checks. Web run events are routed to the page that started the run (`window.rlweb.handlers[page]`). |
| `pipeline_shared.py` + `webgui/pipeline.py` | Pipeline / Walk-Forward / Fine-tune shared by rl_app and the web GUI. `pipeline_shared.py`: pipeline train/backtest/chart stage commands, the backtest range fixed at review (`resolve_backtest`), split preview, Backtest-CSV follow rule, `WF_PROGRESS` + verdict parsing, WF chart lookup, fine-tune recipe/mode hints (rl_app delegates). `webgui/pipeline.py`: the three forms → commands with the CTk rules (`build_pipeline` + `pipeline_stages`, `wf_from_train` = ⧉ Copy settings from Train, `build_wf_command`, `build_ft_command`). The web runner's `start_sequence` runs stages gated on exit code and stays busy across stages. Relabel/Build stages are switched off in the CTk pipeline, so both apps run Train PPO → Backtest → Chart. Parity: `test_ui_training_contract.py::test_web_pipeline_*`, `test_web_walkforward_*`, `test_web_finetune_*`. |
| `data_tools.py` + `regime_tools.py` | Data Tools and Regime Check shared by rl_app and the web GUI. `data_tools.py`: collector import (+ M1 + params), split by date (+ auto-clean), correlation heatmap (Figure-based now — the CTk worker used to call `matplotlib.use('Agg')` on the running app), clean redundant features, smart UTF-16 reader, params-sidecar helpers, ONNX export plan; every function reports through `log(text, tag)` / `notice(...)` callbacks. `regime_tools.py`: detect + Gemini commands (the key is masked in anything shown), events status, breakpoint table, cutoff → `<stem>_from_<date>.csv`. rl_app delegates; the web runs the in-process tools on a worker thread (`Api._tool_job`, page 'tools'). Parity: identical output files + log lines vs the pre-refactor code (one-off check), `test_web_regime_cutoff_writes_the_same_file_as_ctk`, `test_web_analyze_and_regime_commands_equal_ctk`, `test_web_gemini_command_masks_the_key_like_ctk`. |
| `app_settings.py` | Settings storage shared by rl_app and the web GUI: `COLOR_PRESETS`, `branding_config.json` read/write (same `{branding, colors}` format), `api_keys.json` load/save, `web_settings.json` (web theme + brand accent). The web Settings page never returns a stored key to the page (only set + last 4 chars). |
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
| `rl_walkforward.py` (root) | 5-window rolling train/test. Four verdict levels: ROBUST / MOSTLY ROBUST / UNSTABLE / NOT ROBUST. Validates the training *recipe*, not a saved model file (retrains per window). Accepts the full Train recipe (reward mode/profile/overrides/formula, PPO hyperparams, max hold, net_arch); train episodes are clamped to the available window rows instead of walking past the dataframe. GUI has a "⧉ Copy settings from Train" button so WF validates the SAME recipe you train with. |
| `rl_finetune.py` (root) | Smart fine-tune: mix old(30%) + new(70%), lower LR (1e-4), ~50k steps. Prevents catastrophic forgetting. |
| `rl_analyze.py` (root) | Confidence → accuracy analysis (is there a usable threshold?). Resolves the model's action profile/config and max hold from train metadata, with action-count fallback for older models. |
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
| `api_keys.json` (gitignored) | User's Gemini API key. Loaded by `app_settings.load_api_keys()` (alias `_load_api_keys()` in `rl_app.py`). Edit via the Settings page → "API Keys" card. **Never** commit this file. |

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
- Feature selection (correlation) must use TRAIN data only. The Train page **Correlation threshold** field (→ `rl_train.py --corr_threshold`, default 1.0 = constants only, 0.95 ≈ 1.8x faster) prunes on Train rows inside the trainer; the Walk-forward page has the same field (blank = the CSV's own `.features.json` threshold) and "⧉ Copy settings from Train" copies it. A `_clean` CSV without `.features.json` (cleaned before 2026-09) is rejected — train the original CSV with this field instead.

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
| `branding_config.json`, `api_keys.json`, `known_events.json`, `web_settings.json` | Per-user runtime files (gitignored). |
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
| MT5 tester opens hedged/stacked positions the backtest never showed | EA template used `InpMaxPositions = 3` and opened Buy whenever not-long (including while short) — TradingEnv/backtest are strictly single-position (`position == 0` gate). Fixed 2026-07-27: default 1 + flat-only opens. Packages exported before the fix carry the old defaults — set the EA input `Max concurrent positions = 1` in the tester, or re-export + recompile. |
| MT5 compile: 25× "undeclared identifier" (`SESSION_ASIA_START`, `EX_TEMA_PERIOD`, …) in `<model>_config.mqh` | Version mismatch: the generated config assigns globals that only exist in the **current** `RL_Indicators.mqh` (extra-features update, 2026-07). The terminal being compiled still has a pre-update copy in `MQL5/Include/`. Fix: copy the bundled `packages/<name>/MQL5/Include/RL_Indicators.mqh` (or `mt5_files/MQL5/Include/RL_Indicators.mqh`) into that terminal's Include folder — always install BOTH files from a package, never just the config. |
| Export fails "action profile has 4 actions, model has 6" | Stale train meta: `rl_train` writes `<name>.train.json` at train **start**, so starting a re-train with the same name and stopping it mid-run leaves a meta describing a run whose zip was never written. `export_to_onnx` now trusts the zip (auto-detects the profile from the model's action count, warns, uses that profile's DEFAULT action params). If you had customized action params, retrain or fix the meta before exporting. |
| Extras-model EA uses the wrong ATR for SL/TP | `ATR_PMIN/PSTEP` grids may not contain period 14, so indexing `g_h_atr[14-ATR_PMIN]` is invalid. The EA template now uses dedicated `g_h_atr14_ref`; re-export affected packages and recompile rather than hand-editing generated EAs. |
| Backtest/Analyze silently use default action/max-hold settings | Both flows now resolve the saved training contract. Backtest GUI passes model-meta `max_hold`; direct Backtest CLI auto-resolves it; Analyze resolves action profile/config/params plus max hold and validates action dimension. |
| Developer Mode unchecked but reward JSON formula still trains | Formula-bearing JSON is no longer forwarded when Developer Mode is off; the visible base profile and slider overrides are passed without the JSON formula. |
| Switching Backtest datasets leaves the old M1 selected | Auto-selected M1 values now carry provenance and are re-evaluated on dataset change; no matching companion resets the dropdown to `(none)`, while a genuine manual choice is preserved. |
| Fine-tune output name equals base model | GUI and CLI block case-insensitive name collisions before any train meta or weights can be overwritten; use a suffix such as `_ft` or `_v2`. |
| Rapid double-click starts two subprocesses / immediate Stop does nothing | `ProcessRunner` now treats `_starting` as busy, rejects a second start, and records cancellation until Popen returns. Keep process operations on the captured local `proc`; only clear shared state when `self.proc is proc`. |
| Closing the app leaves Python/train children running | `WM_DELETE_WINDOW` now confirms, stops the shared runner, terminates the current pipeline child, waits briefly, then kills if needed before destroying the GUI. |
| Data Tools worker crashes with `main thread is not in main loop` | Snapshot all widget `.get()` values before spawning workers; `_log` and dropdown refreshes marshal through `after(0, ...)`, and shutdown-time logging becomes a safe no-op. |
| Numeric text such as `300,000` or `2e5` bricks a run page | Parse integer fields with `_parse_int_field` before disabling buttons or changing run status; it accepts comma/scientific whole numbers and rejects fractional/invalid values with a warning. |
| Stored backtest report changes after a failed retry or model dropdown change | Failed retries preserve the last complete meta in place and write a failure sidecar; the GUI captures the model at launch and renders that model on completion. |
| Regime HTML is blank from disk / stale cutoff targets another CSV | Charts inline their JSON payload for `file://`; result JSON records the resolved source CSV, and the cutoff action rejects stale provenance or an empty slice. |
| Exported config says params are embedded when only a no-op exists | Generated configs define `RL_PARAMS_EMBEDDED` as 1 or 0; UI readers use that marker (with the old `// Embedded from` comment as legacy fallback). Deploy names are normalized to valid ASCII MQL identifiers. |
| Constant features explode during normalization | New train/WF runs floor std values below `1e-6` to `1.0`; Python and MQL consumers divide by the saved std exactly. Existing model norms are left unchanged unless explicitly migrated/retrained. |
| Collected indicator features use the wrong periods/thresholds (2026-08 datasets: nearness 100→250 and 250→1500, every divergence column 0, `candle_marubozu` on 67% of bars, `candle_mathold` always 0) | MQL5 `iCustom()` counts **every `input group` line as one parameter slot**, so values passed by the collector/EA land one slot later per group. Indicators read via iCustom must declare NO `input group` (PriceNearness v1.11, PriceDivergence v1.10). DataCollector_RL self-checks nearness against `iHighest()` and stops on mismatch. **CandlePatterns still has groups on purpose**: its shifted params (marubozu 0.30, MatHold never fires, …) are identical in collector and EA, so every trained model and live EA is consistent — do not "fix" it without a versioned indicator name + full re-collect/retrain. Details: `docs/data_side_2026-10-10.md`. |
| Viewer page is blank / `Failed to resolve module specifier "three"` when opened by double-click | The viewer imports the vendored Three.js (`tools/viz/vendor/three`, r160) as ES modules, which browsers block on `file://`. Open it through `tools/viz/live_viewer.py` (`--model`, `--run_dir` or `--html`), which serves `tools/viz` + the run's `viz/` folder on 127.0.0.1. |
| A Python ATR disagrees with the EA's SL distance by ±10-18% | MT5's built-in `iATR` is a SIMPLE moving average of true range (Indicators/Examples/ATR.mq5), not Wilder/RMA smoothing. `srb_features.atr_mt5` reproduces it; with it the twin's 1.5*ATR stop matches the tester's R to ±2%. |
| Continuous SL head trains but every trade uses the minimum stop | SB3's orthogonal init zeroes the action-net bias, so the SL mean starts at 0 ATR and the env clips every sample to `sl_range[0]`; PPO then sees identical outcomes for different samples and the head never moves. `HybridPolicy._build` sets the mean's bias to the range midpoint after `super()._build` (42e766a). |
| A sparse event (<3% of bars) given as a feature is never used by PPO | Seen with the SRB columns (`docs/srb_rl_ab_2026-10-12.md`): entries on signal bars = chance, direction = chance, extra columns only add noise. Give the event to the env as a gate (`--entry_gate`) instead of as an input. |
| A pywebview js_api call never resolves in JS (promise stays pending) | The return value is not valid JSON for the bridge — seen with `float('inf')` (METRIC_INFO bounds → `Infinity`). The web GUI exposes `webgui.api.JsBridge`, which returns `to_json(result)` (ASCII, inf/NaN → null) and catches exceptions; app.js JSON.parses. Also: pywebview only exposes real class methods (instance-attribute functions are invisible) and must not hold a `Window` as an attribute. |
| MTM mode appears to accept reward sliders/formula | Reward controls are disabled and labeled ignored in MTM mode; train metadata records `reward_profile_active=false`. DQN/A2C meta nulls unsupported PPO-only fields and constructors use their supported GUI hyperparameters. |

---

## 10. Reference Documents

- `README.md` — project overview, install, Codex workflow, sample models.
- `docs/metafxclub studio guide/` — **current user flow guide** (start at `00_reading_order.md`):
  dashboard flow → data prep → quality/regime check → train → backtest → walk-forward →
  custom feature guide → project folder map. HTML deep-dives, one per dashboard step.
- `docs/codex_git_update_prompts.md` — safe "pull + reinstall deps" prompt for Codex.
- `docs/codex_custom_action_prompts.md` — prompt for asking Codex to add/change an Action
  Profile across train/backtest/fine-tune/walk-forward/export/EA consistently.
- `docs/bug_audit_2026-07-12.md` — full-flow bug audit backlog (47 confirmed findings with
  trigger + fix sketch each; checkboxes track fix status).
- `docs/codex_bug_fix_prompts.md` — copy-paste Codex prompts to work that backlog in 4 batches
  (HIGH data-corruption → ProcessRunner/threading structural → per-page MEDIUM → LOW).
- `docs/explainers/11_production_readme.md` — full 5-step production deployment guide.
- `docs/metafxclub studio guide/12_monte_carlo_deep_dive.html` — what the MC process is, the two probes (shuffle-DD, skip-retention), and the Train-MC vs Backtest-MC comparison (idealized env trades vs full-cost SimAccount trades; screening vs capital sizing).
- `docs/metafxclub studio guide/13_regime_check_deep_dive.html` — regime detection algorithms (HMM/K-Means/PELT), per-method parameter tables (each method takes a DIFFERENT param set: States vs k+Window vs Penalty), fixed in-code constants (daily resample, 25-of-30-day persistence, ±90d match tolerance), tuning recipes and limits.
- `docs/metafxclub studio guide/14_mc_chart_reading.html` — how to read every element/number of the MC fan chart + verdict panel (envelopes, median/95%-worst/P(hard stop), PASS/CAUTION/FAIL rule, orange-line interpretation, in-sample trap).
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
