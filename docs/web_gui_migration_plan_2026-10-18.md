# Plan: moving rl_app.py (CustomTkinter) to a web GUI in pywebview

Decision context (2026-10-18): the user chose the "Clear Liquid" design for the training
Monitor and wants the main app to match. CustomTkinter cannot render glass (no blur, no
per-widget alpha), so the path is a web front end inside a **native pywebview window** —
same launcher, same window behaviour, no browser. Demo: `tools/viz/gui_demo.py`.

## 1. Target architecture

```
run_rl_app.bat  ──▶  rl_web.py  (pywebview window, native title bar, icon, close hook)
                         │  window.pywebview.api  (JSON in / JSON out, one process)
                         ▼
                     webgui/api.py        Api class: one method per user action
                         │                 (list_csvs, csv_info, start_training, ...)
                         ├── webgui/runner.py   ProcessRunner → Popen + stdout pump → evaluate_js
                         ├── rl_train.py / backtest_live.py / ... (UNCHANGED CLI scripts)
                         └── artifact_paths / training_data / action_profiles (UNCHANGED)
                     webgui/ui/index.html + app.js + theme.css   (one page, 10 panels)
                     tools/viz/*            live visual (already pywebview; same theme)
```

Rules that keep this safe:
- **No logic moves into JavaScript.** Validation, recipe hashing, overwrite confirms, path
  resolution, meta reading stay in Python (`Api`). JS only renders state and calls methods.
- **CLI scripts are the contract.** Every page action builds the same command line the CTk
  app builds today; `test_ui_training_contract.py` is ported to call `Api` methods instead
  of widget handlers so the recipe hash / argument set cannot drift.
- **Old app stays runnable** until the last page is ported (`rl_app.py` untouched; the bat
  gets a `--classic` flag).

## 2. What exists today and where it goes

| rl_app.py today (10,336 lines, 273 methods) | goes to |
|---|---|
| `NAV_ITEMS` 10 pages, lazy page builders | `index.html` panels + `app.js` router (demo has it) |
| `ProcessRunner` (busy flag, cancel, owner page routing of stdout/done) | `webgui/runner.py`: Popen, reader thread, `evaluate_js("onLogLine(...)")` per line, `onDone(rc, page)`; same `_starting`/busy semantics, same `_is_process_busy()` guard |
| `_handle_log_line` (progress regex, VIZ_SNAPSHOT/ROLLOUT, metric pills, page routing) | Python keeps parsing (`_parse_progress`, `_classify_log`, viz launch); sends structured events `{line, cls, progress, metrics, viz}` — JS only paints |
| `_start_training` / `_review_training_command` (overwrite confirm, recipe fingerprint, viz args) | `Api.start_training(cfg)` returns `{command, recipe_sha256, warnings}` → JS shows the review card → `Api.confirm_training(token)` launches. Same two-step as today |
| `_collect_model_rows` (cached listing, Realism column) | `Api.list_models()` (demo already reads real meta via `train_meta_path`) |
| Backtest report panel (meta JSON → equity PNG + tables) | `Api.backtest_report(model)` returns the meta; HTML renders tables; PNG served as data URL |
| Regime page (run detection → JSON → table → cutoff → switch to Train) | `Api.regime_run / regime_results / regime_cutoff`; cross-page state via the JS router |
| Data Tools worker threads (`after(0, ...)` marshalling) | worker threads post events through `evaluate_js`; same "snapshot inputs before spawning" rule |
| Settings: branding, API keys (`api_keys.json`), colour presets | `Api.settings_get/set`; theme = CSS variables; presets become CSS files (`clear`, `deep`, `classic-dark`) |
| `_on_close` (confirm, stop runner, kill children) | pywebview `window.events.closing` → same routine |
| Live visual (`_launch_viz_window`, mode menu, Open Visual) | unchanged; `Api.open_visual()` already in the demo |

## 3. Phases (each ends with the old app still working)

| phase | scope | days | exit test |
|---|---|---:|---|
| 0 | `webgui/` skeleton: window, router, theme tokens from the demo, `runner.py`, event protocol, `--classic` in the bat | 2 | a subprocess (`rl_train.py --help`) streams into the page; closing the window kills it |
| 1 | **Train** page complete (all fields incl. advanced PPO, reward sliders, action JSON, gate, costs, viz) + review card + overwrite confirm + metric pills + reward trend | 4 | `test_ui_training_contract.py` ported → passes against `Api` (same commands, same recipe hash as CTk for the same inputs) |
| 2 | **Models** (listing, Realism column, equity preview, open files, PDF export — the classic page has no delete/rename) + **Settings** (branding, API keys, theme) | 2 | listing equals `_collect_model_rows()` output on the same folder |
| 3 | **Backtest** page + report panel + M1 auto-select logic | 3 | report renders from an existing `_live_bt.meta.json` identically (numbers compared) |
| 4 | **Pipeline** (stage gating by exit code, health pills) + **Walk-Forward** (copy-settings-from-Train) + **Fine-tune** | 4 | pipeline runs relabel→train→backtest→chart on a smoke CSV |
| 5 | **Analyze**, **Regime Check** (table, Gemini refresh, cutoff → Train), **Data Tools** (import, M1, clean, verify) | 4 | regime cutoff produces the same `_from_<date>.csv` as today |
| 6 | polish: keyboard focus, window icon/title, PyInstaller spec, docs + AGENTS.md, retire `--classic` default | 2 | fresh machine run from the bat |

Total ≈ 21 working days; Train+Models+Settings usable after phase 2 (~8 days).

## 4. Event protocol (Python → page)

```
onLogLine({page, line, cls})                      every stdout line (cls: info|warn|error|metric|viz)
onProgress({page, step, total, eta_s})
onMetrics({page, ep_rew_mean, approx_kl, clip_fraction, explained_variance, entropy_loss, value_loss, health})
onViz({page, kind: "rollout"|"eval", index, step, path})   → Api launches the viewer itself
onDone({page, rc, summary})
onBusy({busy: true|false, page})
```
Page → Python: plain `Api` methods returning JSON; long work always goes through `runner.py`
(never blocks the bridge thread).

## 5. Risks and how they are handled

- **Logic buried in widget code** (e.g. `_start_training` validates widget values inline):
  port method-by-method with the contract test as the oracle; any divergence in the built
  command line fails the test.
- **Thread safety**: only `evaluate_js` touches the page; same discipline as `after(0, ...)`.
- **WebView2 missing** on an old machine: `rl_web.py` checks `webview.platforms` and prints
  the runtime download link; Windows 10/11 ship it.
- **File dialogs / clipboard**: pywebview dialogs are native; clipboard via JS `navigator.clipboard`
  works in WebView2.
- **Performance**: pages are static DOM; only the log pane streams (cap at 2,000 lines like
  the CTk textbox). The live visual keeps its own window and its measured CPU profile.
- **Two apps during migration**: shared state is on disk only (artifacts, configs) — no
  conflict, but never run a training from both at once (`ProcessRunner` busy flag is per app).

## 6. Not in scope (unchanged)

CLI scripts, TradingEnv/recipes, artifact layout, MQL/EA export, tests of the engine.

## Status

- 2026-10-18 — **phase 0 + 1 done**: `rl_web.py`, `webgui/{commands,logparse,runner,api}.py`, `webgui/ui/`,
  `run_rl_web.bat` (the classic `run_rl_app.bat` is unchanged; the bat swap happens in phase 6).
  Train page: dataset (CSV list, browse, attach .params, rows/params status, split preview, gate column
  suggestions), recipe (steps + suggested, window, max hold, train %, corr threshold, MC, costs, seed),
  PPO hyperparameters + presets, reward sliders in display units + reward JSON load/save + developer
  formula (validated), action params + action JSON, review modal (same summary as CTk, overwrite warning,
  command + recipe sha256 + full recipe), live log (2,000-line cap), progress, 6 metric pills/KPIs with
  health colours, reward trend, live visual auto-open on the first VIZ line + Open Visual, close-window
  confirm while training. `metric_health.py` now holds METRIC_INFO / classify_metric (moved out of rl_app.py).
  Tests: `test_webgui.py` (6) + 3 parity tests in `test_ui_training_contract.py`; real-window end-to-end run
  (validation error → review → confirm → 4,096-step training → complete) passed.
- 2026-10-18 — **phase 2 done**: Models + Settings pages are live in the web GUI.
  Shared modules so both apps run one code path: `model_results.py` (rows, cell text, params status,
  equity lookup, PDF export) and `app_settings.py` (colour presets, branding/api-key/web-settings files);
  rl_app.py now delegates to them. Models page: same 10 columns + Realism, sortable headers, filter,
  equity preview (data URL thumbnail), open equity image / HTML chart, export report PDF; a model-only row
  shows the "not backtested yet" hint like the CTk viewer. The classic page never had delete/rename, so
  none was added. Settings page: web theme (Clear Liquid / Deep Glass, applied live, `web_settings.json`),
  optional brand accent, branding text + colour fields with presets and colour pickers (invalid hex is
  rejected with the field highlighted; the classic app silently dropped it), reset, open config file,
  Gemini key save/clear (the page only ever sees "set" + last 4 characters), working dir / python info.
  The window title and sidebar logo/name follow the branding (title also updates on Save).
  Exit test: `test_models_table_equals_the_ctk_rows` (65 rows, identical iid order and cells) plus a
  one-off check that the pre-refactor rl_app code and the new module return identical rows. Tests:
  test_webgui.py now 12; the whole UI/viz/gate suite 49 passed. Real-window e2e (temp settings folder,
  the user's real branding/key files untouched): models table + equity preview, sort, filter, model-only
  hint, bad colour → field marked, save → title updated, key save → "…LAST" status, Deep Glass switch.
- 2026-10-18 — **phase 3 done**: Backtest page live in the web GUI. `backtest_report.py` holds the report
  text, stat cards, companion-M1 lookup, M1 auto-select rule and Skip-% preview (moved out of rl_app.py,
  which now delegates); `webgui/backtest.py` builds the `backtest_live.py` / `backtest_chart.py` commands.
  Page: model + Final/Best checkpoint, dataset, M1 (auto-picked companion file, explicit choice kept until
  the dataset changes, browse), confidence, Skip % with the Skipped/Backtest date preview, Pure vs SL/TP mode
  (SL/TP + intrabar fields hidden in Pure like the CTk page), risk, slippage, swaps, random baseline, MC,
  window; Run / Generate Chart / Open Chart / Stop; live log; results = 4 stat cards, equity PNG, Monte Carlo
  fan chart, the report as a table (same lines split into section + text) and as raw text. Choosing a model
  shows its stored report; after a run the report of the run's model is shown even if the dropdown changed.
  Exit test: the pre-refactor CTk report code and the new module produce identical text and stat cards for
  all 48 stored reports in artifacts/models; command parity tests (defaults; Agent+SL/TP + optimistic +
  Best + M1 + swaps + window + meta max_hold) and invalid-input parity pass. Suite: 56 passed. Real-window
  e2e (window placed off-screen and captured with PrintWindow, so the user's screen is never touched):
  stored report, M1 auto pick, skip preview, explicit '(none)' kept, bad confidence → field marked, a real
  backtest of web_e2e_tmp3 → ✓ complete, 95 log lines, report rendered (PF 0.99, WR 53.7%).
