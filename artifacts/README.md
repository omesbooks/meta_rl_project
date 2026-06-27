# Sample Backtest Artifacts

This folder intentionally tracks a small curated subset of generated model artifacts so new users can try the Backtest page before training their own model.

Included sample pairs:

| Model | Dataset | Notes |
| --- | --- | --- |
| `rl_uj_h4` | `uj_h4_dataset.csv` | Uses the tracked UJ dataset and params sidecar. |
| `rl_au_h4` | `data_au_h4_dataset_clean.csv` | Uses the tracked cleaned AU dataset and params sidecar. |

Each included model folder keeps:

- `<model>.zip`
- `<model>_norm.csv`
- `<model>.params.json`
- `<model>.train.json` when available
- `backtests/*_live_bt.meta.json`
- `backtests/*_live_bt_trades.csv`
- `backtests/*_live_bt_equity.png`
- `backtests/*_backtest_chart.html`

Omitted on purpose:

- `logs/`
- `best/`
- ONNX export files
- generated MT5 deployment packages

Quick CLI smoke test:

```powershell
.\.venv\Scripts\python.exe backtest_live.py rl_uj_h4 uj_h4_dataset.csv --conf 0 --max_positions 1 --window 0 --mode pure_agent
```

GUI path:

```text
Backtest -> select model -> select dataset -> Run Backtest
```
