# Meta RL Trading Project

End-to-end Reinforcement Learning trading studio for MT5 data collection, PPO training, backtesting, walk-forward validation, and MT5 export.

The active application is `rl_app.py` through the Windows launcher `run_rl_app.bat`.

## Quick Install

Prerequisites on Windows:

- Python 3.10 or 3.11 available as `py -3` or `python`
- MetaTrader 5 desktop terminal if you will run DataCollector_RL, Strategy Tester, MT5 export, or live/pull-data tools
- Git for cloning/pulling the repo

For a new machine or another Codex workspace:

```powershell
git clone https://github.com/omesbooks/meta_rl_project.git
cd meta_rl_project
.\run_rl_app.bat
```

`run_rl_app.bat` is the recommended path on Windows. It creates `.venv` when missing, installs `requirements.txt` during first setup, and launches the dashboard.

If you already have a `.venv` and later pull new dependency changes, refresh the environment before launching:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the virtual environment already exists, you can also run:

```powershell
.\.venv\Scripts\python.exe rl_app.py
```

## Using With Codex

After cloning the repo, open the project folder in Codex and ask:

```text
อ่าน AGENTS.md แล้วเช็ค setup จากนั้นรัน .\run_rl_app.bat ให้หน่อย
```

Codex should use the project instructions in `AGENTS.md`, keep the working directory at the repo root, and use `.venv\Scripts\python.exe` for Python commands.

To update an existing local checkout, open the project folder in Codex and paste this prompt:

```text
ช่วยอัพเดทโปรเจคนี้จาก GitHub และติดตั้ง dependencies ล่าสุดแบบปลอดภัย

ขั้นตอน:
1. เช็คว่าอยู่ใน repo ถูกต้อง:
   git remote -v
   git branch --show-current
   git log -1 --oneline

2. เช็ค local changes:
   git status --short

3. ถ้า worktree สะอาด ให้ update จาก remote:
   git fetch --all --prune
   git pull --ff-only origin main

4. ถ้ามี local changes ห้าม reset / checkout / ทับไฟล์เอง
   ให้หยุดแล้วสรุปไฟล์ที่เปลี่ยนก่อน เพื่อให้เลือกว่าจะเก็บงาน local ไว้อย่างไร

5. หลัง pull แล้ว ให้ติดตั้ง dependencies ตามไฟล์ของโปรเจค:
   - ถ้ามี requirements.txt ให้รัน .venv\Scripts\python.exe -m pip install -r requirements.txt
   - ถ้ามี package-lock.json ให้รัน npm install
   - ถ้ามี pyproject.toml ให้ดู README หรือไฟล์ config ว่าควรใช้ pip / poetry / uv

6. เช็ค dependency conflict:
   .venv\Scripts\python.exe -m pip check

7. รัน smoke test เบื้องต้นของโปรเจค เช่น:
   .venv\Scripts\python.exe -B -c "import rl_app; print('app import ok')"

8. สรุปผล:
   - ก่อนอัพเดทอยู่ commit ไหน
   - หลังอัพเดทอยู่ commit ไหน
   - dependencies ติดตั้ง/ตรวจผ่านไหม
   - มี error/warning อะไรเหลือไหม
   - git status หลังทำเสร็จเป็นยังไง
```

The same update prompt is also kept in `docs\codex_git_update_prompts.md`.

## What Users Need Locally

This repo does not include large generated runtime files such as datasets, trained models, logs, ONNX exports, or user-specific params.

Each user should generate or provide their own:

- MT5 CSV dataset from DataCollector_RL
- optional matching `.params.json` sidecar for feature parity
- trained model `.zip`
- ONNX/EA export artifacts
- API keys in local `api_keys.json` if using Gemini event labeling

These generated files are intentionally ignored by git.

## Main Dashboard Flow

1. Run DataCollector_RL in MT5 to create a CSV in `Common\Files`.
2. Open the dashboard and go to `Data Tools`.
3. Use `Import from DataCollector_RL` to copy CSV + params into the project.
4. Use date split and feature analysis if needed.
5. Train PPO on the `Train` page or run the `Pipeline`.
6. Backtest the trained model.
7. Use Walk-Forward and Regime Check for robustness.
8. Export passing models to ONNX/MT5 EA.

The detailed user guide is in:

```text
docs\metafxclub studio guide
```

Start with:

- `00_reading_order.md`
- `03_dashboard_user_flow_guide.html`
- `15_data_prep_import_detail.html`

Reference/background explainers are separated in:

```text
docs\explainers
```

## Core Files

| Path | Purpose |
| --- | --- |
| `rl_app.py` | Main CustomTkinter dashboard |
| `run_rl_app.bat` | Windows launcher and setup path |
| `trading_env.py` | Gymnasium trading environment and reward behavior |
| `rl_train.py` | PPO training CLI |
| `backtest_live.py` | Production-style backtest |
| `backtest_chart.py` | Backtest chart/report generation |
| `rl_walkforward.py` | Walk-forward validation |
| `rl_finetune.py` | Fine-tuning workflow |
| `export_to_onnx.py` | PPO model to ONNX + MT5 EA/config |
| `regime_compare.py` | Regime detection / structural break tooling |
| `gemini_labeler.py` | Optional Gemini-based event labeling |
| `mt5_files/` | MT5 indicators, collector, EA template, include files |
| `tools/` | Supporting CLI tools grouped by purpose |
| `legacy/` | Older PyCaret/MQL files kept for reference |
| `docs/metafxclub studio guide/` | Current flow guide plan and deep-dive docs |

## Common Commands

Train from CLI:

```powershell
.\.venv\Scripts\python.exe rl_train.py <csv> --steps 200000 --window 10 --name rl_v1
```

Backtest:

```powershell
.\.venv\Scripts\python.exe backtest_live.py rl_v1 <csv> --conf 0.85 --window 10 --mode pure_agent
```

Walk-forward:

```powershell
.\.venv\Scripts\python.exe rl_walkforward.py <csv> --windows 5 --steps 50000
```

Export to MT5:

```powershell
.\.venv\Scripts\python.exe export_to_onnx.py rl_v1 --name rl_v1_deploy
```

## Data Notes

For the RL workflow, the CSV should contain market state columns such as:

- `timestamp`, `symbol`
- `open`, `high`, `low`, `close`, `volume`
- numeric indicators/features from the MT5 collector

RL does not require `future_return` or `UP/DOWN/FLAT` target labels. Those are supervised/PyCaret concepts and are kept only for legacy workflows.

## Safety Notes

This is research software. A model that trains successfully is not automatically tradable.

Always validate with:

- out-of-sample backtest
- walk-forward validation
- drawdown review
- confidence threshold review
- MT5 Strategy Tester before any live deployment

Trading involves substantial risk of loss. Do not deploy without extensive validation.
