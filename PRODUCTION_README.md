# 🚀 Production Deployment Guide

End-to-end guide สำหรับ deploy RL Trading System บน MT5

---

## 📋 Prerequisites

```bash
# Python packages
pip install MetaTrader5 stable-baselines3 gymnasium pandas numpy

# MT5 Terminal (เลือก broker)
# - Exness MT5
# - IC Markets MT5
# - หรือ broker อื่นๆ ที่ support MT5
```

---

## 🎯 Optimal Train/Test Split

### สำหรับ Test = 2021-2026 (5+ ปี)

```
✅ แนะนำ:
   Train: 2010-2020 (10 ปี)
   Test:  2021-2026 (5+ ปี)
   ratio: 67:33

✅ Walk-Forward เสริม:
   หลัง train → run rl_walkforward.py
   ทดสอบ 5 windows ใน test period
```

### Math Check
```
10 ปี × 24h × 5 days × 52 weeks ≈ 62,400 H1 bars
sample/feature ratio = 62,400 / 350 = 178  ✅ (>50)
```

---

## 🔄 Production Workflow (5 Steps)

### Step 1: Pull Training Data

```bash
# Pull 10 years for training (Strategy A)
python pull_mt5_data.py \
    --symbol XAUUSDm \
    --start 2010-01-01 \
    --end 2020-12-31 \
    --name train_data_2010_2020

# Pull 5 years for out-of-sample test (DON'T TOUCH ในขั้น train!)
python pull_mt5_data.py \
    --symbol XAUUSDm \
    --start 2021-01-01 \
    --end 2026-04-30 \
    --name test_data_2021_2026
```

### Step 2: Relabel (if needed)

```bash
# ถ้า class imbalance — quantile balance ก่อน
python relabel.py train_data_2010_2020.csv
# Output: train_data_2010_2020_relabeled.csv
```

### Step 3: Train RL Model

```bash
# Train scratch — 200k steps (~30 นาที CPU)
python rl_train.py train_data_2010_2020_relabeled.csv \
    --steps 200000 \
    --window 10 \
    --name rl_prod_v1
```

### Step 4: Validate

```bash
# 4a) Quick backtest (TradingEnv) — ดู baseline
python rl_backtest.py rl_prod_v1 test_data_2021_2026.csv

# 4b) Walk-Forward 5 windows
python rl_walkforward.py test_data_2021_2026.csv \
    --windows 5 --steps 50000 \
    --name wf_prod_v1

# 4c) ⭐ LIVE LOGIC BACKTEST (สำคัญที่สุด — ใกล้ live จริง)
python backtest_live.py rl_prod_v1 test_data_2021_2026.csv \
    --conf 0.85 --atr_sl 2.0 --atr_tp 4.0

# 4d) Grid search หา hyperparameters ที่ดีสุด
python grid_search.py
```

**Decision Gate:**
- ✅ rl_backtest (env): PF > 1.0
- ✅ Walk-forward: ผ่าน ≥ 4/5
- ✅ **backtest_live: PF > 1.1** (สำคัญที่สุด!)
- → ผ่านครบ 3 ข้อค่อย deploy

**Decision Gate:**
- ✅ ถ้า Walk-Forward ผ่าน ≥ 4/5 windows + PF mean > 1.05 → ไป Step 5
- ❌ ถ้าไม่ผ่าน → กลับไป tune (features, hyperparams, reward shaping)

### Step 5: Deploy to MT5

#### 5.1 Demo first (1 เดือน, 30+ trades minimum)

```bash
# แก้ live_trader.py CONFIG ให้ตรง
# - SYMBOL = "XAUUSDm"
# - MODEL_NAME = "rl_prod_v1"

# รัน demo
python live_trader.py --demo
```

#### 5.2 Live with small lot (1-3 เดือน)

```bash
# ก่อน live: confirm พร้อมจริง
# - Demo 1 เดือนแล้ว
# - Live vs backtest ใกล้กัน
# - Risk parameters set

python live_trader.py --live
# จะถาม confirmation "Type YES"
```

#### 5.3 Scale Up

- เริ่มที่ lot 0.01
- ค่อยๆ เพิ่มเป็น 0.05 → 0.10 → ...
- ทุกเดือนดู rolling PF, max DD

---

## 🔄 Maintenance Cycle

### Quarterly (ทุก 3 เดือน) — Fine-tune

```bash
# Pull data ใหม่ 3 เดือนล่าสุด
python pull_mt5_data.py --days 90 --name latest_data

# Relabel
python relabel.py latest_data.csv

# Fine-tune — smart mixing 30% old + 70% new
python rl_finetune.py rl_prod_v1 \
    --old_csv train_data_2010_2020_relabeled.csv \
    --new_csv latest_data_relabeled.csv \
    --steps 50000 --mix_ratio 0.3 \
    --name rl_prod_v2

# Validate
python rl_walkforward.py latest_data_relabeled.csv \
    --windows 5 --name wf_prod_v2

# ถ้าผ่าน gate → swap model
cp rl_prod_v1.zip rl_prod_v1_archive_2026Q1.zip  # backup
cp rl_prod_v2.zip rl_prod_v1.zip                 # promote
cp rl_prod_v2_norm.csv rl_prod_v1_norm.csv

# Restart live_trader
```

### Yearly (ทุก 1 ปี) — Retrain Scratch

```bash
# Pull all data again
python pull_mt5_data.py \
    --start 2010-01-01 \
    --end $(date +%Y-12-31) \
    --name train_data_yearly

# Train from scratch (200k steps)
python rl_train.py train_data_yearly_relabeled.csv \
    --steps 200000 --name rl_prod_yearly

# Walk-forward
python rl_walkforward.py test_data.csv \
    --name wf_yearly

# Promote ถ้าผ่าน
```

---

## 🛡️ Risk Management Settings

### live_trader.py CONFIG

```python
# Confidence
CONFIDENCE_THRESHOLD  = 0.85    # ลด → trade เยอะแต่คุณภาพต่ำลง

# Risk per trade
RISK_PER_TRADE        = 0.01    # 1% — เริ่มที่นี่ 0.5% ยิ่งดี

# Position limits
MAX_POSITIONS         = 3       # เริ่มที่ 1 แล้วค่อยเพิ่ม

# Drawdown
HARD_DD_PCT           = 0.15    # 15% → ปิดทุก position
SOFT_PAUSE_PF         = 0.90    # PF < 0.9 → pause

# SL/TP
ATR_SL_MULTIPLIER     = 1.5     # SL = ATR × 1.5
ATR_TP_MULTIPLIER     = 3.0     # TP = ATR × 3.0 (RR 1:2)
```

---

## 📊 Monitoring

### Inspect live_trades.db

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("live_trades.db")

# Recent trades
trades = pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50", conn)
print(trades)

# Events log
events = pd.read_sql("SELECT * FROM events ORDER BY id DESC LIMIT 20", conn)
print(events)

# Rolling stats
trades_executed = pd.read_sql(
    "SELECT * FROM trades WHERE executed=1 ORDER BY id DESC LIMIT 50", conn
)
```

### Daily Checklist

- [ ] Live trader still running?
- [ ] Account equity OK?
- [ ] Trades happening (not 0)?
- [ ] Slippage acceptable?
- [ ] Any error events in DB?

### Weekly Checklist

- [ ] Rolling PF > 0.95?
- [ ] Win rate > 50%?
- [ ] Max DD < 10%?
- [ ] Live result vs backtest within 20%?

### Monthly Checklist

- [ ] Performance review
- [ ] Compare with last month
- [ ] Decision: continue / fine-tune / pause

### Quarterly Checklist

- [ ] Pull latest data
- [ ] Fine-tune
- [ ] Walk-forward validate
- [ ] Deploy if passes

---

## 🚨 Emergency Procedures

### Live trader keeps losing
```bash
# 1. Check live_trades.db for patterns
# 2. Pause trading
# 3. Run walk-forward on recent period
# 4. If broken → revert to previous model
cp rl_prod_v1_archive_*.zip rl_prod_v1.zip
```

### MT5 disconnected
```bash
# Watchdog should auto-restart
# Manual check:
ps aux | grep live_trader
# Or Task Scheduler / systemd status
```

### Model file corrupted
```bash
# Use archived backup
cp rl_prod_v1_archive_LATEST.zip rl_prod_v1.zip
cp rl_prod_v1_archive_LATEST_norm.csv rl_prod_v1_norm.csv
```

---

## 📁 File Structure

```
pycaret_trainer/
├── 🤖 Model Files
│   ├── rl_prod_v1.zip
│   ├── rl_prod_v1_norm.csv
│   └── archives/
│
├── 📊 Data
│   ├── train_data_2010_2020.csv
│   ├── train_data_2010_2020_relabeled.csv
│   ├── test_data_2021_2026.csv
│   └── latest_data_*.csv (quarterly)
│
├── 🔧 Core Scripts
│   ├── features.py            ⭐ feature calc (Python)
│   ├── mt5_connector.py       ⭐ MT5 wrapper
│   ├── live_trader.py         ⭐ main loop
│   ├── pull_mt5_data.py       ⭐ data puller
│   ├── trading_env.py         (RL env)
│   ├── rl_train.py            (train scratch)
│   ├── rl_finetune.py         (fine-tune)
│   ├── rl_backtest.py
│   ├── rl_walkforward.py
│   ├── rl_analyze.py
│   ├── rl_backtest_filtered.py
│   ├── relabel.py
│   └── quarterly_update.py    (orchestrator)
│
├── 📊 Logs & Reports
│   ├── live_trades.db         (SQLite)
│   ├── *_equity.png
│   └── wf_*_results.csv
│
└── 📚 Documentation
    ├── PRODUCTION_README.md   (this file)
    └── *.html (visualizations)
```

---

## 🎓 Quick Reference

| Task | Command |
|---|---|
| **Pull training data** | `python pull_mt5_data.py --start 2010-01-01 --end 2020-12-31 --name train` |
| **Train scratch** | `python rl_train.py train.csv --steps 200000 --name rl_v1` |
| **Backtest** | `python rl_backtest.py rl_v1 test.csv` |
| **Walk-forward** | `python rl_walkforward.py test.csv --windows 5 --name wf_v1` |
| **Run demo** | `python live_trader.py --demo` |
| **Run live** | `python live_trader.py --live` |
| **Paper test** | `python live_trader.py --paper` |
| **Quarterly fine-tune** | `python rl_finetune.py rl_v1 --old_csv old.csv --new_csv new.csv --steps 50000` |

---

## ⚠️ DO / DON'T

### ✅ DO
- Demo testing 1+ เดือนก่อน live
- Backup model + DB ทุกวัน
- Fine-tune ทุก 3 เดือน
- Monitor daily
- Start small (lot 0.01)
- Walk-forward ทุกครั้งก่อน deploy
- Document ทุกการเปลี่ยนแปลง

### ❌ DON'T
- ❌ Live trade ทันทีหลัง train
- ❌ Skip walk-forward
- ❌ Auto-deploy โดยไม่มี gate
- ❌ ใช้ leverage สูงตอนเริ่ม
- ❌ Trade ทุก signal (ใช้ confidence filter)
- ❌ ลืม backup model เก่า
- ❌ Touch test data ตอน train

---

## 🚀 Roadmap

```
Now:        Phase 1 — Manual quarterly                 ← เริ่มที่นี่
2-3 เดือน:   Phase 2 — Scheduled (Task Scheduler)
6+ เดือน:   Phase 3 — Trigger-based + drift detection
1 ปี:       Multi-asset portfolio + ensemble
```
