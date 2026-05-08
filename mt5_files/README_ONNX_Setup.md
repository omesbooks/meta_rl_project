# 🚀 MT5 Deployment via ONNX

วิธี deploy RL model ไปทดสอบใน MT5 Strategy Tester (และเทรด live) ผ่าน ONNX

---

## 📋 Prerequisites

```
✅ MT5 build 3550+ (Help → About to verify ONNX support)
✅ Python with: stable_baselines3, torch, onnx, onnxruntime, onnxscript
✅ Trained PPO model (.zip + _norm.csv)
✅ MetaEditor (มากับ MT5)
```

---

## 🗂️ Folder Structure

```
mt5_files/
├── README_ONNX_Setup.md            ← (this guide)
└── MQL5/                            ← copy contents to <MT5>/MQL5/
    ├── Files/
    │   └── <deploy_name>.onnx      (generated)
    ├── Include/
    │   ├── <deploy_name>_config.mqh (generated)
    │   └── RL_Indicators.mqh        (template)
    └── Experts/
        ├── ML_RL_Trader_template.mq5  (template — don't compile)
        └── <deploy_name>_EA.mq5       (generated — compile this!)
```

---

## 🎯 Quick Workflow (2 ways)

### **Method A: Use the GUI (recommended)** ⭐

1. เปิด `rl_app.py` → Tools page
2. ⑤ "🚀 Export to MT5 (ONNX)" card:
   - **Source model:** เลือกจาก dropdown (e.g., `rl_prod_v10_enriched`)
   - **Deploy name:** ตั้งชื่อ (default = model_name) เช่น `rl_v10`
   - **Output folder:** default = `mt5_files/MQL5/`
3. กด **"🚀 Export to ONNX + Generate MT5 Files"**
4. ดู Tools Log → ทุก file สร้างใน `mt5_files/MQL5/`

### **Method B: Command Line**

```bash
# Default deploy name = model name
python export_to_onnx.py rl_prod_v10_enriched

# Custom deploy name
python export_to_onnx.py rl_prod_v10_enriched --name rl_v10

# Custom output dir
python export_to_onnx.py rl_prod_v10_enriched --name rl_v10 \
    --output_dir mt5_files/MQL5
```

---

## 📦 What Gets Generated

```
<output_dir>/
├── Files/<deploy_name>.onnx           (~1.5 MB — model weights)
├── Include/<deploy_name>_config.mqh   (norm constants)
├── Include/RL_Indicators.mqh          (helpers — copied)
└── Experts/<deploy_name>_EA.mq5       (customized EA)
```

The EA template's placeholders are replaced:
- `__MODEL_NAME__`     → deploy_name
- `__CONFIG_HEADER__`  → `<deploy_name>_config.mqh`
- `__ONNX_FILE__`      → `<deploy_name>.onnx`

ONNX is embedded into compiled `.ex5` via `#resource` (works in Tester).

---

## 🚚 Deploy to MT5 (3 steps)

### **STEP 1: Locate MT5 data folder**
ใน MT5: **File → Open Data Folder**

จะเปิด: `C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\<ID>\`

### **STEP 2: Copy generated files**

จาก `mt5_files/MQL5/` → ไปที่ `<MT5_Data>/MQL5/`

```
Copy:
  mt5_files/MQL5/Files/<name>.onnx       → <MT5>/MQL5/Files/
  mt5_files/MQL5/Include/<name>_config.mqh → <MT5>/MQL5/Include/
  mt5_files/MQL5/Include/RL_Indicators.mqh → <MT5>/MQL5/Include/
  mt5_files/MQL5/Experts/<name>_EA.mq5   → <MT5>/MQL5/Experts/
```

⚠️ **Note:** การ copy ตามโครงสร้างเดียวกัน (Files→Files, Include→Include, Experts→Experts)

### **STEP 3: Compile**

1. เปิด **MetaEditor** (F4 ใน MT5)
2. Navigate: `Experts/<deploy_name>_EA.mq5`
3. กด **F7** (Compile)
4. Errors tab → ต้อง 0 errors

---

## 🧪 Run Strategy Tester

```
Symbol:    EURUSD (หรือ symbol ที่ train)
Period:    H4   ⭐ ต้องตรงกับ training!
Date:      2021-01-01 → 2026-01-01 (test period)
Modeling:  Every tick based on real ticks
```

EA Inputs (default match training):
```
InpConfidence:        0.95
InpRiskPct:           0.01 (1% per trade)
InpMaxPositions:      3
InpMaxHoldBars:       30
InpATR_SL_Mult:       2.0
InpATR_TP_Mult:       4.0
InpHardDD_Pct:        0.15

# Session Filter (avoid Market closed errors)
InpFilterSession:     true
InpUseSymbolSession:  true   ⭐ ใช้ broker session config (recommended)
InpCheckMarketOpen:   true
InpSkipFridayLate:    true
InpFridayCutoffHour:  21
```

---

## 📋 Expected Results

```
Python backtest (V10, conf 0.95):
  PF:     1.03
  Return: +2.33%
  DD:    -14.67%

MT5 Tester should be CLOSE to:
  PF:     0.95-1.10  (similar to Python)
  Return: 0% to +3%
  DD:     -10% to -18%
```

Differences come from:
- Real broker variable spread (vs fixed 0.0002 in Python)
- Real slippage on market orders
- Order execution timing (tick-level vs bar-level)
- Symbol naming (EURUSD vs EURUSD_TDS)

---

## 🐛 Troubleshooting

### **❌ "file '...' not found" during compile**
- ตรวจ `.mqh` files อยู่ใน `MQL5/Include/` ไม่ใช่ `Experts/`
- Re-copy from `mt5_files/MQL5/Include/`

### **❌ "Failed to load ONNX" at runtime**
- `.onnx` ขาดใน `MQL5/Files/` ตอน COMPILE (ต้องอยู่ตรงนั้นเพื่อ embed ใน .ex5)
- Re-compile หลัง copy `.onnx` ใหม่

### **❌ "Market closed" errors**
- ไม่ใช่ปัญหาของ EA — เป็นการ skip ปกติ ตอน rollover/weekend
- Session filter จัดการให้แล้ว

### **⚠️ ผลต่างจาก Python backtest มาก (>30% gap)**
- ตรวจ Symbol name (EURUSD vs EURUSD_TDS — ใช้ symbol ที่ train)
- ตรวจ Window Size (must match training, default 20)
- ดู Journal log → first features printed match Python

### **⚠️ Bars warmup ไม่ครบ (no trades early)**
- ปกติ — ต้องรอ 20 H4 bars แรก (~3 days)
- Use date range with extra 200+ bars before start_date

---

## 🎓 Architecture Summary

```
PyTorch PPO              MT5 EA                    Strategy Tester
─────────────            ──────────────────       ───────────────────
.zip + norm.csv          On every new H4 bar:     Real broker
       │                  1. Compute 65 features  spread/slippage/swap
       ▼                  2. Normalize
   export_to_              3. Push to buffer (20)
   onnx.py                 4. OnnxRun() → probs
       │                   5. ArgMax + filter
       ▼                  6. Buy/Sell/Close
.onnx + config.mqh        7. Manage SL/TP/hold
       │
       ▼
   Embedded
   in .ex5
   via #resource
```

---

## 🛠️ Updating the EA

ถ้าจะ retrain model + redeploy:

```bash
# 1. Train ใหม่ (rl_app.py)
# 2. Export
python export_to_onnx.py <new_model> --name <new_name>

# 3. Copy new files ไป MT5 (overwrite ของเก่า)
# 4. Re-compile EA in MetaEditor
```

ถ้าใช้ **deploy_name** ใหม่ (เช่น เพิ่ม v11) → ได้ EA ใหม่แยก ไม่ทับของเก่า

---

## ⚠️ Disclaimer

- Use on demo account first
- Start with micro lots (0.01)
- Monitor performance daily
- Re-train model quarterly (concept drift)
- Don't deploy capital you can't afford to lose

---

**Last updated: 2026-05-08**
**Compatible with: rl_prod_v10 (and any V4-reward model exported via this pipeline)**
