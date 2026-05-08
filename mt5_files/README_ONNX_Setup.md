# 🚀 ML_RL_Trader — ONNX Setup Guide

วิธี deploy V10 RL model ไปทดสอบใน MT5 Strategy Tester ผ่าน ONNX

---

## 📋 Prerequisites

```
✅ MT5 build 3550+ (รองรับ ONNX) — เช็คใน Help → About
✅ Python 3.8+ พร้อม stable_baselines3, torch, onnx, onnxruntime, onnxscript
✅ Trained PPO model (.zip + _norm.csv + _best/)
✅ MetaEditor (มากับ MT5)
```

---

## 🗂️ ไฟล์ที่จะใช้

```
Project root/
├── export_to_onnx.py           ← Python script: convert PPO → ONNX
├── rl_prod_v10_enriched_best/  ← trained model
├── rl_prod_v10_enriched_norm.csv
└── mt5_files/
    ├── ML_RL_Trader.mq5        ← Main EA
    ├── RL_Indicators.mqh       ← Feature computation helpers
    └── README_ONNX_Setup.md    ← (this file)
```

---

## 🎯 Quick Setup (5 ขั้นตอน)

### **STEP 1: Export model → ONNX (Python)**

```bash
cd <project_root>
python export_to_onnx.py rl_prod_v10_enriched
```

**ผลลัพธ์:**
```
rl_prod_v10_enriched.onnx          (2.7 KB)  ← Model weights
rl_prod_v10_enriched_config.mqh    (5.9 KB)  ← Norm constants for EA
```

✅ Verification: script จะตรวจว่า ONNX output match PyTorch (max diff < 1e-5)

---

### **STEP 2: หา MT5 data folder**

ใน MT5: **File → Open Data Folder**

จะเปิด path เช่น:
```
C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\<ID>\
```

---

### **STEP 3: Copy ไฟล์ไปที่ถูกที่**

```
<MT5_Data>/MQL5/Files/
   └─ rl_prod_v10_enriched.onnx                  ← จาก STEP 1

<MT5_Data>/MQL5/Include/
   └─ rl_prod_v10_enriched_config.mqh            ← จาก STEP 1
   └─ RL_Indicators.mqh                           ← จาก mt5_files/

<MT5_Data>/MQL5/Experts/
   └─ ML_RL_Trader.mq5                            ← จาก mt5_files/
```

⚠️ **สำคัญ:** ต้องตรงตำแหน่งเป๊ะ ไม่งั้น `#include` จะ error

---

### **STEP 4: Compile ใน MetaEditor**

1. เปิด MetaEditor (F4 ใน MT5)
2. Navigate: `Experts → ML_RL_Trader.mq5`
3. กด **F7** (Compile)
4. ดู Errors tab — ต้อง **0 errors**

ถ้ามี error เกี่ยวกับ `#include`:
- เช็คว่าไฟล์ `.mqh` อยู่ใน `Include/` ตรงๆ
- ลอง close + reopen MetaEditor

---

### **STEP 5: รันใน Strategy Tester**

1. ใน MT5: **View → Strategy Tester** (Ctrl+R)
2. ตั้งค่า:
   ```
   Expert:    ML_RL_Trader
   Symbol:    EURUSD (หรือ symbol ที่ train)
   Period:    H4  ⭐ ต้องตรงกับที่ train!
   Date:      2021-01-01 → 2026-01-01 (test period)
   Modeling:  Every tick based on real ticks (recommended)
   Optimization: Disabled (เริ่มต้น)
   ```

3. กด **Settings**:
   - Confidence:  **0.95** (match Python backtest)
   - Risk %:       1.0
   - Max Pos:      3
   - Max Hold:     30
   - ATR×SL:       2.0
   - ATR×TP:       4.0
   - Hard DD %:    0.15

4. กด **Start**

---

## 📊 Expected Results

### **คาดว่าจะใกล้เคียง Python backtest:**
```
Python backtest (V10, conf 0.95):
  PF:     1.03
  Return: +2.33%
  DD:    -14.67%

MT5 Tester (expected):
  PF:     0.95-1.10  (≈ Python)
  Return: 0% to +3%
  DD:     -10% to -18%
```

### **ทำไมอาจต่างกัน:**
- 🔸 Real broker spread (variable, vs fixed 0.0002 in Python)
- 🔸 Real slippage on market orders
- 🔸 Order execution timing (Python instant vs MT5 next tick)
- 🔸 Symbol naming (EURUSD vs EURUSD_TDS — model trained on what?)
- 🔸 Bar synchronization (D1 features depend on D1 timeframe data)

### **🟢 ถ้าผลใกล้เคียง (gap < 30%) = SUCCESS!**
Python backtest validated by real broker simulation

### **🔴 ถ้าผลต่างมาก:**
ตรวจ:
1. Symbol name (EURUSD vs EURUSD_TDS)
2. Spread differences (real vs simulated)
3. ดู journal log ว่า EA load ONNX สำเร็จไหม

---

## 🔍 Verification Checklist

ก่อนรัน Tester ให้เช็ค:

```
☐ MT5 build 3550+ (Help → About)
☐ ONNX file in Files/, MQH in Include/, EA in Experts/
☐ Compile success (0 errors, 0 warnings)
☐ Test on H4 timeframe
☐ Symbol matches training (likely EURUSD)
☐ Window=20 in EA (matches model.observation_space)
☐ Confidence=0.95 (matches Python backtest)
☐ Date range matches Python test period
```

---

## 🐛 Troubleshooting

### **❌ "Failed to load ONNX"**
- ไฟล์ไม่อยู่ใน `MQL5/Files/` ให้ check path
- ONNX corrupted → re-export
- MT5 build เก่า → upgrade

### **❌ "OnnxRun failed"**
- Input shape mismatch — ตรวจว่า config.mqh ตรงกับ model
- Re-export ใหม่

### **❌ "Failed to create RSI/EMA handle"**
- Indicator periods invalid for symbol/timeframe
- ตรวจ Symbol สนับสนุน historical data หรือไม่

### **⚠️ Bars warmup ไม่ครบ (no trades)**
- ปกติ — ต้องรอ 20 bars แรกก่อน
- ใน Strategy Tester ให้ใช้ data range ก่อน start_date 200 bars

### **⚠️ Predictions ทุกบาร์เป็น Hold (action=0)**
- Confidence < 0.95 ทุกบาร์ → ลอง print probs
- หรือ features ผิด → compare กับ Python output

### **⚠️ MT5 Tester ผลต่างจาก Python มาก**

**Common causes:**
1. **Symbol mismatch** — Python trained on `EURUSD_TDS`, Tester ใช้ `EURUSD`
2. **D1 alignment** — D1 features ใน MT5 อาจต่างจาก Python aggregation
3. **Spread** — Tester ใช้ "Real Spread" mode → ผลแย่กว่า Python ที่ฟิกซ์
4. **Missing data** — H4 data ไม่ครบทำให้ indicators ไม่ valid

**Debug steps:**
```
1. ตั้ง EA ให้ print features ใน OnTick (debug mode)
2. Run 1 bar → copy printed values
3. Compare กับ Python output ของบาร์เดียวกัน
4. ถ้า diff > 5% ใน feature ใดๆ → fix MQL5 computation
```

---

## 🎓 Architecture Notes

### **Feature mapping** (ใน RL_Indicators.mqh):

```
[0-3]   RSI(4..30) aggregate   ← min/max/mean/std of 27 RSI values
[4-7]   EMA(20,50,100,200)     ← MULTI mode, 4 periods
[8-11]  ATR(5..50) aggregate   ← 46 ATR periods
[12-15] Stoch(5..21) aggregate ← 17 Stoch periods (%K only)
[16-19] CCI(5..30) aggregate
[20-23] WPR(5..30) aggregate
[24-27] ADX(7..30) aggregate
[28]    EMA(200) (ema_long)
[29-31] MACD(12,26,9)
[32]    BB position
[33-37] Time (hour, dow, sessions)
[38-42] Returns (1,3,5,10,20)
[43-45] Statistical (zscore, rank, sharpe20)
[46-47] Bar shape
[48-55] D1 multi-TF (rsi, ema_fast/slow, trend, atr, atr_pct, adx, alignment)
[56-60] Volatility regime
[61-64] Range/distance
```

### **Inference Pipeline**:

```
H4 bar closes
    ↓
Compute 65 features (mix of MT5 indicators + custom calcs)
    ↓
Normalize: (x - mean) / std  using config.mqh constants
    ↓
Push into circular buffer [20 × 65]
    ↓
After 20 bars (warm-up):
    Build state vector [1303 floats]:
        - 20 bars × 65 features (oldest → newest)
        - + position_side, unrealized_pnl, bars_in_position
    ↓
OnnxRun(handle, state) → 4 action probabilities
    ↓
action = argmax(probs)
confidence = probs[action]
    ↓
If conf >= 0.95 → execute Buy/Sell/Close
```

---

## 📈 Performance

```
Inference time: ~1-3 ms per bar
Total runtime for 5-year H4 test (~7,500 bars): ~30 sec
Indicator computation: ~5-10 ms per bar (heavy due to many handles)
Total: ~5-7 minutes for 5-year H4 backtest
```

---

## ⚙️ Customization

### **เปลี่ยน confidence:**
แก้ใน EA inputs: `InpConfidence`

### **ใช้กับ model อื่น:**
1. Run `python export_to_onnx.py <new_model>`
2. Copy `_config.mqh` to Include/
3. แก้ `#include` ใน ML_RL_Trader.mq5
4. Re-compile

### **เพิ่ม/แก้ features:**
- แก้ใน `RL_Indicators.mqh`
- ต้อง MATCH กับ Python feature_engineer.py
- Re-compile

---

## 🎯 Next Steps After Successful Tester Run

1. **Forward test** — ใช้ EA บน demo account 1 เดือน
2. **Compare metrics** — ดูว่า live PF ≈ test PF ไหม
3. **Adjust confidence** — fine-tune in real conditions
4. **Set up monitoring** — alert เมื่อ DD spike
5. **Scheduled retrain** — quarterly retrain pipeline

---

## ⚠️ Disclaimer

This is research/experimental code. Trading involves substantial risk.
- ✅ Use on demo account first
- ✅ Start with micro lots (0.01)
- ✅ Monitor daily
- ✅ Have a kill switch ready
- ❌ Don't deploy with > 1% risk per trade
- ❌ Don't use leverage > 1:30

---

## 🔗 Related Files

- `trading_env.py` — Training environment (Python)
- `backtest_live.py` — Production-grade Python backtest
- `live_trader.py` — Live MT5 trader (Python via MetaTrader5 package)
- `feature_engineer.py` — Multi-TF feature pipeline
- `DataCollector_v3.mq5` — Original feature source (MT5 EA)

---

**Created: 2026-05-08**
**Compatible with: rl_prod_v10_enriched (and any V4 reward model)**
