# 📊 DataCollector_v4.mq5 — v3 + Candle Patterns

Same as v3 (range-based features) **+ 10 candlestick pattern features**

---

## 🆕 What's New vs v3

```
v3 produces 50 features per row.
v4 produces 60 features per row (+10 candle patterns).
```

### **10 New Columns (added before future_return)**

```
candle_hammer       -1 (shooting star) / 0 / +1 (hammer)
candle_engulfing    -1 / 0 / +1
candle_inside        0 / 1
candle_outside       0 / 1
candle_star         -1 (evening) / 0 / +1 (morning)
candle_soldiers     -1 (3 crows) / 0 / +1 (3 white)
candle_marubozu     -1 / 0 / +1
candle_harami       -1 / 0 / +1
candle_piercing     -1 (dark cloud) / 0 / +1 (piercing)
candle_mathold      -1 (bear) / 0 / +1 (bull)  ⭐ 5-bar continuation
```

---

## 📦 Prerequisites

```
1. CandlePatterns.mq5 must be COMPILED in:
   <MT5>/MQL5/Indicators/CandlePatterns.ex5
   
2. DataCollector_v4.mq5 placed in:
   <MT5>/MQL5/Experts/DataCollector_v4.mq5
```

---

## 🚀 Setup Steps

### **STEP 1: Deploy CandlePatterns indicator**
```
Copy: mt5_files/MQL5/Indicators/CandlePatterns.mq5
  →  <MT5>/MQL5/Indicators/CandlePatterns.mq5

MetaEditor: F7 → compile → CandlePatterns.ex5
```

### **STEP 2: Deploy DataCollector_v4**
```
Copy: mt5_files/MQL5/Experts/DataCollector_v4.mq5
  →  <MT5>/MQL5/Experts/DataCollector_v4.mq5

MetaEditor: F7 → compile → DataCollector_v4.ex5
```

### **STEP 3: Run in Strategy Tester**
```
Symbol:    EURUSD (or any)
Timeframe: H4 (or H1, etc.)
Modeling:  Every tick (slowest but accurate)
Period:    long range — e.g., 2003-2026

EA Inputs:
  FileName:           training_data_v4_h4.csv
  RSI_Mode:           AGGREGATE
  EMA_Mode:           MULTI
  ... (same as v3 defaults)

  CP_EnableHammer:    true
  CP_EnableEngulfing: true
  ... (all 10 toggles default true)
```

### **STEP 4: Find output**

```
<MT5>/MQL5/Files/training_data_v4_h4.csv
```

Use the existing import tool in `rl_app.py` Tools page to convert UTF-16 → UTF-8.

---

## ⚙️ All Inputs (extended from v3)

### **Standard Indicator Settings** (same as v3)
```
RSI_Mode, EMA_Mode, ATR_Mode, Stoch_Mode, CCI_Mode, WPR_Mode, ADX_Mode
RSI_Min/Max/Step, EMA_Min/Max/Step, ATR_Min/Max/Step ...
BB_Period, MACD_Fast/Slow/Signal
Stat_Window, Rank_Window, MinBarsRequired
ForwardBars, UpThreshold, DownThreshold
```

### **NEW: Candle Patterns Toggles**
```
CP_EnableHammer        = true
CP_EnableEngulfing     = true
CP_EnableInsideBar     = true
CP_EnableOutsideBar    = true
CP_EnableStar          = true
CP_EnableSoldiers      = true
CP_EnableMarubozu      = true
CP_EnableHarami        = true
CP_EnablePiercing      = true
CP_EnableMatHold       = true
```

### **NEW: Candle Patterns Thresholds**
```
CP_MarubozuThreshold    = 0.95
CP_HammerWickRatio      = 2.0
CP_HammerBodyMaxPct     = 0.30
CP_HammerOppWickMaxPct  = 0.10
CP_EngulfingMinRatio    = 2.0
CP_StarMidBodyMaxPct    = 0.40
CP_StarOuterBodyMinPct  = 0.70
CP_MatHoldOuterBodyMin  = 0.60
CP_MatHoldMidBodyMax    = 0.40
CP_MatHoldRequireBreak  = true
CP_InsideOutsideStrict  = true
CP_PiercingMinBodyRatio = 0.5
```

→ ทุก threshold สามารถปรับได้จาก Strategy Tester input panel

---

## 🧬 Architecture

```
Strategy Tester Bar by Bar:
        │
        ▼
┌──────────────────────────────────────────────┐
│ DataCollector_v4 OnTick                       │
│                                                │
│ 1. Read OHLCV                                  │
│ 2. Get values from 7 indicator handles        │
│    (RSI, EMA, ATR, Stoch, CCI, WPR, ADX)      │
│ 3. Get values from BB, MACD, EMA_long         │
│ 4. Compute statistical (zscore, sharpe, rank) │
│ 5. ⭐ Get 10 patterns from CandlePatterns      │
│    via CopyBuffer(g_h_candles, idx, ...)      │
│ 6. Compute label (future_return, target)      │
│ 7. Write CSV row                               │
└──────────────────────────────────────────────┘
        │
        ▼
training_data_v4.csv (60 features per row + label)
```

---

## 🔍 CSV Column Order

```
[1-7]   timestamp, symbol, OHLCV
[8-N]   Indicator features (RSI/EMA/ATR/Stoch/CCI/WPR/ADX aggregates)
        + ema_long, MACD x3, bb_position
[N+1-5] hour, dow, sessions x3
[N+6-10] returns x5 (1, 3, 5, 10, 20)
[N+11-15] zscore, pct_rank, sharpe_20, hl_range, body_size
[N+16-25] candle_hammer, ..., candle_mathold ⭐ NEW
[Last-1] future_return
[Last]   target (UP / DOWN / FLAT)
```

Total columns depend on indicator modes. Default config produces ~67 columns
including 10 candle patterns + 2 label.

---

## 🐛 Troubleshooting

### **❌ "Failed to load CandlePatterns indicator"**
- ตรวจ CandlePatterns.mq5 compiled แล้ว (มี .ex5 file)
- ตรวจอยู่ใน MQL5/Indicators/ ไม่ใช่ Experts/

### **❌ Pattern values ทั้งหมดเป็น 0**
- ดู `CP_Enable*` toggles — ปิดอยู่ไหม?
- ตรวจ thresholds ไม่ strict เกินไป (เช่น CP_HammerOppWickMaxPct = 0.01)
- ลด threshold ดู → pattern ยังตรวจไม่เจอเลย = อาจ symbol ผันผวนน้อย

### **⚠️ Bars take long to fill**
- v4 ต้องการ MinBarsRequired (default 250) + max_period
- 5-bar Mat Hold ต้องการ ≥ 4 bars ก่อนจึงจะ detect ได้

### **⚠️ Candle pattern column = 0 always**
- เช็ค Data Window ของ CandlePatterns indicator
- ถ้า indicator ไม่เห็น pattern → datacollector ก็จะเห็นเป็น 0

---

## 🎯 Use Case: Train Model with Candle Features

```
1. Run DataCollector_v4 in Tester → training_data_v4_h4.csv
2. Import via rl_app.py → clean_train.csv
3. Train model with 60 features (instead of 50)
4. Compare:
   - V10 model (50 features, no candles)  — PF ~1.03
   - V11 model (60 features, with candles) — expected PF improvement?

Hypothesis: Candle patterns add direction signal that's
            independent from oscillators → may help
```

---

## 📊 Expected Pattern Frequency

```
On EURUSD H4 (7 years ~ 10K bars):
  
  Pattern         Detections (approx)   Per-day rate
  ─────────────────────────────────────────────────
  Engulfing       ~150-300              0.06-0.12
  Hammer/Star     ~200-400              0.08-0.15
  Inside Bar      ~500-800              0.20-0.30
  Outside Bar     ~400-600              0.15-0.22
  Star (3-bar)    ~30-80                0.01-0.03
  Soldiers/Crows  ~50-150               0.02-0.06
  Marubozu        ~300-500              0.10-0.20
  Harami          ~100-200              0.04-0.08
  Piercing/DCC    ~80-150               0.03-0.06
  Mat Hold        ~10-30                0.004-0.01  (rare!)
```

→ Mat Hold rare but high-value when seen.

---

## 🔄 Compatibility

```
v3 columns ✓ — all preserved
v4 adds 10 columns BEFORE future_return
→ Models trained on v3 won't work on v4 directly (different column count)
→ Need to retrain from scratch with v4 data
```

---

**Created: 2026-05-08**
**Compatible with: CandlePatterns v1.40+**
