# 🕯️ CandlePatterns.mq5 — Custom Indicator

ตรวจจับ candlestick patterns 9 รูปแบบและส่งเป็น buffers ให้ EA เรียกใช้

---

## 📊 9 Patterns Detected

| Buffer | Pattern | Encoding | Bars Needed |
|---|---|---|---|
| 0 | Hammer / Shooting Star | -1 / 0 / +1 | 1 |
| 1 | Engulfing (Bull/Bear) | -1 / 0 / +1 | 2 |
| 2 | Inside Bar | 0 / 1 | 2 |
| 3 | Outside Bar | 0 / 1 | 2 |
| **4** | **Morning / Evening Star** ⭐ | -1 / 0 / +1 | 3 |
| 5 | Three Soldiers/Crows | -1 / 0 / +1 | 3 |
| 6 | Marubozu | -1 / 0 / +1 | 1 |
| 7 | Harami (Bull/Bear) | -1 / 0 / +1 | 2 |
| 8 | Piercing / Dark Cloud | -1 / 0 / +1 | 2 |

> ℹ️ **Doji removed in v1.20** — ค่อนข้างซ้ำซ้อนกับ Hammer (small body + wicks).
>    Star pattern อยู่ที่ buffer [4] = Morning Star (+1) / Evening Star (-1).

**Encoding convention:**
- `+1` = Bullish pattern (potential upward signal)
- `-1` = Bearish pattern (potential downward signal)
- `0`  = No pattern detected

---

## 🚀 Installation

```
mt5_files/MQL5/Indicators/CandlePatterns.mq5
   ↓ copy to:
<MT5_Data>/MQL5/Indicators/CandlePatterns.mq5

แล้ว: MetaEditor → F7 → compile
```

---

## 🎯 Two Usage Modes

### **Mode A: Standalone Visual Indicator**

Drag onto chart → เห็น arrow markers บนแท่งที่มี pattern

- Green ↑ = Bullish pattern below bar
- Red ↓ = Bearish pattern above bar
- Tooltip: ชื่อ pattern (Star, Engulfing, Hammer, etc.)

### **Mode B: Programmatic Use (จาก EA)**

```mql5
// In your EA's OnInit():
int g_h_candles = iCustom(_Symbol, _Period, "CandlePatterns");
if(g_h_candles == INVALID_HANDLE) {
    Print("Failed to load CandlePatterns indicator");
    return INIT_FAILED;
}

// In OnTick or feature builder:
double hammer[], engulfing[], star[];
CopyBuffer(g_h_candles, 0, 1, 1, hammer);     // BufHammer[1]
CopyBuffer(g_h_candles, 1, 1, 1, engulfing);  // BufEngulfing[1]
CopyBuffer(g_h_candles, 4, 1, 1, star);       // BufStar[1] — Morning/Evening

// Use as features for RL model:
features[N+0] = hammer[0];
features[N+1] = engulfing[0];
features[N+2] = star[0];
// ...
```

Buffer index reference (สำหรับ `CopyBuffer`):
```
0: Hammer        5: Soldiers/Crows
1: Engulfing     6: Marubozu
2: InsideBar     7: Harami
3: OutsideBar    8: Piercing/Dark Cloud
4: Star (Morning/Evening) ⭐
```

---

## ⚙️ Inputs

### **Pattern Toggles (เปิด/ปิดทีละแบบ)**
```
InpEnableHammer       = true     [0] Hammer / Shooting Star
InpEnableEngulfing    = true     [1]
InpEnableInsideBar    = true     [2]
InpEnableOutsideBar   = true     [3]
InpEnableStar         = true     [4] Morning / Evening Star ⭐
InpEnableSoldiers     = true     [5] Three Soldiers / Crows
InpEnableMarubozu     = true     [6]
InpEnableHarami       = true     [7]
InpEnablePiercing     = true     [8] Piercing / Dark Cloud
```
> ⚠️ ปิดบาง pattern → buffer นั้นเป็น 0 ตลอด (ไม่ทำ detection)

### **Detection Thresholds**
```
InpMarubozuThreshold    = 0.95   // body > 95% × range = Marubozu

# Hammer / Shooting Star (3 filters):
InpHammerWickRatio      = 2.0    // long wick ≥ 2× body
InpHammerBodyMaxPct     = 0.30   // body ≤ 30% × range
InpHammerOppWickMaxPct  = 0.10   // opposite wick ≤ 10% × range

# Engulfing:
InpEngulfingMinRatio    = 2.0    // cur body ≥ 200% of prev body
                                  //   1.0 = ≥ same size (loose)
                                  //   2.0 = ≥ 2× larger ⭐ default
                                  //   3.0 = ≥ 3× larger (strict)

# Morning / Evening Star (3-bar reversal):
InpStarMidBodyMaxPct    = 0.40   // middle bar body ≤ 40% × middle range
InpStarOuterBodyMinPct  = 0.70   // outer bars body ≥ 70% × middle range
                                  //   ค่าน้อย = หลวม, ค่ามาก = strict
```

### **Morning / Evening Star Detection**

```
Morning Star (Bullish reversal at downtrend):
   Bar 2 (oldest):    Bar 1 (middle/star):  Bar 0 (newest):
   ┌──┐                                      ┌──┐
   │  │       Big bear        Small body     │  │       Big bull
   │  │       body ≥ 70%      ≤ 40% range    │  │       body ≥ 70%
   │  │                       (gap down)     │  │       closes > bar2 mid
   │  │           ↓                ↓         │  │           ↓
   │  │         body0          body1         │  │         body2
   │  │                                      │  │
   └──┘                                      └──┘
   bear           gap down star            big bull
                                            confirms reversal

Evening Star = inverse (bullish trend → small body → big bear closes < mid)
```

### **Hammer / Shooting Star Tuning**

```
ตัวอย่างแท่งสมบูรณ์แบบ (Hammer):
  ─────────  ← high
       │
       │     ← upper_wick (ฝั่งตรงข้าม / opposite)  ≤ 10% × range
       │
   ┌───┐    ← body (≤ 30% × range)
   │   │
   └───┘
       │
       │     ← lower_wick (ฝั่งหลัก / dominant) ≥ 2× body
       │
       │
       │
  ─────────  ← low

3 filters ที่ต้องผ่านพร้อมกัน:
  1. body ≤ InpHammerBodyMaxPct × range       (เนื้อแท่งเล็ก)
  2. dominant wick ≥ InpHammerWickRatio × body (หางยาว)
  3. opposite wick ≤ InpHammerOppWickMaxPct × range (หางตรงข้ามสั้น)
```

### **Strict vs Loose Hammer Settings**

| Setting | Body % | Opp Wick % | Wick Ratio | Result |
|---|---|---|---|---|
| Loose | 0.40 | 0.20 | 1.5× | many detections |
| **Default** ⭐ | **0.30** | **0.10** | **2.0×** | balanced |
| Strict | 0.20 | 0.05 | 3.0× | rare, high-conviction |
| Pure Pin Bar | 0.15 | 0.03 | 4.0× | textbook only |

### **Visual Markers**
```
InpDrawArrows      = true    // Draw arrow markers on chart
```

---

## 📊 Data Window Integration

ทุก buffer แสดงใน **Data Window** ตามชื่อ pattern:

```
Data Window:
  Hammer       : -1, 0, +1
  Engulfing    : -1, 0, +1
  InsideBar    : 0 หรือ 1
  OutsideBar   : 0 หรือ 1
  Star         : -1, 0, +1   ← Morning(+1) / Evening(-1) Star
  Soldiers     : -1, 0, +1
  Marubozu     : -1, 0, +1
  Harami       : -1, 0, +1
  Piercing     : -1, 0, +1
```

→ ใช้สำหรับ debug หรือดูค่าเป็น bar-by-bar ก่อนเอาไปใช้เป็น feature

---

## 🔬 Pattern Definitions

### **Hammer** (bullish reversal at downtrend)
- Small body in upper half
- Long lower wick (≥ 2× body)
- No/tiny upper wick

### **Shooting Star** (bearish reversal at uptrend)
- Inverse of hammer

### **Marubozu** (strong continuation)
- Body fills ≥ 95% of range
- No or tiny wicks

### **Engulfing** (reversal)
- **Bullish**: prev bear, current bull body fully engulfs prev body
- **Bearish**: prev bull, current bear body fully engulfs prev body

### **Inside Bar** (compression — pre-breakout)
- Current high < prev high AND current low > prev low

### **Outside Bar** (volatility expansion)
- Current high > prev high AND current low < prev low

### **Star** (3-bar reversal)
- **Morning Star**: big bear → small body → big bull
- **Evening Star**: big bull → small body → big bear

### **Three Soldiers / Crows**
- 3 consecutive same-direction closes, each higher (bull) or lower (bear)

### **Harami** (small inside large body, opposite color)
- Indicates trend exhaustion

### **Piercing / Dark Cloud Cover**
- 2-bar reversal where current bar pierces past midpoint of previous

---

## 📈 Adding to RL Feature Pipeline

To use these as features for the RL model:

1. **Compile indicator** in MetaEditor (F7)
2. **In RL_Indicators.mqh** (your EA's helper):
   ```mql5
   int g_h_candles = INVALID_HANDLE;
   
   // In RL_InitIndicators():
   g_h_candles = iCustom(symbol, tf, "CandlePatterns");
   
   // In RL_BuildFeatures(), add 10 new feature slots:
   double cp_buf[];
   for(int i = 0; i < 10; i++) {
       CopyBuffer(g_h_candles, i, shift, 1, cp_buf);
       features[N + i] = cp_buf[0];
   }
   ```
3. **Re-train the model** with these new features included
4. **Re-export ONNX** with updated norm.csv

---

## ⚠️ Notes

- **Pattern detection is approximate** — exact criteria can vary by source
- Adjust thresholds via inputs to match your strategy
- Three-bar patterns need warmup (start from bar index 2)
- Visual arrows can be disabled (`InpDrawArrows = false`) for performance

---

## 🎯 Feature Importance Estimate

จากประสบการณ์ trading literature:
- **Strongest signal**: Engulfing, Star, Three Soldiers (rare but reliable)
- **Medium**: Hammer, Shooting Star, Marubozu
- **Weak**: Doji, Inside Bar (need context — trend, location)

Best used **combined with trend filter** (e.g., only act on Bullish Engulfing in uptrend).

---

**Created: 2026-05-08**
