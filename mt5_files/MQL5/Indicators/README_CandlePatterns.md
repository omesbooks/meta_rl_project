# 🕯️ CandlePatterns.mq5 — Custom Indicator

ตรวจจับ candlestick patterns 10 รูปแบบและส่งเป็น buffers ให้ EA เรียกใช้

---

## 📊 10 Patterns Detected

| Buffer | Pattern | Encoding | Bars Needed |
|---|---|---|---|
| 0 | Doji | 0 / 1 | 1 |
| 1 | Hammer / Shooting Star | -1 / 0 / +1 | 1 |
| 2 | Engulfing (Bull/Bear) | -1 / 0 / +1 | 2 |
| 3 | Inside Bar | 0 / 1 | 2 |
| 4 | Outside Bar | 0 / 1 | 2 |
| 5 | Star (Morning/Evening) | -1 / 0 / +1 | 3 |
| 6 | Three Soldiers/Crows | -1 / 0 / +1 | 3 |
| 7 | Marubozu | -1 / 0 / +1 | 1 |
| 8 | Harami (Bull/Bear) | -1 / 0 / +1 | 2 |
| 9 | Piercing / Dark Cloud | -1 / 0 / +1 | 2 |

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
double doji[], hammer[], engulfing[];
CopyBuffer(g_h_candles, 0, 1, 1, doji);       // BufDoji[1]
CopyBuffer(g_h_candles, 1, 1, 1, hammer);     // BufHammer[1]
CopyBuffer(g_h_candles, 2, 1, 1, engulfing);  // BufEngulfing[1]

// Use as features for RL model:
features[N+0] = doji[0];
features[N+1] = hammer[0];
features[N+2] = engulfing[0];
// ...
```

Buffer index reference (สำหรับ `CopyBuffer`):
```
0: Doji          5: Star (Morning/Evening)
1: Hammer        6: Soldiers/Crows
2: Engulfing     7: Marubozu
3: Inside        8: Harami
4: Outside       9: Piercing/Dark Cloud
```

---

## ⚙️ Inputs

### **Pattern Toggles (เปิด/ปิดทีละแบบ)**
```
InpEnableDoji         = true     [0]
InpEnableHammer       = true     [1] Hammer / Shooting Star
InpEnableEngulfing    = true     [2]
InpEnableInsideBar    = true     [3]
InpEnableOutsideBar   = true     [4]
InpEnableStar         = true     [5] Morning / Evening Star
InpEnableSoldiers     = true     [6] Three Soldiers / Crows
InpEnableMarubozu     = true     [7]
InpEnableHarami       = true     [8]
InpEnablePiercing     = true     [9] Piercing / Dark Cloud
```
> ⚠️ ปิดบาง pattern → buffer นั้นเป็น 0 ตลอด (ไม่ทำ detection)

### **Detection Thresholds**
```
InpDojiThreshold     = 0.10   // body < 10% × range = Doji
InpMarubozuThreshold = 0.95   // body > 95% × range = Marubozu
InpHammerWickRatio   = 2.0    // wick > 2× body = Hammer
InpHammerBodyRatio   = 0.30   // body < 30% × range = Hammer-eligible
InpEngulfingMinRatio = 2.0    // Engulfing: cur body ≥ 200% of prev body
                              //   1.0 = ≥ same size (loose, แค่ engulf)
                              //   2.0 = ≥ 2× larger ⭐ default (recommended)
                              //   3.0 = ≥ 3× larger (strict, เฉพาะ strong reversal)
```

### **Visual Markers**
```
InpDrawArrows      = true    // Draw arrow markers on chart
InpDrawDojiArrow   = false   // Doji is too common — skip arrow by default
```

---

## 📊 Data Window Integration

ทุก buffer แสดงใน **Data Window** ตามชื่อ pattern:

```
Data Window:
  Doji         : 0 หรือ 1
  Hammer       : -1, 0, +1
  Engulfing    : -1, 0, +1
  InsideBar    : 0 หรือ 1
  OutsideBar   : 0 หรือ 1
  Star         : -1, 0, +1
  Soldiers     : -1, 0, +1
  Marubozu     : -1, 0, +1
  Harami       : -1, 0, +1
  Piercing     : -1, 0, +1
```

→ ใช้สำหรับ debug หรือดูค่าเป็น bar-by-bar ก่อนเอาไปใช้เป็น feature

---

## 🔬 Pattern Definitions

### **Doji** (indecision)
Body ≤ 10% ของ range → market ลังเล

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
