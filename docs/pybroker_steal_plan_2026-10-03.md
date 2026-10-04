# แผนขโมยฟีเจอร์จาก PyBroker มาใส่ Metafxclub RL Studio

> วันที่: 2026-10-03
> อ้างอิง: https://www.pybroker.com/en/latest/ (BCa bootstrap, EvalMetrics, walkforward)
> สถานะ: เฟส 1-2 เสร็จแล้ว (2026-10-04) · เฟส 3-4 ยังไม่เริ่ม

---

## สรุปผู้บริหาร

PyBroker = framework วิจัย ML สำหรับหุ้น/คริปโต (ไม่มี RL, ไม่มี live execution)
ของเรา = ระบบ RL → MT5 ครบวงจร **ไม่ต้องย้ายไป** หยิบเฉพาะชิ้นที่เขาทำดีกว่ามาใส่

**4 เฟส เรียงตามคุ้มค่า:**

| เฟส | เรื่อง | ต้นทุน | ผลที่ได้ |
|---|---|---|---|
| 1 | Bootstrap CI (PF/Sharpe/DD) | ต่ำ | แก้ปัญหา significance ที่ค้างอยู่ |
| 2 | เมตริกเสริม 6 ตัว | ต่ำ | ตัดสินใจ "กล้าลงเงินจริงมั้ย" ได้แม่นขึ้น |
| 3 | ลด obs dim + ทดสอบ GPU | กลาง | เทรนเร็วขึ้น 2-4x (วัดแล้วว่าคอขวดอยู่ตรงนี้) |
| 4 | Multi-symbol portfolio backtest | สูง | สะสม trades ให้ถึง significance + กระจายความเสี่ยง |

---

## ผลวัดคอขวดจริง (2026-10-03) — ข้อมูลที่เปลี่ยนแผน

วัดบน `uj_h4_extra_features_clean.csv` (8,493 แถว / 122 features / window 30 → obs dim 3,663)

| รายการ | ค่า |
|---|---|
| `env.step()` ล้วน | **36,424 steps/s** |
| PPO เต็ม (env + torch) | **151 steps/s** |
| → env กินเวลา | **0.4%** ของเวลาเทรนทั้งหมด |
| → torch กิน | **99.6%** |
| แยกใน torch: rollout (forward batch=1) | 44% |
| แยกใน torch: update (20 epochs) | 56% |
| 4 envs ขนาน | เร็วขึ้นแค่ **1.18x** |
| 8 envs ขนาน | เร็วขึ้น **1.02x** (CPU 14 threads อิ่มแล้ว) |
| torch build ปัจจุบัน | `2.11.0+cpu` — **cuda_available = False** |
| 400k steps | ≈ 32 นาที |

**ข้อสรุป:** Numba เร่ง env **ไม่คุ้มเลย** (เพดานการเร่ง = 1.00x) — ตัดทิ้งจากแผน
ตัวการจริงคือ **ขนาด observation**: layer แรก 3,663 × 256 = 937k params = **~95% ของพารามิเตอร์ทั้งโมเดล**
→ ลด obs ครึ่งหนึ่ง ≈ เทรนเร็วขึ้นเกือบเท่าตัว โดยไม่ต้องแตะโค้ด torch เลย

---

## เฟส 1 — Bootstrap Confidence Interval ⭐ เริ่มที่นี่

### ทำไม
ปัญหาที่ค้างมานาน: `rl_uj_extra` ได้ p=0.10 ที่ 138 เทรด, PSR 91.2% — ตัดสินไม่ขาดว่ามี edge จริงไหม
PyBroker ใช้ **BCa bootstrap** (bias-corrected & accelerated) สุ่ม resample ผลเทรดพันรอบ
แล้วรายงานเป็น **ช่วงความเชื่อมั่น** ซึ่งอ่านง่ายและตอบตรงกว่า p-value

ผลจริงบน `rl_uj_extra` best checkpoint (OOS `--start 0.85`, 138 เทรด):
```
Profit Factor : 1.42   90% CI [0.87, 2.17]   95% CI [0.82, 2.36]
Sharpe/trade  : 0.110  95% CI [-0.041, 0.263]
Max DD        : 90% worst -5.92% | 95% worst -6.93% | 99% worst -9.00%
block-5 check : PF 95% low 0.82 (เท่ากับ iid — เทรดไม่เป็นสตรีค)
verdict       : CI คร่อม 1.0 — ยังตัดความเป็นไปได้ว่าไม่มี edge ไม่ได้
```
**เกณฑ์ตัดสินใหม่ที่ชัดกว่าเดิม:** ขอบล่างของ PF 95% CI > 1.0 → edge ผ่าน

### แก้ที่ไหน
| ไฟล์ | จุด | ทำอะไร |
|---|---|---|
| `backtest_live.py` | ~L1101 (หลังบล็อก significance) | เพิ่มฟังก์ชัน `_bca_bootstrap()` + คำนวณ CI |
| `backtest_live.py` | ~L1142 `sig_result` dict | เพิ่ม key `bootstrap` |
| `rl_app.py` | L7586 (`res.get("significance")`) | แสดง CI ต่อท้าย PSR |

### รายละเอียดเทคนิค
- วัตถุดิบมีครบแล้ว: `fracs_all` (array ผลเทรดรายไม้) ที่ L1101
- BCa = resample ผลเทรด N รอบ → คำนวณ metric แต่ละรอบ → แก้ bias (z0) + acceleration (jackknife)
- จำนวน resample: 1,000 (เท่า MC เดิม) — ใช้ `np.random.default_rng(seed)` ให้ reproducible
- Metrics ที่ทำ CI: **log(Profit Factor)** (PyBroker ใช้ log เพราะกระจายตัวดีกว่า), **Sharpe/trade**, **Max DD**
- ต้นทุน runtime: ~1-2 วินาที (1000 × 138 ตัวเลข) ไม่ต้องกลัว

### ความเสี่ยง
- ต่ำ — เป็นการ "เพิ่ม" เมตริก ไม่แตะ logic การเทรด
- ข้อควรระวัง: bootstrap บนเทรดที่ **เรียงเวลา** มี autocorrelation — ถ้าผลเทรดไม่อิสระ CI จะแคบเกินจริง
  → ทำ **block bootstrap** (สุ่มเป็นก้อนละ 5 เทรด) เป็น option เสริมไว้เทียบ

### ยืนยันผล
- รันบน `rl_uj_extra` (138 เทรด, PF 1.42) → CI ควรกว้างและคร่อม 1.0 (สอดคล้อง p=0.10 เดิม)
- รันบน random baseline → CI ต้องคร่อม 1.0 ชัดเจน (เป็น sanity check)

---

## เฟส 2 — เมตริกเสริมจาก EvalMetrics

ผลจริงบนการรันเดียวกัน: sortino 1.55 · calmar 2.28 (ปีละ +9.25% / DD 4.06%) · ulcer 1.18 (UPI 7.84) ·
equity R2 0.710 · ชนะติดกันสูงสุด 6 / แพ้ติดกันสูงสุด 7 · ถือเฉลี่ย 6.7 แท่ง (เพดาน 40)

### เพิ่ม 6 ตัวที่มีค่าจริงกับการตัดสินใจ

| เมตริก | ความหมาย | ทำไมต้องมี |
|---|---|---|
| `sortino` | Sharpe ที่ลงโทษเฉพาะขาลง | Sharpe ลงโทษกำไรพุ่งด้วย ซึ่งไม่ยุติธรรม |
| `calmar` | ผลตอบแทนต่อปี ÷ Max DD | ตัวเลขเดียวที่บอก "คุ้มเสี่ยงมั้ย" |
| `ulcer_index` | RMS ของ drawdown ทุกจุด | จับ "เจ็บนาน" ที่ Max DD จับไม่ได้ |
| `max_losses` | แพ้ติดกันสูงสุดกี่ไม้ | **จิตวิทยาคนเทรด** — รู้ว่าต้องทนอะไร |
| `equity_r2` | R² ของ equity curve | ความเรียบของกราฟทุน |
| `avg_trade_bars` | ถือเฉลี่ยกี่แท่ง | เทียบกับ max_hold ว่าชนเพดานบ่อยไหม |

### แก้ที่ไหน
- `backtest_live.py` ~L908-957 (บล็อกคำนวณ metrics) + ~L1217 (`meta["result"]`)
- ข้อมูลครบใน `trades` list + `eq` array อยู่แล้ว ไม่ต้องเก็บเพิ่ม
- `rl_app.py` หน้า Results — เพิ่มแถวในตารางสรุป

### ความเสี่ยง
ต่ำมาก — คำนวณล้วน ไม่กระทบ logic

---

## เฟส 3 — เร่งการเทรน (แทน Numba ที่ตัดทิ้ง)

### 3.1 ลด observation dimension (ทำก่อน — ฟรีและได้สองเด้ง)

ปัจจุบัน: window 30 × 122 features = 3,663 → layer แรกกิน 95% ของ params

**เด้งที่ 1 (ความเร็ว):** ลด obs ครึ่ง → เทรนเร็วขึ้นเกือบเท่าตัว
**เด้งที่ 2 (คุณภาพ):** 🩺 diagnosis ชี้แล้วว่า EV แค่ 0.30 เพราะ features ซ้ำซ้อน — การตัดฟีเจอร์ซ้ำช่วยทั้งคู่

ทางเลือก (ทดลองแบบ A/B ทีละตัว):
| ทางเลือก | obs dim | ผลคาดหวัง |
|---|---|---|
| baseline | 3,663 | 32 นาที/400k |
| window 30 → 15 | 1,833 | ~17 นาที — แต่สายตาสั้นลง |
| ตัด features ซ้ำ 122 → 60 | 1,803 | ~17 นาที + EV อาจดีขึ้น |
| ทั้งสองอย่าง | 903 | ~9 นาที |

**วิธีเลือกฟีเจอร์ที่จะตัด:** ใช้ correlation matrix — RSI/ATR/Stoch/CCI/WPR/ADX หลาย period ซ้อนกันเยอะ
(เช่น rsi_6/10/14/18/22/26/30/34/36 → เก็บ 3 ตัวพอ) ทำเป็น tool `tools/data/prune_features.py`

### 3.2 ทดสอบ GPU (ต้องวัดจริง ไม่ฟันธงล่วงหน้า)

- torch ปัจจุบันเป็น `+cpu` build → cuda_available = False ทั้งที่เครื่องมี RTX 4060
- ติดตั้ง `torch --index-url https://download.pytorch.org/whl/cu121` (~2.5GB)
- **ไม่รับประกันว่าเร็วขึ้น:** rollout เป็น forward batch=1 (44% ของเวลา) ซึ่ง GPU อาจช้ากว่า CPU เพราะ latency ต่อ call
  → ถ้าจะใช้ GPU ต้องคู่กับ **เพิ่ม n_envs เป็น 8-16** เพื่อ batch รวม rollout
- **แผนวัด:** รัน `bench_split.py` ซ้ำบน GPU build เทียบ 1/4/8/16 envs ก่อนตัดสินใจย้าย
- ⚠️ ติดตั้ง CUDA torch ทับ env เดิมมีความเสี่ยง — ทำใน venv แยกก่อน

### 3.3 สิ่งที่ตัดทิ้งแล้ว
~~Numba เร่ง env~~ — วัดแล้ว env = 0.4% ของเวลา เพดานการเร่ง 1.00x ไม่คุ้มแม้แต่น้อย

---

## เฟส 4 — Multi-symbol portfolio backtest

### ทำไม
PyBroker รันหลาย instrument พร้อมกันพร้อม ranking/position sizing ส่วนเราทดสอบทีละ symbol
**ประโยชน์ตรงกับปัญหาเรา:** significance ต้องการ ~225 เทรด (p<0.05) / ~450 เทรด (p<0.01)
ตอนนี้ UJ ให้ 138 เทรด — ถ้ารวม XAUUSD + คู่อื่นที่โมเดลเดียวกันเทรดได้ จะสะสมถึงเกณฑ์เร็วขึ้นมาก

ตอนนี้มี dataset พร้อมแล้ว: `usdjpy_h4_2025.csv`, `xauusd_h4_2025.csv` (+ M1 ทั้งคู่)

### ขอบเขต (เสนอให้เล็กไว้ก่อน)
**Phase 4a (เล็ก):** รัน backtest แยก symbol แล้ว **รวมสถิติ** (pooled significance + equity รวม)
- ไม่ต้องแตะ engine เลย — เขียน `tools/analysis/portfolio_report.py` อ่าน trades CSV หลายไฟล์มารวม
- ได้ pooled n_trades, PF รวม, correlation ระหว่าง equity curves

**Phase 4b (ใหญ่ — ค่อยตัดสินใจทีหลัง):** engine รันพร้อมกันจริง มี margin/position sizing ร่วม
- ต้องรื้อ `backtest_live.py` พอสมควร — ยังไม่แนะนำตอนนี้

### ข้อควรระวัง
โมเดลเทรนบน UJ ตัวเดียว — เอาไปรัน XAUUSD คือ **ทดสอบ generalization ข้ามตลาด** ซึ่งคนละเรื่องกับการสะสม sample
ถ้าจะ pool ต้องชัดว่ากำลังตอบคำถามไหน: "edge ทนข้ามตลาดมั้ย" หรือ "edge บน UJ จริงมั้ย"

---

## สิ่งที่ **ไม่** ขโมย (พิจารณาแล้วไม่คุ้ม)

| ฟีเจอร์ PyBroker | เหตุผลที่ไม่เอา |
|---|---|
| Numba JIT backtester | วัดแล้ว env ไม่ใช่คอขวด (0.4%) |
| Data source adapters (Alpaca/Yahoo) | เรามี MT5 collector ที่ parity กับ EA แล้ว — ของเขาไม่มี forex |
| Rule-based signal API | ขัดกับปรัชญา RL (เราให้โมเดลเรียน policy เอง ไม่ใส่กฎคน) |
| Parameter optimization (grid) | เสี่ยง overfit สูงกับ RL + รอบเทรนแพง — ใช้ A/B ทีละตัวแปรดีกว่า |

---

## ลำดับการลงมือ

1. **เฟส 1** (bootstrap CI) — เริ่มได้ทันที ไม่กระทบอะไร
2. **เฟส 2** (เมตริกเสริม) — ทำต่อเนื่องในไฟล์เดียวกัน commit รวมได้
3. **เฟส 3.1** (prune features) — ต้อง A/B เทรนเทียบ ใช้เวลาจริงหลายชั่วโมง
4. **เฟส 3.2** (GPU) — ทำเมื่อว่าง เสี่ยงกับ env เดิม
5. **เฟส 4a** (pooled report) — ทำหลังมีผล backtest หลาย symbol

---

## Sources
- https://www.pybroker.com/en/latest/ — feature list
- https://www.pybroker.com/en/latest/reference/pybroker.eval.html — EvalMetrics, BootstrapResult, DrawdownMetrics
- https://www.pybroker.com/en/latest/notebooks/3.%20Evaluating%20with%20Bootstrap%20Metrics.html — BCa method
- ผลวัดคอขวด: scratchpad `bench_bottleneck.py`, `bench_split.py` (2026-10-03)
