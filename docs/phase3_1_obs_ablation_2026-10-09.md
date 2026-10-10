# เฟส 3.1 — การทดลองลดขนาด observation (feature pruning × window)

> วันที่เตรียม: 2026-10-09
> สถานะ: รันรอบแรกแล้ว 2026-10-09 (tag `20261009_run1`) — ดู §7 ผลและข้อสรุป
> แผนแม่: [pybroker_steal_plan_2026-10-03.md](pybroker_steal_plan_2026-10-03.md) เฟส 3.1
> runner: `tools/analysis/obs_ablation.py`

---

## 1. คำถามที่ต้องการตอบ

การวัดคอขวด (3 ต.ค.) พบว่า env กินเวลาเทรนแค่ 0.4% ตัวการจริงคือขนาด observation:
window × features (3,663 ในโมเดลเดิม) ทำให้ layer แรกของ network ถือ ~95% ของพารามิเตอร์ทั้งหมด

คำถาม 2 ข้อ ตอบพร้อมกันในการทดลองเดียว:
1. **ความเร็ว** — ลด observation ลงครึ่งหนึ่งถึงหนึ่งในสี่ เทรนเร็วขึ้นกี่เท่าจริง
2. **คุณภาพ** — ผล OOS ของ best checkpoint ไม่แย่ลงใช่ไหม (วัดด้วยเกณฑ์ใหม่: PF พร้อม bootstrap CI)

สมมติฐานรอง: 🩺 diagnosis ของ `rl_uj_85` ให้ EV แค่ 0.30 และชี้ว่า features ซ้ำซ้อน ถ้าตัด feature ซ้ำแล้ว EV สูงขึ้น ถือเป็นโบนัส

ความเร็วที่วัดจริงล่วงหน้า (recipe เดียวกับที่จะใช้, 8,192 steps, CPU 14 threads):

| window | features | obs | steps/s | 200k steps |
|---:|---:|---:|---:|---:|
| 30 | 202 | 6,063 | 187 | 17.8 นาที |
| 30 | 100 | 3,003 | 323 | 10.3 นาที |
| 30 | 50 | 1,503 | 405 | 8.2 นาที |
| 15 | 202 | 3,033 | 237 | 14.1 นาที |
| 15 | 100 | 1,503 | 335 | 10.0 นาที |

อ่านได้ว่าการเร่งจากการลด obs มีเพดานราว 2.2x (ไม่ใช่เชิงเส้น เพราะ rollout batch=1 มี overhead คงที่)

---

## 2. ข้อค้นพบก่อนเริ่ม — dataset ชุดใหม่มีคอลัมน์เสีย 2 กลุ่ม

ตรวจ `usdjpy_h4_2025.csv` และ `xauusd_h4_2025.csv` (17,143 / 17,593 แถว, 202 features, 2014-12-31 → 2025-12-30 เก็บเมื่อ 2026-08-09 21:00) ก่อนใช้:

### 2.1 Divergence ทั้ง 10 คอลัมน์เป็นค่าคงที่ตลอดไฟล์

`rsi_14_div_*`, `macd_hist_div_*` = 0 ทุกแถว, `div_bull_age` / `div_bear_age` = 1.0 ทุกแถว (nunique = 1) ทั้งสอง symbol

**สาเหตุ (ยืนยันจากเวลาไฟล์):** dataset ถูกเก็บ 2026-08-09 21:00 แต่ `PriceDivergence.ex5` / `DataCollector_RL.ex5` ที่แก้บั๊กแล้ว compile 2026-08-10 17:18 → การเก็บใช้ build ก่อนแก้ ซึ่งมีบั๊กที่ Codex review พบ (สัญญาณบนแท่งที่ปิดแล้วถูกล้างตอน recalculate และ guard กันไม่ให้ตรวจซ้ำ) collector อ่าน shift=1 จึงเห็นศูนย์เสมอ

### 2.2 Nearness: คอลัมน์ 250 และ 1500 เหมือนกัน 100% และชื่อไม่ตรงค่า

เทียบกับ pandas `rolling(N).max()` จาก high ในไฟล์เดียวกัน:

| คอลัมน์ | ตรงกับหน้าต่างจริง |
|---|---|
| `near_high_100` | **250** แท่ง (98.7%) |
| `near_high_250` | **1500** แท่ง (98.3%) |
| `near_high_1500` | **1500** แท่ง (98.3%) |

เหมือน indicator ได้รับ horizon เป็น (250, 1500, 1500) แทน (100, 250, 1500) ทั้งที่ `params.json` บันทึก NEAR_N1..3 = 100/250/1500 และบนกราฟสด (screenshot AUDUSD H1) ค่าทั้งสามต่างกันถูกต้อง → เป็นพฤติกรรมเฉพาะตอนเก็บใน Strategy Tester ที่ยังหาสาเหตุไม่ได้จากการอ่านโค้ด (การส่งพารามิเตอร์ iCustom และการอ่าน buffer 0..8 ถูกต้อง)

### 2.3 ผลต่อการทดลองนี้

ตัด 19 คอลัมน์ที่เสียออก → ได้ `usdjpy_h4_2025_base.csv` **183 features ชุดเดียวกับ `uj_h4_extra_features.csv` ทุกตัว** (ตรวจแล้วไม่ต่างกัน) แต่ยาว 2 เท่า — การทดลองจึงตอบคำถาม obs reduction บนชุด feature ที่รู้จักดี ไม่ปนกับปัญหา dataset

### 2.4 งานที่ต้องทำก่อนใช้ feature ชุดใหม่ (นอกขอบเขตการทดลองนี้)

- [ ] หาสาเหตุ nearness ใน tester: ใส่ `Print` ใน `OnInit` ของ `PriceNearness.mq5` แสดง InpN1..3 และ `rates_total` รอบแรก แล้วรัน collector ช่วงสั้น
- [ ] compile `PriceDivergence` / `PriceNearness` / `DataCollector_RL` build ปัจจุบันใน terminal ที่ใช้เก็บ
- [ ] เก็บ `usdjpy` / `xauusd` ใหม่ แล้วรัน `tools/data/divergence_features.py --edge` เทียบว่า div จาก collector ตรงกับ Python twin
- [ ] ค่อยรัน arm ที่ชนะจากการทดลองนี้ซ้ำบน dataset ที่มี feature ครบ (= 3.1b)

---

## 3. การออกแบบ

### Dataset และการแบ่ง

`usdjpy_h4_2025_base.csv` 17,143 แถว, `--train_pct 0.70` → โค้ดเดือน ก.ย. แบ่ง holdout ครึ่งๆ อัตโนมัติ:

| ส่วน | แถว | ช่วงเวลา | ใช้ทำอะไร |
|---|---:|---|---|
| Train | 12,000 | 2014-12-31 → 2022-09-13 | fit prune/normalize + เรียน |
| Validation | 2,571 | 2022-09-13 → 2024-05-07 | EvalCallback เลือก best checkpoint |
| Test | 2,572 | 2024-05-07 → 2025-12-30 | **OOS backtest** (`--start 0.85`) ไม่ถูกใช้เลือกอะไร |

ใช้ 70/15/15 แทน 85/7.5/7.5 เพราะ Test 2,572 แท่ง (~20 เดือน) น่าจะให้ 250+ เทรด → CI แคบพอตัดสิน (138 เทรดของ `rl_uj_extra` ให้ CI กว้าง [0.82, 2.36])

smoke test ของ runner (12k steps, 2026-10-09): arm A ให้ 278 เทรดบน Test, arm D 79 เทรด — จำนวนเทรดต่างกันตามพฤติกรรมโมเดล ไม่ใช่ตามความยาวข้อมูล ดังนั้น CI ของ arm ที่เทรดน้อยจะกว้างกว่าโดยธรรมชาติ

### Arms

จำนวน feature หลังตัดวัดจริงจากช่วง Train (`prune_features` ตัวเดียวกับใน `rl_train.py`):

| arm | window | corr_threshold | features | obs | เวลาโดยประมาณ (200k) | บทบาท |
|---|---:|---:|---:|---:|---:|---|
| **A** | 30 | 1.00 | 181 | 5,433 | ~16 นาที | baseline (ตัดเฉพาะค่าคงที่: `candle_mathold`) |
| **B** | 30 | 0.95 | 85 | 2,553 | ~10 นาที | ตัด feature ซ้ำ |
| **C** | 15 | 1.00 | 181 | 2,718 | ~13 นาที | ลด window อย่างเดียว |
| **D** | 15 | 0.95 | 85 | 1,278 | ~10 นาที | ทั้งสองอย่าง |
| E | 30 | 0.90 | 65 | 1,953 | ~9 นาที | ตัดแรงขึ้น (ทำเมื่อ B ชนะ A) |

ที่ 0.95 ตระกูลที่โดนตัดคือ period ซ้อน: stoch 22, wpr 15, cci 15, adx 14, rsi 11, atr 10 — ตรงกับที่ diagnosis เคยชี้

### Recipe (เหมือนกันทุก arm)

```
lr 5e-5 · clip 0.1 · ent 0.02 · n_steps 4096 · n_epochs 15 · batch 128
gamma 0.99 · gae 0.97 · vf 0.7 · ep_len 2000 · max_hold 35
net_arch 256,128,64 (pin) · balanced · basic_4 · mc_eval 0
```

- lr 5e-5 + clip 0.1 คือคู่ที่พิสูจน์แล้วว่า KL อยู่ในโซนดี (0.027) ถ้าใช้ 1e-4 ทุก arm จะ noisy พร้อมกันและเทียบยาก
- **`net_arch` ต้อง pin** — ค่า `auto` จะเลือก [128,64] ให้ window < 20 ทำให้ arm C/D ต่างจาก A สองตัวแปร
- `mc_eval 0` ปิด MC ใน quick eval ของ rl_train เพื่อประหยัดเวลา (ทำ MC และ bootstrap ตอน OOS backtest แทน)

### ตัวแปรควบคุม

ทุกอย่างเท่ากันยกเว้น window / threshold: dataset, split, recipe, steps, การเลือก best, การ backtest
ข้อจำกัด: `rl_train.py` ไม่มี `--seed` → แต่ละ run สุ่มต่างกัน ผล 1 run ต่อ arm มี noise ในตัว (ดู §5)

### เมตริกที่เก็บต่อ arm

| กลุ่ม | เมตริก | ที่มา |
|---|---|---|
| ความเร็ว | เวลาเทรน (นาที), steps/s | จับเวลา subprocess |
| สุขภาพการเทรน | EV ตอนจบ, approx_kl, eval peak + step ที่พีค, จำนวน finding ระดับ bad | `train_diagnose.py` → `train.json` |
| คุณภาพ OOS (best checkpoint, Test เท่านั้น) | trades, PF, **PF 95% CI**, return, max DD, **DD 95% worst**, sortino | `backtest_live.py --source best --start 0.85 --bootstrap 1000` |

### กฎตัดสิน (non-inferiority เทียบ A)

ยอมรับ arm ที่เล็กกว่า เมื่อครบทุกข้อ:
1. PF ของ arm อยู่ใน PF 95% CI ของ A
2. ขอบล่าง PF 95% CI ไม่ต่ำกว่าของ A เกิน 0.10
3. max DD ไม่แย่กว่า A เกิน 2 จุดเปอร์เซ็นต์
4. เทรนเร็วกว่า A อย่างน้อย 1.5x

ถ้า arm ที่เล็กกว่า **ดีกว่า** A ชัด (ขอบล่าง CI สูงกว่า + EV สูงกว่า) = หลักฐานว่า feature ซ้ำซ้อนทำร้ายโมเดล ไม่ใช่แค่ทำให้ช้า

---

## 4. วิธีรัน

```powershell
# จาก project root
.\.venv\Scripts\python.exe tools/analysis/obs_ablation.py                 # A,B,C,D @ 200k (~1 ชม.)
.\.venv\Scripts\python.exe tools/analysis/obs_ablation.py --arms E        # เพิ่ม E ภายหลัง
.\.venv\Scripts\python.exe tools/analysis/obs_ablation.py --arms A,B --tag rep2   # รันซ้ำเพื่อดู variance
.\.venv\Scripts\python.exe tools/analysis/obs_ablation.py --dry_run       # ดูคำสั่งทั้งหมดก่อน
```

runner ทำต่อ arm: `rl_train.py` → `train_diagnose.py` → `backtest_live.py` (OOS) → เก็บผล
- ผลลัพธ์: `docs/ablation_<tag>.md` (ตาราง) + `.json` (ค่าดิบ) — เขียนใหม่หลังจบทุก arm จึงไม่เสียผลถ้าหยุดกลางคัน
- log เต็มของทุกขั้น: `artifacts/ablation/<tag>/<model>.{train,diag,backtest}.log`
- โมเดลชื่อ `abl_<arm>_w<window>_t<threshold>` เปิดดูในแอปได้ตามปกติ (ระบบ run-generation ของเดือน ก.ย.)

งบเวลา: A–D ≈ 50 นาทีเทรน + ~2 นาที/arm สำหรับ diagnosis + backtest ≈ **1 ชั่วโมง** · รันซ้ำ A กับ arm ที่ชนะอีกรอบ ≈ +30 นาที

---

## 5. ข้อควรระวังตอนอ่านผล

- **1 run ต่อ arm = ตัวอย่างเดียว** ความต่างของ PF ระหว่าง arm ที่น้อยกว่าความกว้างของ CI ตัดสินไม่ได้ ให้รันซ้ำ A และ arm ที่ชนะก่อนสรุป
- 200k steps บน Train 12,000 แถว ≈ 16.7 passes — ถ้า diagnosis บอก "train reward ยังไต่อยู่" ทุก arm อาจต้องเทียบที่ 400k อีกรอบ (เวลา ×2)
- Validation ถูกใช้เลือก best → ผลบน Validation มี selection bias ห้ามเอามาเทียบ ใช้ Test (`--start 0.85`) เท่านั้น
- window 15 บน H4 = มองย้อน 2.5 วัน vs 30 = 5 วัน ถ้า C/D แพ้ A ชัดแต่ B ชนะ แปลว่าปัญหาคือ feature ซ้ำ ไม่ใช่ความยาว window
- dataset ยาวถึงปลายปี 2025 → ไม่มีช่วง forward ว่างไว้ทดสอบหลังจากนี้ ถ้าจะ deploy arm ที่ชนะต้องเก็บข้อมูล 2026 แยกเป็น test ชุดที่สอง

---

## 6. หลังได้ผล

| ผลที่ออกมา | ทำต่อ |
|---|---|
| B หรือ D ผ่านกฎตัดสิน | ตั้ง corr_threshold 0.95 (และ window ถ้า D) เป็นค่าแนะนำใน Train page · ลอง E |
| เฉพาะ C ผ่าน | ปัญหาคือ window ยาวเกิน ลอง window 20 เพิ่ม |
| ไม่มี arm ไหนผ่าน | obs ไม่ใช่ตัวแปรที่ควรลด → เฟส 3.2 (GPU) เป็นทางเร่งที่เหลือ |
| ทุกกรณี | ทำ §2.4 แล้วรัน arm ที่ชนะบน dataset ที่มี divergence/nearness จริง (3.1b) |

---

## ไฟล์ที่เกี่ยวข้อง

- `tools/analysis/obs_ablation.py` — runner
- `usdjpy_h4_2025_base.csv` + `.params.json` — dataset ฐาน 183 features (ไม่ได้อยู่ใน git เหมือน CSV อื่น)
- `docs/pybroker_steal_plan_2026-10-03.md` — ผลวัดคอขวดและที่มาของเฟสนี้

---

## 7. ผลรอบแรก (tag `20261009_run1`, 200k steps/arm, 1 run ต่อ arm)

ตารางเต็ม: `docs/ablation_20261009_run1.md` · log: `artifacts/ablation/20261009_run1/`

| arm | obs | เวลาเทรน | เร็วกว่า A | eval peak @ step | EV | KL | OOS trades | PF | PF 95% CI | return | max DD |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| A w30 full | 5,433 | 22.5 นาที | 1.0x | 6.03 @ 90k (overfit หลังนั้น) | −0.06 | 0.04 | 213 | 0.85 | [0.55, 1.33] | −7.5% | −16.0% |
| B w30 t0.95 | 2,553 | 12.5 นาที | **1.8x** | 9.11 @ 200k (ยังขึ้นอยู่) | −0.22 | 0.03 | 196 | 0.74 | [0.48, 1.09] | −18.0% | −25.8% |
| C w15 full | 2,718 | 13.5 นาที | 1.7x | 4.33 @ 80k | 0.09 | 0.03 | 81 | 0.63 | [0.35, 1.20] | −17.8% | −22.2% |
| D w15 t0.95 | 1,278 | 11.5 นาที | **2.0x** | 0.25 @ 140k | 0.15 | 0.02 | 149 | 1.00 | [0.62, 1.62]* | −0.8% | −10.8% |

\* block-5 bootstrap ของ D ให้ขอบล่าง 0.52 (เทรดมาเป็นสตรีค) ใช้ค่านี้แทน 0.62

### คำถามข้อ 1 (ความเร็ว) — ตอบได้ชัด

ตัด feature ซ้ำที่ 0.95 (181 → 85) เทรนเร็วขึ้น **1.8x** ลด window ครึ่งหนึ่งได้ 1.7x ทั้งสองอย่างได้ 2.0x
ตรงกับที่วัดไว้ล่วงหน้า และไม่มี arm ไหนให้หลักฐานว่าการลดทำให้แย่ลง (CI ทับซ้อนกันหมด)

### คำถามข้อ 2 (คุณภาพ) — ตัดสินไม่ได้ เพราะ baseline เองก็ไม่มี edge

กฎ non-inferiority ใช้ไม่ได้: **ไม่มี arm ไหนมี edge บน Test** — PF ทุก arm ≤ 1.0, CI ทุกตัวคร่อม 1.0, return ติดลบ 3 ใน 4
สัญญาณที่หนักกว่าคือ **EV ≈ 0 ทุก arm (−0.22 ถึง 0.15)** — Critic ทำนายผลตอบแทนไม่ได้เลย
ไม่ว่าจะ obs ใหญ่หรือเล็ก → ข้อจำกัดอยู่ที่ "signal ใน features + reward" ไม่ใช่ขนาด observation

เทียบกับ `rl_uj_extra` (เทรนก่อนงาน ก.ย.): EV 0.60, OOS PF 1.42 — ต่างกันมาก สาเหตุที่เป็นไปได้ (ยังแยกไม่ออก):
1. **ช่วงข้อมูล**: เทรน 2015→2022 แล้วทดสอบ 2024-05→2025-12 ห่างกันมาก ตรงกับข้อสังเกต regime locality ก่อนหน้า (`rl_uj_extra` เทรน 2021→2025 ใกล้ช่วงทดสอบกว่า)
2. **reward ที่แก้เดือน ก.ย.**: รอบยาวของงาน ก.ย. เอง (`codex_training_check_20260905`) ก็ได้ PF 0.98 / 0.43 บน Test — ตั้งแต่แก้ reward ยังไม่มีการเทรนรอบไหนได้ edge บน Test เลย
3. **ผลเก่าอาจมี selection bias**: `rl_uj_extra` เลือก best บน validation แล้ว backtest ช่วงเดียวกัน ส่วนรอบนี้ Test แยกจาก Validation จริง จึงอาจเป็นภาพที่ "ซื่อสัตย์กว่า" ของ edge ที่มีจริง

### ข้อสังเกตย่อย
- B มี validation peak สูงสุด (9.11) แต่ OOS แย่สุด → validation กับ test ไม่ไปด้วยกัน = regime เปลี่ยนระหว่าง 2022-24 กับ 2024-25 หรือ noise
- B peak ที่ 200k (step สุดท้าย) = ยังไม่ overfit อาจได้ประโยชน์จาก steps เพิ่ม
- D: EV สูงสุด, KL ต่ำสุด, DD ดีสุด — ทิศทางสอดคล้องกับ "feature น้อย = noise น้อย" แต่ 1 run ยืนยันอะไรไม่ได้
- A เทรนช้ากว่า benchmark (148 vs 187 steps/s) เพราะรวม load/eval/quick-eval overhead

### สิ่งที่ตัดสินใจได้จากรอบนี้
- **ใช้ `--corr_threshold 0.95` เป็นค่าเริ่มต้นได้** เพื่อความเร็ว (ได้ 1.8x ฟรี ไม่มีหลักฐานว่าเสียหาย)
- **หยุดทดลองเรื่อง obs ต่อ** จนกว่าจะมี baseline ที่มี edge ให้เทียบ

### ขั้นถัดไปที่เสนอ (เรียงตามที่จะให้ข้อมูลมากสุด)
1. **Replication**: เทรน config ของ `rl_uj_extra` (uj_h4_extra_features.csv, window 20, recipe เดิม) ด้วยโค้ดปัจจุบัน แล้ว backtest `--start 0.85` — ถ้าได้ ~1.4 กลับมา = ปัญหาคือช่วงข้อมูล/regime ถ้าไม่ได้ = การแก้ reward เดือน ก.ย. เปลี่ยนผลลัพธ์ ต้องตรวจ
2. **ช่วงเทรนสั้นลง**: ตัด CSV ให้เริ่ม 2019 แล้วรัน arm D ซ้ำ ทดสอบสมมติฐาน regime locality โดยตรง
3. **B ที่ 400k steps**: เพราะยังไม่ถึงพีค
4. §2.4 — เก็บ dataset ใหม่ให้ divergence/nearness ใช้ได้จริง ก่อนรอบ 3.1b

---

## 8. Replication ของ `rl_uj_extra` บนโค้ดปัจจุบัน (2026-10-09, `rep_uj_extra_20261009`)

config เดิมทุกค่า (uj_h4_extra_features, window 20, lr 1e-4, clip 0.15, ent 0.015, 4096/20/128, gae 0.95, vf 0.7,
max_hold 40, 250k steps, train 85%) · log: `artifacts/ablation/replication/`

| | `rl_uj_extra` (ก.ค. โค้ดเก่า) | replication (โค้ดปัจจุบัน) |
|---|---|---|
| EV ตอนจบ | 0.60 | **0.74** |
| approx_kl | 0.192 | 0.189 (เท่ากัน) |
| eval peak | 3.03 @ 130k แล้ว overfit | 2.52 @ 250k ยังไม่ overfit |
| best เลือกจาก | rows 85–100% (ช่วงเดียวกับ backtest) | rows 85–92.5% (validation ของ split ใหม่) |
| OOS `--start 0.85` | **PF 1.42**, 138 เทรด, +7.4%, DD −4.1%, CI [0.82, 2.36] | **PF 0.91**, 125 เทรด, −2.3%, DD −8.4%, CI [0.46†, 1.50] |
| OOS `--start 0.925` (Test ล้วน) | — | PF 0.47, 68 เทรด, −7.4% |

† block bootstrap (เทรดเป็นสตรีค)

### อ่านผล
1. **โค้ดเดือน ก.ย. ไม่ได้ทำให้ "เรียนไม่ได้"** — EV 0.74 สูงกว่าเดิม, KL เท่าเดิม, training dynamics เหมือนเดิม
   → EV ≈ 0 ในการทดลอง 3.1 เป็นเรื่องของ **ช่วงข้อมูล 2015-2022 → 2024-25** ไม่ใช่โค้ด (ข้อ 1 ในสาเหตุที่สงสัยของ §7 ยืนยัน)
2. **ตัวเลข 1.42 ไม่กลับมา** (0.91) แต่ CI ของสองรอบ [0.82, 2.36] กับ [0.46, 1.50] **ทับซ้อนกันมาก** —
   ด้วย ~130 เทรดต่อรอบ สถิติแยก 1.42 จาก 0.91 ไม่ได้ ทั้งสองค่าเข้ากันได้กับ edge จริงที่ ~1.0–1.1
3. ยังแยกไม่ออกระหว่าง (ก) run-to-run variance (ไม่มี seed control) (ข) การแก้ reward ก.ย. เปลี่ยน policy
   (ค) 1.42 เดิมมี selection bias เพราะเลือก best บน rows เดียวกับที่ backtest
4. backtest engine ไม่ใช่ตัวแปร: โมเดลเก่าตัวเดิม backtest ด้วย engine ปัจจุบันได้ 1.42 เท่าเดิมเป๊ะ (รอบ bootstrap 2026-10-03)

### ภาพรวมหลักฐานทั้งหมดหลังงาน ก.ย.
| รอบ | OOS PF | ช่วง test |
|---|---:|---|
| codex_training_check_20260905 (final / best) | 0.98 / 0.43 | 2026-01 → 06 |
| 3.1 arms A/B/C/D | 0.85 / 0.74 / 0.63 / 1.00 | 2024-05 → 2025-12 |
| replication rl_uj_extra | 0.91 (0.47 บน Test ล้วน) | 2025-08 → 2026-06 |

ทุกรอบ CI คร่อม 1.0 — **ยังไม่มี configuration ไหนแสดง edge ที่พิสูจน์ได้** ค่า 1.42 เดิมเป็นค่าเดียวที่โดดออกมา
และเป็นค่าที่ได้จาก protocol ที่มองโลกในแง่ดีที่สุด (เลือก best บน rows ที่ใช้วัด)

### ขั้นถัดไปที่เสนอ
1. **เพิ่ม `--seed` ใน rl_train.py** (PPO seed + env seed) เพื่อให้การเทียบรอบต่อไปควบคุมได้
2. **variance check**: รัน replication ซ้ำ 2 รอบ (seed ต่างกัน) → ถ้า PF กระจาย 0.5–1.4 แปลว่า 1 run ตัดสินอะไรไม่ได้
   ต้องเปลี่ยนวิธีประเมินเป็นหลาย seed · ถ้าเกาะกลุ่ม ≤ 1.0 ค่อยทดสอบโค้ดเก่าใน worktree (ข้อ ข)
3. ถ้าหลาย seed ก็ไม่ได้ edge → หยุดจูน training ไปทำฝั่งข้อมูล (§2.4 + regime) ก่อน

---

## 9. เพิ่ม `--seed` ใน rl_train.py (2026-10-10)

- `--seed N` ส่งให้ PPO/DQN/A2C (`seed=`) → SB3 ตั้ง torch/numpy/random, action space และ train env
  (→ `TradingEnv.reset(seed)` → จุดเริ่ม episode สุ่มผ่าน `np_random`) · eval env ใช้ `seed+1`
- บันทึกใน `train.json` → `hyperparameters.seed` · ไม่ส่ง = None = สุ่มเหมือนเดิม (GUI ยังไม่มีช่องนี้)
- ตรวจแล้ว: seed 7 สองรอบได้ policy weights เหมือนกันทุกไบต์ (quick eval 62 เทรด −2.35% ทั้งคู่), seed 8 ต่าง
- regression: test_training_improvements 22, test_batch1_bugfixes 18, test_ui_training_contract 14 ผ่าน

## 10. Variance check — ผล (2026-10-10)

config เดียวกับ §8 ทุกค่า ต่างกันแค่ seed · log: `artifacts/ablation/replication/`

| run | EV | eval peak @ | `--start 0.85` (val+test, เลือก best บน val) | `--start 0.925` (Test ล้วน) |
|---|---:|---|---|---|
| ไม่มี seed (§8) | 0.74 | 2.52 @ 250k | PF **0.91** · −2.3% · DD −8.4% · 125 เทรด | PF **0.47** · −7.4% · 68 เทรด |
| seed 1 | 0.50 | 3.83 @ 170k | PF **1.35** · +7.1% · DD −5.1% · 149 เทรด | PF **0.89** · −1.3% · 88 เทรด |
| seed 2 | 0.69 | 4.25 @ 120k | PF **1.25** · +5.5% · DD −5.5% · 130 เทรด | PF **0.95** · −0.9% · 68 เทรด |
| `rl_uj_extra` เดิม (ก.ค.) | 0.60 | 3.03 @ 130k | PF **1.42** · +7.4% · DD −4.1% · 138 เทรด (best เลือกบน rows เดียวกัน) | — |

### ข้อสรุป
1. **run-to-run variance ใหญ่มาก**: config เดียวกันให้ PF 0.91 / 1.35 / 1.25 บนช่วงเดียวกัน → 1 run ตัดสินอะไรไม่ได้
   ทุก A/B ต่อจากนี้ต้องใช้ ≥ 3 seed (มี `--seed` แล้ว)
2. **โค้ดเดือน ก.ย. ไม่ใช่สาเหตุ**: 2 ใน 3 seed กลับไปใกล้ 1.42 บนช่วงเดิม → ตัดสมมติฐาน "reward fix ทำพัง" ทิ้งได้
3. **1.42 เป็นค่าที่ selection-inflated**: บน Test ที่ไม่ถูกใช้เลือก best ทั้ง 3 seed ได้ PF < 1 (0.47 / 0.89 / 0.95, return ติดลบหมด)
   ส่วนที่ทำให้ช่วง 0.85 ดูดีมาจากครึ่ง validation ที่ใช้เลือก checkpoint — protocol เดิมของ `rl_uj_extra` (เลือกและวัดบน rows เดียวกัน) จึงมองโลกในแง่ดีโดยโครงสร้าง
4. ข้อจำกัด: Test ล้วนมีแค่ 68–88 เทรด (≈ 3.5 เดือน H4) CI กว้าง ([0.42, 1.67] ฯลฯ) → ยืนยัน "ไม่มี edge" ได้ไม่เด็ดขาด
   แต่ 3 seed ไปทางเดียวกันหมด และไม่มีอะไรบอกว่ามี edge
   รวมเทรด Test ล้วนทั้ง 3 seed: 224 เทรด, pooled PF **0.76**, ผลรวม −9.6% (ไม่ใช่ตัวอย่างอิสระ แต่ไปทางเดียวกัน)

**ตามข้อ 3 ที่ตกลง: หยุดจูน training แล้วไปทำฝั่งข้อมูล** (§2.4 + regime) — สิ่งที่ได้ติดมือจากเฟสนี้:
`--seed`, runner, bootstrap CI เป็นเกณฑ์, `--corr_threshold 0.95` เร็วขึ้น 1.8x โดยไม่มีหลักฐานว่าเสียหาย

