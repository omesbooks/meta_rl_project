# ผลการแก้ Training และ Dashboard

วันที่: 2026-09-05

แก้ชุดหลักแล้วตาม [checklist](training_improvement_checklist_2026-09-05.md): ปิดได้ 62/64 รายการ รวม UI-02/UI-03/UI-T01 ที่แก้เพิ่ม อีก 2 รายการแยกไว้ชัดเจน ไม่ถือว่าผ่าน MT5 หรือพร้อม deploy

## ผู้ใช้ต้องรู้อะไรบ้าง

1. ปิดแล้วเปิด Dashboard ใหม่เพื่อโหลดโค้ดที่แก้
2. ใช้ CSV ต้นฉบับ หรือกด Clean ใหม่ให้ได้ `.features.json` คู่กัน ไฟล์ `_clean.csv` รุ่นเก่าที่ไม่มีหลักฐานช่วงเลือกฟีเจอร์จะถูกปฏิเสธ
3. ใน Data Tools ตั้ง `Fit features on first Train %` ให้ตรงกับ Train หรือใช้ช่วงที่แคบกว่า ห้ามใช้อนาคตเลือกฟีเจอร์แล้วนำไปทดสอบว่าเป็นข้อมูลใหม่
4. Train 85% แบ่งเป็น Train 85%, Validation ประมาณ 7.5%, Test ประมาณ 7.5% ตามลำดับเวลา ชุด Validation ใช้เลือก best checkpoint; Test ไม่ถูกใช้โดย EvalCallback
5. Train 100% ไม่มี unseen Test ผลท้ายการฝึกเป็น in-sample diagnostic หากต้องการประเมินใหม่ต้องมีข้อมูลช่วงหลังแยกต่างหาก
6. รอบ Train/Fine-tune ใหม่อยู่ใน `artifacts/models/<name>/runs/<run_id>/` โดย `current.json` ในโฟลเดอร์ชื่อโมเดลชี้ชุดที่ใช้อยู่ ให้เลือกโมเดลผ่าน Dashboard ไม่ต้องย้ายไฟล์เอง
7. โมเดลเดิมยังเปิดได้ แต่ต้อง Train ใหม่เพื่อเรียนรู้ reward ที่แก้แล้ว การอัปเดต source ไม่ได้เปลี่ยนน้ำหนักโมเดลเดิม

## สิ่งที่แก้

| ส่วน | พฤติกรรมใหม่ |
|---|---|
| บันทึกโมเดล | แต่ละ run แยกโฟลเดอร์ ตรวจ zip/norm/metadata/params แล้วสลับ `current.json` ครั้งเดียว การฝึกล้มเหลวหรือยกเลิกไม่แทนชุดเดิม |
| ตรวจ artifact | ตรวจลำดับฟีเจอร์, observation/action dimensions, โหลด zip กลับ, บันทึก hashes และสถานะ run |
| การอ่าน artifact | Backtest/Analyze/Fine-tune/Export ยึด generation เดียวกัน ไม่หยิบ sidecar หรือรายงานเก่าข้ามรุ่น; Export best ไม่ fallback ไป best เก่า |
| Reward | Close, Max Hold, managed stop และจบ episode ใช้การปิดร่วมกัน นำ closing PnL เข้าคิด reward ก่อนคืน step และไม่ปิดซ้ำ |
| Equity / idle | equity curve จบตรงกับบัญชี; Close ขณะว่างไม่หลบ idle penalty; episode ที่ liquidate เป็น terminal ไม่ bootstrap ต่อ |
| เตรียมข้อมูล | ตรวจ timestamp/ซ้ำ/OHLC/NaN/Inf/ฟีเจอร์ขาดร่วมกัน; คัดค่าคงที่และ correlation จาก Train เท่านั้น |
| Backtest / Analyze | อ่านรายชื่อและลำดับฟีเจอร์จาก norm ของโมเดล ไม่ใช้ลำดับคอลัมน์ CSV; ยอมให้ CSV มีฟีเจอร์เกิน แต่ฟีเจอร์จำเป็นขาดไม่ได้ |
| Walk-Forward | ใช้ CSV ต้นฉบับก่อนเลือกฟีเจอร์ แล้ว fit selection/normalization ใหม่ทุก window; บันทึก `<name>_preprocessing.json` |
| Fine-tune | ส่ง reward overrides/formula และ action config, ตั้ง LR พร้อม schedule, ใช้ช่วง old/new ต่อเนื่องแยก episode, ไม่เขียน logs ทับ base |
| Dashboard | แสดงเฉพาะ PPO, เพิ่มตัวตรวจ PPO ร่วมกัน, แสดง Train/Validation/Test และเพิ่มสรุปใน log |
| ยืนยันก่อนเริ่ม | Train/Pipeline แสดงค่าที่ resolve แล้วและจำนวนฟีเจอร์หลังคัดเลือกก่อนเริ่ม process; ตรวจ fingerprint ซ้ำก่อนฝึกและบันทึก confirmed recipe ใน metadata |
| ตรวจค่าหน้าแอป | ปฏิเสธช่องว่าง, NaN/Inf, percentage เกินขอบเขต และจำนวนเต็มที่มีทศนิยม; ไม่ clamp reward/action เงียบ ๆ; Backtest Window ใช้ `auto` อย่างชัดเจน |
| Copy ไป Walk-Forward | เก็บค่าตัวเลขตามที่กรอก พร้อม custom action JSON; ไม่มี fallback เมื่อช่อง Train ว่าง และล้าง JSON เดิมเมื่อเปลี่ยน action profile เอง |

Fine-tune บันทึกจำนวนแถวแต่ละ segment และจำนวน training steps ที่เกิดขึ้นจริง สัดส่วนที่ใช้ฝึกอาจไม่เท่ากับ ratio เป๊ะในรอบสั้น ผล quick eval ของ Fine-tune ใช้ข้อมูลที่เพิ่งฝึก จึงระบุว่า in-sample

## หลักฐานทดสอบ

รอบ UI เพิ่มเติม: tests ใหม่ **14/14** และ regression **40/40** ผ่าน รวม **54 tests**; integration รันซ้ำผ่าน **11/11** และ ONNX parity 50 observations มี maximum absolute error `2.9802322387695312e-08` ดู [รายงานรอบ UI](ui_training_contract_validation_2026-09-05.md)

- Regression + tests ใหม่ผ่าน **40/40**: `test_batch1_bugfixes.py` และ `test_training_improvements.py`
- Integration ผ่าน **11 checks**: Train basic_4/manage_6, failed retrain, Train 100%, Backtest, chart, Analyze, Fine-tune, Walk-Forward, Export และ ONNX parity
- Integration ใช้ CSV ที่มีคอลัมน์ค่าคงที่ และสลับลำดับคอลัมน์สำหรับ Backtest/Analyze เพื่อทดสอบว่าใช้ feature contract จากโมเดลจริง
- Fine-tune ตรวจ reward override `trade_penalty=0.012`, LR schedule/optimizer `0.0001` และ hashes ของทุกไฟล์ base model ต้องไม่เปลี่ยน
- ONNX เทียบกับ PPO บน observation เดียวกัน 50 ตัวอย่าง ค่าคลาดเคลื่อนสูงสุดบันทึกใน [รายงาน integration](rl_training_validation_2026-09-05.json)
- โหลดโมเดลเดิม `rl_uj_extra` ได้: 183 features, observation 3663, 4 actions โดยไม่แก้ไฟล์ของผู้ใช้
- ตรวจ Walk-Forward เพิ่มบนข้อมูล 600 แถว: window แรก fit แถว 0:300 และ window สอง fit แถว 150:450; วันที่สิ้นสุด fit ต้องก่อน Test และค่า mean ใน metadata ตรงกับช่วง Train ของแต่ละรอบ
- ตรวจ syntax 13 ไฟล์ และ IDs/internal anchors ของคู่มือ HTML ที่แก้ 4 ไฟล์ ไม่มี visual screenshot QA รอบนี้

คำสั่งสำหรับรันทดสอบซ้ำจาก project root:

```powershell
.\.venv\Scripts\python.exe -m unittest test_training_improvements test_batch1_bugfixes -v
.\.venv\Scripts\python.exe tools/analysis/training_smoke_check.py --report "$env:TEMP\rl_training_validation.json"
```

Integration สร้างข้อมูล/โมเดลในโฟลเดอร์ชั่วคราวและล้างเมื่อจบ รายงาน JSON เก็บผลไว้ แต่ path ชั่วคราวในรายงานไม่ได้เป็นไฟล์ถาวร

## Train รอบยาว

ชื่อโมเดลทดสอบ: `codex_training_check_20260905`

- CSV: `uj_h4_extra_features.csv`, 8,493 แถว
- Train 7,219 / Validation 637 / Test 637 แถว; 183 ฟีเจอร์ก่อนคัด เหลือ 182 หลังตัด `candle_mathold` ที่คงที่ใน Train
- PPO ขอ 200,000 steps ใช้จริง **200,704** ตามขนาด rollout; window 10, basic_4, balanced, max hold 30
- Run ID: `b2ddbd224c784daa982b34d1b19cf8c7`; ตรวจ `current.json` และ metadata เป็น completed แล้ว
- Test: 2026-01-15 16:00 ถึง 2026-06-12 16:00 ทั้ง final และ best ใช้ข้อมูล 637 แถวชุดเดียวกัน

| Checkpoint | Trades | Return | PF | Max DD |
|---|---:|---:|---:|---:|
| Final | 61 | -0.27% | 0.98 | -6.34% |
| Best จาก Validation | 39 | -6.49% | 0.43 | -8.04% |

เทียบด้วย `backtest_live.py`, `pure_agent`, confidence 0, max hold จาก metadata และ MC 100 รอบ เก็บผลของแต่ละ checkpoint แยกในพื้นที่ชั่วคราวเพื่อไม่ทับรายงานกัน ดูค่าตั้งทั้งหมดใน [รายงานรอบยาว](rl_full_training_validation_2026-09-05.json)

ผลนี้ยืนยันการทำงานครบวงจร ไม่ได้ยืนยันว่าการแก้ทำให้กำไรเพิ่มขึ้น และไม่ได้เป็นการเปรียบเทียบก่อน/หลังแก้ reward ด้วยหลาย seeds โมเดลทั้งสองยังไม่ผ่านข้อสรุปว่ามี edge; ยังไม่ได้ใช้ M1/ticks หรือ broker-specific swap ในการเทียบนี้

## Divergence ที่เป็นศูนย์ตลอด

สองไฟล์ `usdjpy_h4_2025.csv` และ `xauusd_h4_2025.csv` มี divergence flags 8 คอลัมน์เป็น 0 และ age 2 คอลัมน์เป็น 1 ตลอด

พบ source `PriceDivergence.mq5` ที่ติดตั้งใน terminal `5FFA568149E88FCD5B44D926DCFEAA79` ต่างจาก source ใน repo:

- Installed SHA256: `B7FCEF87EA69331F194D225DFBF40DE87874393EBC6B916B9D00FADA4CEC36BC`
- Repo SHA256: `6ABDAADBF1F5F6EED3D77275B1F6ACBCB91B8498526EA08069382D30B009CEF5`
- Source ที่ติดตั้งยังวนถึง `i < rates_total` รวมแท่งกำลังก่อตัว และมี guard ไม่บันทึก pivot เดิมซ้ำ ขณะที่ source ใน repo แก้เป็น closed bars (`i < rates_total - 1`) แล้ว
- คำนวณราคาเดิมด้วย Python divergence ปัจจุบัน: USDJPY พบ any-bull 251 / any-bear 311; XAUUSD พบ 291 / 357 จึงไม่ควรสรุปว่าข้อมูลช่วงนี้ไม่มี divergence จริง

นี่เป็นหลักฐานชี้ไปที่ indicator รุ่นเก่า แต่ยังไม่ยืนยันว่า `.ex5` ที่สร้าง CSV มาจาก source ใดแน่นอน ต้อง compile/install รุ่นปัจจุบันและเก็บ CSV ใหม่จาก MT5 แล้วตรวจ flags/age/sidecar อีกครั้ง รอบนี้ไม่เขียนทับ indicator ใน terminal และไม่แก้ CSV เดิม

## งานที่ยังเหลือ

อีก 2 รายการคือ `DATA-T05` และ `FLOW-07`: ยืนยัน Divergence ผ่าน MT5 จริง และทดสอบ lifecycle ผ่าน Dashboard กับ subprocess จริงทุกหน้า ส่วน UI-02/UI-03/UI-T01 ปิดแล้วตามหลักฐานรอบ UI

ดูรายละเอียดและเกณฑ์ปิดงานใน [checklist](training_improvement_checklist_2026-09-05.md) ยังไม่มี MT5 compile/test/live proof และไม่ได้ commit หรือ push การแก้รอบนี้
