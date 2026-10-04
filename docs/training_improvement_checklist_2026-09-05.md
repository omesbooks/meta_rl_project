# Checklist: ปรับปรุงกระบวนการ Train และ Dashboard

วันที่จัดทำ: 2026-09-05

สถานะ: แก้และตรวจผ่าน 62/64 รายการในรอบ 2026-09-05 รวม UI-02, UI-03 และ UI-T01; อีก 2 รายการยังไม่ปิด ดูรายละเอียดด้านล่าง

เอกสารนี้แปลงผลตรวจแอปและไฟล์ Train เป็นรายการงาน เรียงตามลำดับที่ควรทำ:

**การบันทึกไฟล์ -> Reward -> ข้อมูลและการประเมิน -> Fine-tune และหน้าแอป**

## วิธีใช้ Checklist

- ติ๊กแต่ละรายการเมื่อทำเสร็จและตรวจผลแล้ว พร้อมบันทึกหลักฐานในท้ายเอกสาร
- การแก้โค้ดเสร็จยังไม่ถือว่าจบชุดงาน ต้องผ่านเกณฑ์ทดสอบของชุดนั้นด้วย
- ใช้ข้อมูลจำลองและชื่อโมเดลสำหรับทดสอบแยกจากโมเดลที่ผู้ใช้มีอยู่
- ใช้ `.venv/Scripts/python.exe` และรันจาก project root
- แยกผลการตรวจโค้ด, unit test, Train สั้น, Backtest, Export และ MT5 ออกจากกัน ไม่ใช้ผลระดับหนึ่งยืนยันแทนอีกระดับ

ผลตรวจตั้งต้น: regression test เดิมผ่าน 18/18 และตรวจ syntax ไฟล์หลักผ่าน 11 ไฟล์ แต่ยังไม่ครอบคลุมปัญหาใหม่ใน checklist นี้

## ชุดที่ 1: ป้องกันโมเดลกับไฟล์ประกอบไม่ตรงกัน

ความสำคัญ: ทำก่อน เพราะการเทรนทับชื่อเดิมแล้วล้มเหลวอาจเปลี่ยน normalization/metadata ของโมเดลเดิม

ไฟล์หลัก: [rl_train.py](../rl_train.py), [artifact_paths.py](../artifact_paths.py), [rl_app.py](../rl_app.py)

### งานที่ต้องแก้

- [x] SAVE-01 ตรวจข้อมูลและค่าตั้งต้นก่อนเขียนทับ artifact ใด ๆ รวมถึงเงื่อนไขของ PPO เช่น batch size และ rollout size
- [x] SAVE-02 ให้แต่ละรอบ Train มี run ID และโฟลเดอร์ staging ของตัวเอง โดยโมเดลที่ใช้งานอยู่ยังไม่เปลี่ยน
- [x] SAVE-03 บันทึก `.zip`, `_norm.csv`, `.train.json` และ `.params.json` เมื่อมี sidecar ให้เป็นชุดของ run เดียวกัน; ถ้าไม่มี sidecar ต้องระบุว่าขาด ไม่ดึงของรอบเก่ามาปะปน
- [x] SAVE-04 ตรวจ feature count/order, observation dimension, action profile และตำแหน่ง artifact ก่อนเผยแพร่ชุดโมเดลใหม่
- [x] SAVE-05 ออกแบบการสลับชุดโมเดลที่กู้คืนได้เมื่อถูกขัดจังหวะบน Windows โดยการอ่านโมเดลต้องไม่เห็นชุดที่สลับไฟล์ไปเพียงบางส่วน
- [x] SAVE-06 บันทึกสถานะ `running / completed / failed / cancelled` ของรอบ Train แยกจาก metadata ของโมเดลเดิมที่สำเร็จแล้ว
- [x] SAVE-07 ให้ปุ่ม Stop และการปิด Dashboard บันทึกสถานะยกเลิก; กรณี process ถูก kill ให้ตรวจพบ staging ที่ค้างเมื่อเปิดครั้งถัดไป
- [x] SAVE-08 ให้ best model, normalization และรายงาน Backtest ผูกกับรุ่นโมเดลที่ถูกต้อง ไม่แสดงรายงานรุ่นเก่าเป็นผลของรุ่นใหม่

### เกณฑ์ทดสอบก่อนปิดชุดงาน

- [x] SAVE-T01 จำลองค่าผิด, การเทรนล้มเหลว, Stop และการถูกขัดจังหวะระหว่างบันทึก แล้วตรวจว่าโมเดลเดิมและไฟล์ประกอบยังไม่เปลี่ยน
- [x] SAVE-T02 เทรนสำเร็จแล้วโหลดโมเดลพร้อม normalization/metadata ของ run เดียวกันได้
- [x] SAVE-T03 ทดสอบเทรนทับชื่อเดิมโดย CSV ใหม่ไม่มี sidecar แล้วไม่หลงเหลือ params เก่ามาประกอบโมเดลใหม่
- [x] SAVE-T04 ตรวจว่าโมเดลรูปแบบเก่าที่ยังรองรับอยู่สามารถค้นหาและโหลดได้ตามเดิม

## ชุดที่ 2: แก้ Reward และการปิดสถานะ

ความสำคัญ: แก้ก่อน Train รอบจริง เพราะกระทบสิ่งที่ agent เรียนรู้

ไฟล์หลัก: [trading_env.py](../trading_env.py), [reward_profiles.py](../reward_profiles.py), [reward_formula.py](../reward_formula.py)

### งานที่ต้องแก้

- [x] REWARD-01 รวมการปิดด้วย Close, Max Hold, managed stop และจบ episode ให้ใช้กระบวนการปิดสถานะร่วมกัน พร้อมระบุเหตุผลการปิด
- [x] REWARD-02 นำกำไรขาดทุนจากการปิดท้าย episode เข้าคำนวณ reward ก่อนคืนผล step โดยไม่คิดซ้ำ
- [x] REWARD-03 อัปเดต equity curve, trade list และสถานะ position หลังปิดให้ตรงกัน
- [x] REWARD-04 แก้ idle penalty ให้พิจารณาสถานะและผลของ action เพื่อไม่ให้ Close ตอนว่างหลบ penalty ได้ และกำหนดพฤติกรรมของ invalid action ให้ชัดเจน
- [x] REWARD-05 กำหนดความหมายของ episode boundary ว่าเป็นการสิ้นสุดและปิดสถานะ หรือการตัดช่วงที่ยังมีมูลค่าต่อเนื่อง แล้วจัด `terminated / truncated` และ PPO bootstrapping ให้สอดคล้องกัน
- [x] REWARD-06 ตรวจความสอดคล้องในโหมด realized, MTM และ custom formula รวมถึง action profiles ที่รองรับ
- [x] REWARD-07 บันทึกเวอร์ชันพฤติกรรม environment/reward ใน run metadata เพื่อแยกโมเดลก่อนและหลังแก้

### เกณฑ์ทดสอบก่อนปิดชุดงาน

- [x] REWARD-T01 ปิดสถานะเดียวกันที่ราคาและต้นทุนเดียวกันด้วย Close กับ episode end แล้วได้ net PnL และส่วน reward จากการปิดเท่ากัน
- [x] REWARD-T02 แต่ละสถานะถูกปิดและบันทึก trade เพียงครั้งเดียว แม้ Max Hold หรือ stop เกิดตรงท้าย episode
- [x] REWARD-T03 Hold/Close ตอนว่างไม่เปิดช่องให้หลบ idle penalty ตามกติกาที่กำหนด
- [x] REWARD-T04 equity สุดท้ายตรงกับปลาย equity curve และตรวจผลตอบแทน/drawdown จากชุดข้อมูลที่รู้คำตอบ
- [x] REWARD-T05 รัน PPO สั้นด้วยทั้ง `basic_4` และ `manage_6` แล้วไม่มี observation/reward ผิดรูปแบบหรือ NaN/Inf

หมายเหตุ: โมเดลเก่ายังเก็บไว้ใช้เปรียบเทียบได้ แต่ต้อง Train ใหม่เพื่อเรียนรู้ reward ที่แก้แล้ว การแก้ source ไม่ได้เปลี่ยนน้ำหนักโมเดลเดิม

## ชุดที่ 3: แยกข้อมูลและประเมินผลให้ถูกต้อง

ความสำคัญ: ป้องกันข้อมูล Test มีส่วนในการเตรียม Train และป้องกันผู้ใช้ตีความผลประเมินผิด

ไฟล์หลัก: [rl_app.py](../rl_app.py), [rl_train.py](../rl_train.py), [rl_walkforward.py](../rl_walkforward.py)

### งานที่ต้องแก้

- [x] DATA-01 ใช้กติกาโหลดและตรวจข้อมูลร่วมกันใน Data Tools, Train, Pipeline และ Walk-Forward
- [x] DATA-02 ตรวจ timestamp ที่หาย/แปลงไม่ได้/ซ้ำ, ลำดับเวลา, ราคาที่ไม่ถูกต้อง, NaN/Inf และฟีเจอร์ขาด พร้อมแยกข้อผิดพลาดที่ต้องหยุดกับคำเตือน
- [x] DATA-03 แสดงคอลัมน์ค่าคงที่ และบันทึกว่าจะตัดหรือเก็บคอลัมน์ใดด้วยเหตุผลอะไร
- [x] DATA-04 กำหนดช่วง Train/Validation/Test ก่อนเลือกฟีเจอร์ โดยแสดงช่วงวันที่และจำนวนแถวจริงให้ผู้ใช้เห็น
- [x] DATA-05 ให้ Clean Redundant Features คำนวณรายการฟีเจอร์จาก Train เท่านั้น แล้วใช้รายการเดียวกันกับ Validation/Test
- [x] DATA-06 บันทึกรายชื่อและลำดับฟีเจอร์, correlation threshold และช่วงข้อมูลที่ใช้เลือกฟีเจอร์ไว้กับ run
- [x] DATA-07 คง normalization จาก Train เท่านั้น และหยุดพร้อมข้อความชัดเจนเมื่อชุดประเมินขาดฟีเจอร์ที่จำเป็น
- [x] DATA-08 ให้ Walk-Forward เลือกฟีเจอร์และคำนวณ normalization ใหม่จาก Train ของแต่ละรอบ; ใช้ source ที่ยังไม่ผ่านการเลือกฟีเจอร์ด้วยข้อมูลอนาคต
- [x] DATA-09 แยก Validation สำหรับติดตาม/เลือก checkpoint ออกจาก Test สุดท้าย และตรวจไม่ให้ช่วงวันที่ซ้อนกันโดยไม่ตั้งใจ
- [x] DATA-10 Train 100% ต้องแสดงว่าไม่มีข้อมูลทดสอบที่ไม่เคยเรียน หรือให้เลือกชุดประเมินแยก; ห้ามเรียก train-tail fallback ว่า out-of-sample
- [x] DATA-11 ทำข้อความและ metadata ของ Train, Pipeline และผลประเมินให้บอกตรงกันว่าใช้ข้อมูลช่วงใดและมีบทบาทอะไร
- [x] DATA-12 ตรวจต้นทาง Divergence ใน `usdjpy_h4_2025.csv` และ `xauusd_h4_2025.csv` ว่า collector/indicator คำนวณและส่งค่าจริงหรือไม่ ก่อนตัดสินใจสร้างข้อมูลใหม่หรือตัดฟีเจอร์

### เกณฑ์ทดสอบก่อนปิดชุดงาน

- [x] DATA-T01 เปลี่ยนเฉพาะข้อมูล Test แล้วรายการฟีเจอร์และค่า normalization ของ Train ต้องไม่เปลี่ยน
- [x] DATA-T02 ทุก Walk-Forward window ใช้ข้อมูลเตรียม Train ที่ไม่เกินขอบเขตเวลาของรอบนั้น
- [x] DATA-T03 CSV ที่ timestamp เสีย, ราคาผิด, มี Inf หรือขาดฟีเจอร์ ต้องได้ผลตรวจที่ถูกต้องก่อนเริ่ม Train
- [x] DATA-T04 Train 100% และไฟล์ประเมินที่ซ้อนช่วง Train แสดงสถานะ/คำเตือนถูกต้อง ไม่รายงานเป็นการทดสอบข้อมูลใหม่
- [ ] DATA-T05 สรุปสาเหตุ Divergence ค่าคงที่จากหลักฐานต้นทาง; หากสร้าง CSV ใหม่ให้ตรวจค่าที่เปลี่ยนจริงและ sidecar ที่ตรงกัน

### ข้อมูลตั้งต้นจากการตรวจ CSV หลัก

| ไฟล์ | แถว | ฟีเจอร์ | คอลัมน์ค่าคงที่ |
|---|---:|---:|---:|
| `uj_h4_extra_features.csv` | 8,493 | 183 | 1 |
| `uj_h4_extra_features_clean.csv` | 8,493 | 122 | 0 |
| `usdjpy_h4_2025.csv` | 17,143 | 202 | 11 |
| `xauusd_h4_2025.csv` | 17,593 | 202 | 11 |

ทั้งสี่ไฟล์เรียงเวลา ไม่มี timestamp ซ้ำ ไม่มี NaN/Inf ในฟีเจอร์ และมี `.params.json` คู่กัน ณ วันที่ตรวจ ตัวเลขนี้ไม่ได้ยืนยันว่าทุกฟีเจอร์คำนวณถูกต้องหรือปราศจาก look-ahead

สองไฟล์ปี 2025 มีสัญญาณ Divergence 8 คอลัมน์เป็น 0 ตลอด และ age 2 คอลัมน์เป็น 1 ตลอด ยังไม่ยืนยันสาเหตุว่าเป็นปัญหาที่ indicator, collector หรือกระบวนการสร้างข้อมูล

## ชุดที่ 4: Fine-tune และ Dashboard ใช้ค่าตรงกัน

ความสำคัญ: ทำให้ค่าที่ผู้ใช้เลือกตรงกับสิ่งที่ระบบใช้จริง

ไฟล์หลัก: [rl_finetune.py](../rl_finetune.py), [rl_app.py](../rl_app.py), [backtest_live.py](../backtest_live.py), [export_to_onnx.py](../export_to_onnx.py)

### งานที่ต้องแก้

- [x] FT-01 ส่ง reward config/overrides/formula และ action config ของโมเดลเดิมเข้า Fine-tune ให้ครบ และตรวจ dimension ก่อนเริ่ม
- [x] FT-02 ปรับ learning rate พร้อม schedule ที่ PPO ใช้จริง ไม่ใช่เปลี่ยนเฉพาะค่าบน optimizer ชั่วคราว
- [x] FT-03 เปลี่ยน Mixed sampling จากสุ่มรายแถวเป็นช่วงแท่งต่อเนื่อง หรือแยก episode ระหว่างข้อมูลเก่ากับข้อมูลใหม่
- [x] FT-04 ไม่ให้ observation window หรือสถานะที่ถืออยู่ข้ามรอยต่อของช่วงข้อมูลที่นำมาต่อกัน และรายงานสัดส่วน old/new ที่ได้จริง
- [x] FT-05 ใช้การบันทึก artifact ที่ปลอดภัยจากชุดที่ 1 กับผล Fine-tune ด้วย
- [x] UI-01 สำหรับ Dashboard สอน PPO ให้ซ่อน DQN/A2C จนกว่าจะรองรับครบ Train -> Backtest -> Analyze -> Export และแจ้งชัดเมื่อเลือกโมเดลชนิดที่ไม่รองรับ
- [x] UI-02 แสดงสรุปก่อนเริ่ม: algorithm, ช่วงข้อมูล, จำนวนฟีเจอร์, window, steps, reward/action config, การแบ่งชุด และตำแหน่งบันทึก
- [x] UI-03 ให้ Train/Pipeline/Walk-Forward แสดงและส่งค่าตั้งเดียวกันอย่างสอดคล้อง พร้อมตรวจค่าที่ไม่ถูกต้องก่อนเริ่ม process
- [x] UI-04 อัปเดตคู่มือ flow ให้ตรงกับพฤติกรรมที่แก้แล้ว โดยอธิบายขั้นตอนใช้งานก่อนรายละเอียด

### เกณฑ์ทดสอบก่อนปิดชุดงาน

- [x] FT-T01 Fine-tune ใช้ reward override และ action config ตรงกับ metadata ของโมเดลเดิม
- [x] FT-T02 ตรวจ learning rate ที่ optimizer ใช้หลัง PPO update แรกและ update ถัดไปว่าตรงกับค่าที่ตั้ง
- [x] FT-T03 ตรวจ timestamp ใน episode/window ว่าไม่กระโดดข้ามช่วงที่สุ่มหรือรอยต่อ old/new
- [x] UI-T01 สรุปหน้าแอป, arguments ที่ส่งให้ subprocess และ metadata ที่บันทึกต้องตรงกัน
- [x] UI-T02 ไม่มีตัวเลือก algorithm ที่ทำให้ผู้ใช้ Train สำเร็จแล้วติดขั้นตอนถัดไปโดยไม่มีข้อความอธิบาย

## ตรวจครบ Flow ก่อนปิดงานทั้งหมด

- [x] FLOW-01 รัน regression tests เดิมและ tests ใหม่ของทั้ง 4 ชุดให้ผ่าน
- [x] FLOW-02 รัน Train สั้นด้วยข้อมูลจำลอง/ชุดทดสอบ แล้วโหลด artifact ที่ได้กลับได้
- [x] FLOW-03 รัน Backtest ของโมเดลทดสอบ และตรวจ feature/action/window/normalization ตรงกับ Train
- [x] FLOW-04 รัน Walk-Forward สั้นด้วย recipe เดียวกัน แล้วตรวจช่วง Train/Test ของแต่ละรอบ
- [x] FLOW-05 รัน Fine-tune สั้น แล้วตรวจ learning rate, reward config และการรักษาโมเดลต้นฉบับ
- [x] FLOW-06 รัน Export และตรวจ output ของ ONNX เทียบกับโมเดลต้นฉบับบน observation ชุดเดียวกัน
- [ ] FLOW-07 ตรวจหน้า Dashboard ที่เกี่ยวข้อง รวม Start/Stop/error/completed และการเปิดใช้โมเดลเดิม
- [x] FLOW-08 เมื่อ smoke tests ผ่าน จึง Train รอบยาวด้วยชื่อโมเดลใหม่ แล้วเปรียบเทียบผลด้วยข้อมูลและกติกาประเมินที่ควบคุมไว้
- [x] FLOW-09 สรุปสิ่งที่แก้, tests ที่ผ่าน, ข้อจำกัด และรายการที่ยังไม่ได้ทดสอบ โดยระบุแยกว่ามีหรือยังไม่มีหลักฐานจาก MT5

## บันทึกหลักฐานการแก้

เติมหนึ่งแถวต่อชุดงานหรือประเด็นที่ปิดได้ ไม่ติ๊กผ่านจากการอ่านโค้ดอย่างเดียวเมื่อรายการนั้นต้องทดสอบพฤติกรรม

| วันที่ | รหัสงาน | สิ่งที่แก้ | คำสั่ง/วิธีทดสอบ | ผลและไฟล์หลักฐาน | ข้อจำกัดที่เหลือ |
|---|---|---|---|---|---|
| 2026-09-05 | SAVE / REWARD / DATA / FT | แยก artifact ต่อ run, แก้ liquidation, preprocessing และ fine-tune | `test_training_improvements` + `test_batch1_bugfixes` | ผ่าน 40 tests; ดู [สรุปการแก้](training_changes_2026-09-05.md) | ไม่ใช่หลักฐาน live/MT5 |
| 2026-09-05 | FLOW-02 ถึง FLOW-06 | Train 4/6 actions, Train 100%, failed retrain, Backtest, Analyze, Fine-tune, WF, Export | `tools/analysis/training_smoke_check.py` | [11 integration checks](rl_training_validation_2026-09-05.json) ผ่าน; ONNX 50 observations | ข้อมูลจำลอง; outputs ชั่วคราว |
| 2026-09-05 | SAVE-T04 | โหลดโมเดลเก่าผ่าน path helpers และ PPO | โหลด `rl_uj_extra` โดยไม่แก้ artifact | 183 features, obs 3663, 4 actions | ตรวจโหลด ไม่ใช่ยืนยันกำไร |
| 2026-09-05 | FLOW-08 | Train จริง 200,704 steps ด้วยชื่อแยก | `codex_training_check_20260905`; เปรียบเทียบ final/best บน Test 637 แถวเดียวกัน | [รายงานรอบยาว](rl_full_training_validation_2026-09-05.json) | final -0.27%, best -6.49%; ยังไม่พร้อม deploy |
| 2026-09-05 | DATA-12 | เทียบ indicator source ที่ติดตั้งกับ repo และคำนวณ divergence จากราคาเดิม | hashes + Python divergence recomputation | พบ source คนละรุ่น และพบสัญญาณหลายร้อยครั้งจาก CSV เดิม | ยังไม่ยืนยัน binary MT5/เก็บข้อมูลใหม่ |

## รายการที่ยังไม่ปิด

UI-02 / UI-03 / UI-T01 ปิดแล้ว: หน้าสรุป Train/Pipeline ใช้ตัวเตรียมข้อมูลเดียวกับ CLI, ตรวจค่าตัวเลขโดยไม่ fallback/clamp เงียบ ๆ, ส่ง custom action JSON ต่อถึง Walk-Forward และเทียบ UI -> arguments -> metadata ด้วยการฝึกสั้นจริง ดู [หลักฐานรอบ UI](ui_training_contract_validation_2026-09-05.md): tests 54/54 และ integration 11/11 ผ่าน

- **DATA-T05:** พบ source indicator ที่ติดตั้งเป็นรุ่นเก่าซึ่งคำนวณบนแท่งกำลังก่อตัว แต่ยังไม่ได้ compile/install แล้วเก็บข้อมูลใน MT5 ใหม่ จึงยังไม่ยืนยันสาเหตุครบจาก binary ที่ใช้งานจริง และยังไม่แก้ CSV ของผู้ใช้
- **FLOW-07:** สร้างหน้าจอ Tk จริงและตรวจ handlers ด้วย tests แล้ว ยังเหลือทดสอบผู้ใช้กด Start/Stop/ปิดหน้าต่าง/error/completed ครบเส้นทางกับ subprocess จริงทุกหน้า ไม่ใช้ผล CLI แทนหลักฐานนี้

รายงานฉบับอ่านง่าย: [training_changes_2026-09-05.md](training_changes_2026-09-05.md)
