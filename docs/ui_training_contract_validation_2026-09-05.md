# ผลตรวจ UI-02 / UI-03 / UI-T01

วันที่: 2026-09-05

สถานะ: ปิดทั้ง 3 รายการแล้ว; checklist รวม 62/64

## วิธีใช้หลังอัปเดต

1. ปิดแล้วเปิด Dashboard ใหม่
2. ตั้งค่าใน Train หรือ Full Pipeline แล้วกดเริ่ม
3. ระบบตรวจข้อมูลก่อนและเปิดหน้าสรุป โดยยังไม่เริ่ม subprocess ฝึก
4. ตรวจจำนวนฟีเจอร์หลังคัดเลือก, Train/Validation/Test พร้อมวันและจำนวนแถว, window/steps/network, reward/action ที่ resolve แล้ว, JSON/overrides/formula และโฟลเดอร์บันทึก
5. กด Start training / Start pipeline เพื่อยืนยัน หรือ Cancel เพื่อกลับไปแก้

Pipeline แสดงช่วง Backtest และ M1 ที่ resolve แล้วด้วย เมื่อเลือก start = auto จะเริ่มจาก Test ที่ไม่ถูกใช้เลือก checkpoint; Train 100% ที่ไม่มี external Test ระบุชัดว่าเป็น in-sample diagnostic

## สิ่งที่เปลี่ยน

- `rl_train.prepare_training()` เป็นจุดเดียวสำหรับหน้าสรุปและ CLI: ตรวจข้อมูล, แบ่งเวลา, คัดฟีเจอร์, normalize และ resolve recipe ก่อนสร้างโฟลเดอร์ run
- ส่ง `--expected_recipe_sha256` จากหน้าสรุป ตรวจข้อมูล/JSON/sidecar ก่อนและหลังเตรียมข้อมูล และตรวจ fingerprint อีกครั้งใน subprocess ก่อนฝึก
- Metadata เก็บ `confirmed_recipe` และ `recipe_sha256` ไว้เทียบค่าที่ผู้ใช้ยืนยันกับการรันจริง
- `ui_values.py` ตรวจตัวเลข/เปอร์เซ็นต์ร่วมกัน ไม่เติม default เมื่อว่าง ไม่รับ NaN/Inf หรือจำนวนเต็มที่มีเศษ
- Train %, Feature Train %, MC Skip %, Backtest Start และ Fine-tune Mix รับรูปแบบ `85`, `85%`, `0.85` โดยใช้กฎเดียวกัน; `1` = 100%, `1%` = 1% ขอบเขตเปิด/ปิดขึ้นกับหน้าที่ของช่อง
- Confidence และ Risk ใช้สัดส่วน 0..1; Swap/Stop Slippage ใช้หน่วยเปอร์เซ็นต์ตามป้ายช่อง ไม่แปลงด้วย heuristic เดียวกับ Train %
- ช่อง reward/action ที่ผิดขอบเขตแสดงข้อผิดพลาดและคงข้อความให้แก้ ไม่ปรับค่าลงมาเอง; period ของ action ต้องเป็นจำนวนเต็มทั้ง UI และ JSON
- Backtest Window ใช้ `auto` เป็นค่าเริ่มต้นที่มองเห็นได้ ช่องว่างไม่ใช่ auto อีกต่อไป
- Walk-Forward Copy ไม่เติมค่าช่อง Train ที่ว่าง และส่ง custom action JSON ต่อ ไม่สูญเสียรายการ action ที่ต่างจาก preset

## ผลทดสอบ

| ชุด | ผล | หลักฐานที่ตรวจ |
|---|---|---|
| `test_ui_training_contract` | 14/14 ผ่าน | สร้าง Tk widgets จริง, จับคำสั่ง, ฝึกสั้นจริง แล้วอ่าน metadata เทียบกับ confirmed recipe |
| `test_training_improvements` + `test_batch1_bugfixes` | 40/40 ผ่าน | Regression ด้านข้อมูล/reward/artifact/lifecycle handlers และค่าจากหน้าแอป |
| `training_smoke_check.py` | 11/11 ผ่าน | Train 4/6 actions, failed retrain, Train 100%, Backtest, chart, Analyze, Fine-tune, WF, Export, ONNX parity |
| Syntax | ผ่าน 7 ไฟล์ | `rl_app`, `rl_train`, `training_data`, `ui_values`, `action_profiles` และ tests ที่แก้ |
| HTML anchors | ผ่าน 3 คู่มือ | Train, Backtest, Walk-Forward ไม่มี ID ซ้ำหรือลิงก์หัวข้อขาด |
| Git diff whitespace | ผ่าน | `git diff --check` ไม่มี error |

ONNX เทียบ PPO 50 observations: maximum absolute error `2.9802322387695312e-08`

### เคส UI ที่ครอบคลุม

- Resolve ทุก reward preset กับทั้ง basic_4/manage_6 รวม 10 คู่; Train 100%/MTM แสดงว่าไม่มี unseen Test
- ฝึกจากคำสั่งหน้า Train ด้วยค่าปกติ, reward/action JSON, slider overrides และ formula เปิด/ปิด แล้วเทียบ recipe/config/hyperparameters/features/ช่วงวันที่ใน metadata
- Pipeline ใช้ external CSV, overrides และ formula แล้วรัน Train -> Backtest -> chart จริง; Backtest เริ่มที่ครึ่งหลังของ external CSV ตาม Test ไม่รวม Validation
- Custom action JSON 5 actions ผ่าน UI Train preparation และ Copy ไป Walk-Forward; รัน WF สั้นจริง 2 windows แล้วเทียบ recipe ใน preprocessing JSON
- ตัวเลขผิดบน Train, Pipeline, Walk-Forward, Backtest, Fine-tune, Feature Analysis/Clean และ Regime Check ไม่เริ่ม runner/worker
- Cancel ไม่เริ่มฝึก; ปุ่มยืนยันแนบ fingerprint; sidecar เปลี่ยนหลังยืนยันแล้วถูกปฏิเสธก่อนสร้าง artifact

## รันทดสอบซ้ำ

จาก project root:

```powershell
.\.venv\Scripts\python.exe -m unittest test_ui_training_contract -v
.\.venv\Scripts\python.exe -m unittest test_training_improvements test_batch1_bugfixes -v
.\.venv\Scripts\python.exe tools/analysis/training_smoke_check.py --report "$env:TEMP\rl_ui_contract_smoke.json"
```

การฝึกทดสอบใช้ข้อมูลจำลอง 600 แถวและช่วง external 120 แถว โดยตั้ง 32 steps เพื่อทดสอบการส่งค่า ไม่ใช่ประเมินกำไร โมเดล/รายงานทดสอบอยู่ใน temporary directories และถูกล้างเมื่อจบ ไม่เขียนทับ CSV หรือโมเดลผู้ใช้

## ข้อจำกัด

- ทดสอบตัวแทนของ recipe และขอบเขตสำคัญ ไม่ใช่ exhaustive test ทุกค่าตัวเลขหรือ JSON ที่เป็นไปได้
- สร้าง Tk UI และเรียก handlers/buttons ด้วย tests แต่ยังไม่ใช่ manual visual QA หรือการกด Start/Stop/ปิดแอปกับ subprocess จริงครบทุกหน้า; `FLOW-07` ยังเปิดอยู่
- ไม่ compile/install/recollect ใน MT5 และไม่ได้พิสูจน์ผล live; `DATA-T05` ยังเปิดอยู่
- ไม่ได้ commit หรือ push ในรอบนี้
