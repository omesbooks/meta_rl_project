# Codex Custom Action Prompts

ไฟล์นี้เก็บ prompt สำหรับสั่ง Codex ให้เพิ่มหรือปรับ Action Profile ของระบบ RL
โดยต้องแก้ครบทั้ง flow: `train -> backtest -> fine-tune -> walk-forward -> full pipeline -> export -> EA live`

ใช้เมื่อผู้ใช้ต้องการเพิ่ม action ใหม่ เช่น `partial_close`, `scale_in`, `trail_sl_atr`,
`move_sl_breakeven` หรือ action จัดการ position รูปแบบอื่น ๆ

## Prompt: เพิ่มหรือปรับ Action Profile แบบละเอียด

คัดลอก prompt นี้ไปวางใน Codex แล้วแทนค่าช่อง `<...>` ให้ชัดเจน:

```text
อยู่ในโปรเจค C:\Users\omesb\Documents\claude code\pycaret_trainer

ช่วยเพิ่ม/ปรับ action ของ RL ให้ครบทั้ง flow:
dataset → train → backtest → fine-tune → walk-forward → full pipeline → export → EA live

Action ที่ต้องการ:
- Action profile name: <เช่น Manage 8 / Custom Manage>
- ต้องการเพิ่ม action ใหม่ชื่อ: <เช่น trail_sl_atr / move_sl_breakeven / partial_close>
- ความหมายของ action: <อธิบายว่า action นี้ทำอะไร>
- ใช้กับ position แบบไหน: <buy / sell / both / only when position exists>
- เงื่อนไขที่ action ใช้ได้: <เช่น ต้องมีกำไรลอยตัวเกิน 0.0010 ก่อน>
- ถ้า action ใช้ไม่ได้ให้ทำอะไร: <ignore / penalty / treat as hold>
- parameter ที่ user ปรับได้:
  - <เช่น trail_atr_period=14>
  - <เช่น trail_atr_mult=2.0>
  - <เช่น breakeven_min_profit=0.0010>
- ต้องการให้ parameter นี้บันทึกลง model metadata / export config ไหม: ใช่
- ต้องการให้ EA live รองรับ action นี้ด้วยไหม: ใช่

ข้อกำหนดสำคัญ:
1. อ่าน action_profiles.py, trading_env.py, backtest_live.py, rl_train.py, rl_app.py, export_to_onnx.py และ mt5_files\MQL5\Experts\ML_RL_Trader_template.mq5 ก่อน
2. แก้ action_profiles.py:
   - เพิ่ม action primitive ถ้าจำเป็น
   - เพิ่ม/แก้ preset action profile
   - เพิ่ม default params และ label ที่แสดงใน GUI
3. แก้ trading_env.py:
   - ปรับ action space ให้ตรงกับจำนวน action ใหม่
   - เพิ่ม logic การ execute action ระหว่าง train
   - เพิ่ม invalid action penalty ถ้าจำเป็น
   - ตรวจว่า action ไม่ใช้ future data
4. แก้ backtest_live.py:
   - ให้ backtest ใช้ action profile เดียวกับตอน train
   - เพิ่ม logic action ใหม่ให้ตรงกับ train เท่าที่ทำได้
5. แก้ rl_train.py / rl_finetune.py / rl_walkforward.py:
   - ส่ง action_profile และ action params เข้า env ให้ครบ
   - บันทึกลง train metadata เพื่อให้ export รู้ว่า model นี้ train ด้วย action profile ไหน
6. แก้ rl_app.py:
   - เพิ่ม option ในหน้า Train
   - เพิ่ม option ใน Full Pipeline
   - เพิ่ม/ปรับ Action Parameter UI ถ้าจำเป็น
   - ให้ Backtest/Fine-tune/Walk-forward ใช้ action profile ตรงกับ model
7. แก้ export_to_onnx.py:
   - อ่าน action profile จาก model metadata
   - generate config ให้ EA รู้ action id และ parameter ของ action ใหม่
8. แก้ mt5_files\MQL5\Experts\ML_RL_Trader_template.mq5:
   - เพิ่ม logic ฝั่ง EA live ให้ execute action ใหม่ได้
   - แยก logic buy/sell ให้ถูกต้อง
   - ตรวจว่า action ใหม่ไม่เปิดออเดอร์มั่ว และไม่แก้ SL ผิดฝั่ง
9. อัปเดต docs ถ้าจำเป็น โดยเฉพาะส่วน action profile / custom action
10. รันเช็ค:
   - git diff --check
   - .venv\Scripts\python.exe -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['action_profiles.py','trading_env.py','backtest_live.py','rl_train.py','rl_finetune.py','rl_walkforward.py','export_to_onnx.py','rl_app.py']]; print('python ast ok')"
   - compile MQL5 template/EA ถ้าทำได้ หรือบอกชัดเจนว่าต้อง compile ใน MetaEditor
11. สรุป:
   - action ใหม่มี id อะไร
   - profile ไหนมี action นี้
   - parameter ไหนปรับได้
   - model เก่าต้อง retrain/export ใหม่ไหม
   - ไฟล์ที่แก้มีอะไรบ้าง
```

## Prompt: เพิ่ม Action แบบสั้น

ใช้ prompt นี้ถ้า action ไม่ซับซ้อน และต้องการให้ Codex ไล่แก้ครบ flow เอง:

```text
ช่วยเพิ่ม action ใหม่เข้า RL action profile ให้ครบ train/backtest/export/EA live

Action:
- ชื่อ:
- ทำอะไร:
- ใช้ได้เมื่อ:
- parameter ที่ปรับได้:
- ถ้าใช้ไม่ได้ให้ penalty เท่าไหร่:
- ต้องรองรับใน EA live: ใช่

ให้แก้ครบ action_profiles.py, trading_env.py, backtest_live.py, rl_train.py, rl_finetune.py, rl_walkforward.py, rl_app.py, export_to_onnx.py และ ML_RL_Trader_template.mq5
แล้วรันเช็ค/compile พร้อมสรุปผล
```

## ข้อควรจำ

- Action Profile เป็นส่วนหนึ่งของ architecture ไม่ใช่ filter หลัง train
- จำนวน action คือ output dimension ของ PPO policy
- เปลี่ยน action space แล้ว model เก่าใช้ต่อไม่ได้ ต้อง train ใหม่และ export EA ใหม่
- Backtest, Walk-Forward, Fine-tune, Full Pipeline และ EA live ต้องใช้ action meaning เดียวกันเสมอ
- Developer JSON/DSL ควรจำกัดเฉพาะ primitive ที่ระบบรองรับ ไม่ควรปล่อยให้ผู้ใช้เขียน Python/MQL5 อิสระจาก GUI
