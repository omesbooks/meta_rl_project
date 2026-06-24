# Metafxclub Studio Guide Plan

โฟลเดอร์นี้เก็บเฉพาะไฟล์ที่เกี่ยวกับการทำคู่มือ flow การใช้งาน dashboard และไฟล์ deep dive ที่แตกจาก flow นั้นโดยตรง

## Current Guide Files

1. [03_dashboard_user_flow_guide.html](03_dashboard_user_flow_guide.html)
   - คู่มือ flow หลักของ dashboard: Data Prep -> Train -> Backtest -> Walk-Forward -> Export
   - ใช้เป็นหน้ากลางสำหรับสอน user และเป็นแผนที่ว่าจะต้องทำ deep dive รายส่วนอะไรต่อ

2. [15_data_prep_import_detail.html](15_data_prep_import_detail.html)
   - บทละเอียดของ Step 1: เตรียมข้อมูลจาก MT5
   - ครอบคลุม DataCollector_RL, MT5 Common Files, CSV, `.params.json`, import flow และ checklist ว่า data พร้อมใช้หรือยัง

## Reference Files

ไฟล์ explainer เก่าที่เป็น background หรือเอกสารเฉพาะเรื่องถูกแยกไว้ที่ `../explainers/` แล้ว เพื่อไม่ให้ปนกับชุดคู่มือ flow นี้

ตัวอย่างไฟล์อ้างอิงที่อาจลิงก์ไปอ่านต่อ:

- [Data Tools Modules](../explainers/04_data_tools_modules_explained.html)
- [DataCollector v4 Explained](../explainers/05_data_collector_v4_explained.html)
- [Parity Config Explained](../explainers/06_parity_config_explained.html)
- [RL Reward Explained](../explainers/07_rl_reward_explained.html)

## Future Deep-Dive Slots

ไฟล์ที่จะทำต่อในโฟลเดอร์นี้ควรเป็นไฟล์ที่แตกจาก flow หลักโดยตรง:

- `16_train_page_detail.html` สำหรับสอนหน้า Train ทีละช่อง เช่น steps, window, train pct, max hold, reward mode
- `17_backtest_page_detail.html` สำหรับสอนอ่าน PF, return, drawdown, trades, equity curve และ confidence threshold
- `18_walk_forward_detail.html` สำหรับสอน walk-forward split, robust verdict และการอ่านผลหลายช่วงเวลา
- `19_regime_check_detail.html` สำหรับสอน regime detection, breakpoint table, Gemini labels และ train cutoff
- `20_export_mt5_detail.html` สำหรับสอน export ONNX, config, EA และการเอาไปใช้ใน MT5 Strategy Tester

## Rule For This Folder

เก็บไว้เฉพาะ:

- flow overview
- flow planning/index
- deep dive ที่เจาะแต่ละ step ของ flow

ไม่เก็บไว้ในโฟลเดอร์นี้:

- explainer เก่าแบบ background กว้าง ๆ
- legacy/PyCaret explanation
- mockup/reference UI เก่า
- production docs ที่ไม่ได้เป็นบทใน flow guide โดยตรง
