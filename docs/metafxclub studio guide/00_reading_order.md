# Metafxclub Studio Guide Plan

โฟลเดอร์นี้เก็บเฉพาะไฟล์ที่เกี่ยวกับการทำคู่มือ flow การใช้งาน dashboard และไฟล์ deep dive ที่แตกจาก flow นั้นโดยตรง

## Current Guide Files

1. [01_dashboard_user_flow_guide.html](01_dashboard_user_flow_guide.html)
   - คู่มือ flow หลักของ dashboard: Data Prep -> Quality/Regime Check -> Train -> Backtest -> Walk-Forward -> Export
   - ใช้เป็นหน้ากลางสำหรับสอน user และเป็นแผนที่ว่าจะต้องทำ deep dive รายส่วนอะไรต่อ

2. [02_data_prep_import_detail.html](02_data_prep_import_detail.html)
   - บทละเอียดของ Step 1: เตรียมข้อมูลจาก MT5
   - ครอบคลุม DataCollector_RL, MT5 Common Files, CSV, `.params.json`, import flow และ checklist ว่า data พร้อมใช้หรือยัง

3. [03_quality_check_detail.html](03_quality_check_detail.html)
   - บทละเอียดของ Step 2: เช็คคุณภาพข้อมูลและ feature ก่อน train
   - ครอบคลุม file/timestamp checks, Feature Analysis, correlation threshold, Clean Redundant Features, Regime Check และ cutoff dataset

4. [04_train_page_detail.html](04_train_page_detail.html)
   - บทละเอียดของ Step 3: Train โมเดล RL
   - ครอบคลุม dataset/params, PPO, reward mode, training steps, Train %, window size, max hold, advanced PPO และการอ่าน log

5. [05_backtest_page_detail.html](05_backtest_page_detail.html)
   - บทละเอียดของ Step 4: Validation ด้วย Backtest
   - ครอบคลุม model/test CSV, confidence threshold, Pure Agent vs Agent + SL/TP, risk settings, metric interpretation และ decision gate

6. [06_walk_forward_detail.html](06_walk_forward_detail.html)
   - บทละเอียดของ Step 5: Robustness ด้วย Walk-Forward
   - ครอบคลุมเหตุผลที่หน้านี้ retrain model ชั่วคราวทุก window, validate training recipe ไม่ใช่ saved model file, window split, settings, metrics และ robust verdict

## Reference Files

ไฟล์ explainer เก่าที่เป็น background หรือเอกสารเฉพาะเรื่องถูกแยกไว้ที่ `../explainers/` แล้ว เพื่อไม่ให้ปนกับชุดคู่มือ flow นี้

ไฟล์แผนที่โฟลเดอร์ของโปรเจกต์:

- [90_project_folder_map.html](90_project_folder_map.html)
  - อธิบาย root folder, source code, docs, artifacts, mt5_files, reference, tools และพื้นที่ระบบ
  - ใช้เป็น reference ตอน user ต้องการรู้ว่าไฟล์ train/backtest/export ถูกเก็บที่ไหน และโฟลเดอร์ไหนไม่ควรแก้มือ

ตัวอย่างไฟล์อ้างอิงที่อาจลิงก์ไปอ่านต่อ:

- [Data Tools Modules](../explainers/04_data_tools_modules_explained.html)
- [DataCollector v4 Explained](../explainers/05_data_collector_v4_explained.html)
- [Parity Config Explained](../explainers/06_parity_config_explained.html)
- [RL Reward Explained](../explainers/07_rl_reward_explained.html)

## Future Deep-Dive Slots

ไฟล์ที่จะทำต่อในโฟลเดอร์นี้ควรเป็นไฟล์ที่แตกจาก flow หลักโดยตรง:

- `07_export_mt5_detail.html` สำหรับสอน export ONNX, config, EA และการเอาไปใช้ใน MT5 Strategy Tester

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
