# Codex Bug Fix Prompts — audit 2026-07-12

ไฟล์นี้เก็บ prompt สำหรับสั่ง Codex ไล่แก้บั๊กจากผล audit ทั้งโปรแกรม
รายการบั๊กเต็ม 47 ข้อ (พร้อม trigger + fix sketch ต่อข้อ) อยู่ที่ `docs/bug_audit_2026-07-12.md`
— **ทุก prompt ด้านล่างสั่งให้อ่านไฟล์นั้นก่อนเสมอ** เลข item ในนี้อ้างอิงเลขในไฟล์นั้น

แนะนำให้รันทีละ batch ตามลำดับ (1 → 2 → 3 → 4) เพราะ Batch 2 แก้โครงสร้าง
ProcessRunner ที่หลายข้อใน Batch 3 อ้างถึง

## กติการ่วมทุก batch (แนบท้าย prompt ทุกครั้ง)

```text
กติกาการทำงาน:
- อ่าน docs/bug_audit_2026-07-12.md ก่อนเริ่ม — ทุก item มี Trigger และ Fix sketch ให้แล้ว
  ให้ยึด fix sketch เป็นแนวทาง แต่ถ้าเจอวิธีที่ดีกว่าให้อธิบายเหตุผลใน commit message
- python ใช้ .venv\Scripts\python.exe เสมอ และตั้ง PYTHONIOENCODING=utf-8
- ห้ามรัน training จริง (rl_train/rl_walkforward/rl_finetune แบบเต็ม) — ถ้าต้องพิสูจน์ end-to-end
  ให้ใช้ steps จิ๋ว (--steps 300 --n_steps 256) กับ uj_h4_dataset.csv แล้วลบ artifacts ทดสอบทิ้ง
- GUI test แบบ headless: import rl_app; app = rl_app.RLTradingStudio(); app.show_page(...);
  เช็ค widget ด้วย winfo_manager()/grid_info()/cget แล้ว app.destroy()
  (มี pattern ตัวอย่างใน git log ของ commit ช่วง 2026-07-11 ถึง 12)
- ข้อควรระวังของ codebase นี้:
  - ScrollableOptionMenu ไม่มี cget("values") — ใช้ getattr(w, "_values", []);
    .set() ไม่ยิง command callback; .configure(values=...) auto-เลือก values[0] ถ้าค่าปัจจุบันคือ "(none)"
  - แถว grid ใน rl_app.py ชนกันง่าย — เช็คเลข row รอบข้างก่อน insert widget ใหม่
  - artifacts/ ถูก gitignore แต่มีบางไฟล์ tracked อยู่ — ใช้ git add -u artifacts/ ถ้าจำเป็น
  - ห้ามเปลี่ยนชื่อ/ความหมาย CLI flag เดิมของ engine scripts (GUI กับ pipeline เรียกใช้อยู่)
  - ข้อความ UI ใหม่ให้เขียนไทยปนอังกฤษตามสไตล์เดิมของหน้า
- เมื่อแก้ item ไหนเสร็จและเทสผ่าน: ติ๊ก checkbox `- [x] **fix**` ของ item นั้นใน
  docs/bug_audit_2026-07-12.md ใน commit เดียวกัน
- ถ้าบั๊กไหนเป็น gotcha ที่คนจะเจอซ้ำ ให้เพิ่มแถวในตาราง Gotchas ของ AGENTS.md (section 9)
- commit เป็น logical group (1 commit ต่อ 1-3 items ที่เกี่ยวกัน) แล้ว push ขึ้น origin main ได้เลย
- เช็คก่อนจบทุก commit:
  .venv\Scripts\python.exe -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['rl_app.py','rl_train.py','backtest_live.py','rl_walkforward.py','rl_finetune.py','rl_analyze.py','export_to_onnx.py','regime_compare.py','trading_env.py']]; print('ast ok')"
- สรุปตอนจบ: ไฟล์ที่แก้, item ที่ปิดได้/ปิดไม่ได้เพราะอะไร, มีอะไรที่ user ต้องทำเอง
  (เช่น recompile ใน MetaEditor, re-export package)
```

## Batch 1 — HIGH: ข้อมูลพัง / ผลลัพธ์ผิด (items 1–12)

```text
อยู่ในโปรเจค C:\Users\omesb\Documents\claude code\pycaret_trainer

ช่วยแก้บั๊กระดับ HIGH ทั้ง 12 ข้อจาก docs/bug_audit_2026-07-12.md (items 1-12)
อ่านไฟล์นั้นก่อน — ทุกข้อมี Trigger และ Fix sketch แล้ว สรุปหัวข้อ:

กลุ่ม A — ข้อมูลถูกทับเงียบ ๆ (แก้ pattern "ช่อง auto-fill ค้าง" ให้เหมือนกันทั้งสองจุด):
- item 3: Data Tools collector import — ช่อง Output basename ค้างจาก source แรก
  เปลี่ยน source แล้วชื่อไม่ตาม → import ทับ dataset คนละ symbol (CSV+params+M1)
  แก้: ใส่ command callback ให้ tool_collector_src + จำค่า auto-fill ล่าสุดไว้เทียบ
  (ทับได้เฉพาะเมื่อค่าปัจจุบัน == ค่า auto ล่าสุด) + askyesno เตือนก่อนเขียนทับ dataset ชื่อไม่ตรง source
- item 4: Export card — Deploy name ค้างจาก model แรก → export model B ทับ package ของ A
  แก้แบบเดียวกับ item 3 (_on_export_model_change จำค่า auto-fill ล่าสุด)
- item 5: Split Train/Test ไม่ copy .params.json ไปให้ไฟล์ _train/_test
- item 11: Fine-tune ตั้ง output name ซ้ำ base model → ทับ meta+zip ของ base
  แก้: block ตั้งแต่ _run_finetune ถ้า name == base (messagebox แนะนำให้เติม _v2)

กลุ่ม B — ผลเทส/เทรนไม่ตรงความจริง:
- item 2: rl_analyze.py สร้าง TradingEnv แบบ basic_4 + max_hold default เสมอ
  แก้: อ่าน action_profile/action_profile_config + max_hold จาก train meta
  (ดู pattern จาก rl_finetune._resolve_base_contract) ถ้า meta ไม่มีให้ auto-detect
  จากจำนวน action ของ model (profile_for_action_count) แบบเดียวกับ export_to_onnx.py
- item 6: ปิด checkbox Developer Mode แล้ว formula ที่โหลดจาก reward JSON ยังถูกส่งเข้าเทรน
  แก้: _get_train_reward_formula ต้อง return "" เมื่อ checkbox ปิด ไม่ว่า formula มาจากไหน
  และตัด formula ที่ติดมากับ reward_profile_json path ด้วยเมื่อ dev mode ปิด
- item 7: หน้า Backtest ไม่ส่ง --max_hold → บังคับปิด 30 bars เสมอ
  แก้: อ่าน max_hold จาก train meta ของ model ที่เลือก (fallback 30) แล้วส่ง --max_hold
  หมายเหตุ: backtest_live.py มี --max_hold อยู่แล้ว เช็คชื่อ flag ให้ตรง
- item 8: เปลี่ยน Test Dataset แล้วไฟล์ M1 เดิมค้างใน dropdown → replay ด้วย M1 ผิด symbol
  แก้: _auto_select_bt_m1 ต้อง reset เป็น "(none)" ก่อน แล้วค่อย auto-select ตัวที่แมตช์
  dataset ใหม่ (ห้ามทิ้งค่าเก่าไว้เมื่อหา companion ไม่เจอ)
- item 9: verdict "MOSTLY ROBUST" ของ Walk-Forward ถูกโชว์เป็น ✅ ROBUST เขียวเต็ม
  แก้ classifier ใน _handle_log_line (rl_app.py ~9107): แยก 4 กรณีตาม string จริงจาก
  rl_walkforward.py: "✅ ROBUST", "🟡 MOSTLY ROBUST", "⚠️ UNSTABLE", "❌ NOT ROBUST"
  ให้สี/ไอคอน/ข้อความตรงกัน
- item 12: rl_walkforward crash IndexError เมื่อ train slice สั้นกว่า ep_len+window
  แก้: clamp max_steps ต่อ window = min(args.ep_len, len(train_df)-window-3) + เตือนใน log

กลุ่ม C — อื่น ๆ:
- item 10: Output Name หน้า Walk-Forward ไม่ sanitize อักขระต้องห้ามของ Windows
  แก้: validate ก่อน start (ตัว regex [<>:"/\\|?*] + ห้ามว่าง) — ทำที่หน้า Fine-tune,
  Train name และ Export deploy name ด้วยถ้ายังไม่มี (ระวังซ้ำกับ item 15 ใน Batch 3
  ที่เป็นฝั่ง MQL5 identifier — อันนั้นยังไม่ต้องทำ)
- item 1: mt5_files\MQL5\Experts\ML_RL_Trader_template.mq5 CalcSLTP index ATR ผิด
  เมื่อ ATR_PSTEP != 1 (เช่น dataset extras ใช้ PMIN 6 / STEP 4)
  แก้: คำนวณ index = (target_period - ATR_PMIN) / ATR_PSTEP พร้อม clamp 0..count-1
  และเลือก period ใกล้ 14 ที่สุดที่มีจริง — บอกใน summary ว่า packages ที่ export
  ไปแล้ว (rl_uj_extra, rl_uj_h4_extras_2 ฯลฯ) ต้อง copy template ใหม่ + recompile ใน MetaEditor

เทสขั้นต่ำที่ต้องรันให้เห็นผล:
- headless GUI: เปิดหน้า tools/backtest/walkfwd/finetune แล้ว simulate เคส trigger
  ของ items 3,4,7,8,9 (mock _start_runner แล้วตรวจ cmd ที่สร้าง)
- rl_analyze: รันกับ model manage_6 ที่มีอยู่ (UJ_BE_Test) ต้องไม่ error และ log บอก profile ถูก
- rl_walkforward item 12: รันด้วย dataset เล็ก + windows เยอะ ให้เห็นว่า clamp ทำงานแทน crash

[แนบ "กติกาการทำงาน" จากหัวไฟล์นี้ต่อท้ายด้วย]
```

## Batch 2 — โครงสร้าง ProcessRunner + threading (items 18, 19, 36, 37, 38, 44, 46 และเกี่ยวเนื่อง 21)

```text
อยู่ในโปรเจค C:\Users\omesb\Documents\claude code\pycaret_trainer

ช่วยแก้กลุ่มบั๊ก race/threading ใน rl_app.py จาก docs/bug_audit_2026-07-12.md
(items 18, 19, 21, 36, 37, 38, 44, 46) — ทั้งกลุ่มมีรากเดียวกัน แก้ที่โครงสร้างครั้งเดียว:

1. ProcessRunner (rl_app.py ~438-475) — items 19, 36, 44:
   ตอนนี้ self.proc ถูก set ในเธรด worker หลัง Popen เสร็จ ทำให้ช่วง spawn (~50-500ms)
   is_running() ยัง False → กดปุ่ม Run ซ้อนได้ทุกหน้า / Stop ทันทีหลัง Start เป็น no-op
   แก้: เพิ่ม flag self._starting ที่ set บนเธรดหลักก่อน spawn thread และเคลียร์เมื่อ
   proc พร้อมหรือ spawn ล้มเหลว; is_running() = self._starting or (proc is not None และยังไม่จบ)
   start() ต้อง return False ถ้า _starting/proc ยังค้าง; stop() ระหว่าง _starting ให้ตั้ง
   flag ยกเลิกแล้ว terminate ทันทีที่ proc โผล่
2. _stop_pipeline (rl_app.py ~2171) — item 46:
   snapshot ตัวแปรก่อนใช้: proc = self.pipeline_proc; if proc and proc.poll() is None: ...
3. WM_DELETE_WINDOW — item 37:
   ใส่ protocol handler: ถ้ามีงานรัน (runner หรือ pipeline) ถาม askyesno ก่อน แล้ว
   terminate ลูกทั้งหมดก่อน destroy (Windows: ใช้ proc.terminate() แล้ว wait timeout สั้น)
4. Worker threads แตะ widget ตรง ๆ — items 18, 38:
   ไล่ worker เธรดทั้งหมด (_import_mt5_csv_worker, _split_csv_worker, _relabel_csv_worker,
   _export_onnx_worker และตัวอื่นที่เจอ) ให้ marshal การแตะ widget ผ่าน self.after(0, ...)
   ทำ helper กลาง เช่น self._log_safe(widget, msg, tag) ที่ข้างในเรียก after
   ค่า .get() จาก entry ให้อ่านบนเธรดหลักก่อน spawn worker แล้วส่งเป็น argument
5. item 21 (เกี่ยวเนื่อง): _start_training อ่าน Steps เป็น int หลัง disable ปุ่มแล้ว
   ถ้าพิมพ์ "300,000"/"2e5" จะ crash และปุ่มค้าง disabled ถาวร
   แก้: parse + validate ทุกช่องตัวเลข "ก่อน" แตะสถานะปุ่มใด ๆ (ทำ helper
   _parse_int_field(entry, default, label) ที่โชว์ messagebox เองแล้ว return None)
   แล้วใช้กับช่องตัวเลขของหน้า Train/Backtest/WF/Fine-tune ให้ครบ

เทส: headless — mock subprocess ให้ spawn ช้า (monkeypatch Popen ด้วย sleep สั้น)
แล้วยิง start() สองครั้งติดกัน ต้องได้ False ครั้งที่สอง; ทดสอบ _parse_int_field กับ
"300,000", "2e5", "", "50000"; เปิดแอปแล้ว destroy ระหว่าง mock งานรันต้องไม่ throw

[แนบ "กติกาการทำงาน" จากหัวไฟล์นี้ต่อท้ายด้วย]
```

## Batch 3 — MEDIUM รายหน้า (items 13–17, 20, 22–35, 39, 40)

```text
อยู่ในโปรเจค C:\Users\omesb\Documents\claude code\pycaret_trainer

ช่วยแก้บั๊ก MEDIUM ที่เหลือจาก docs/bug_audit_2026-07-12.md ตาม fix sketch ของแต่ละข้อ
เรียงตามหน้า (ข้าม items 18,19,21,36,37,38 ที่ปิดไปแล้วใน Batch 2):

หน้า Train — items 22, 23, 24, 40:
- 22: Reset to profile ระหว่างมี reward JSON โหลดอยู่ → ค่าที่เทรนจริงยังเป็นของ JSON
- 23: พิมพ์ผิด 1 ช่อง reward → slider ทั้งหมดถูก reset เงียบ ๆ (ให้ error เฉพาะช่อง ไม่ล้างของอื่น)
- 24: reward_mode=mtm ไม่ใช้ slider/formula เลย → disable ช่องพวกนั้น + hint เมื่อเลือก mtm
- 40: DQN/A2C ใน rl_train.py hardcode hyperparams → ใช้ค่าจาก args ที่ GUI ส่ง (เท่าที่ algo รองรับ)
  และ log บอกชัดว่า param ไหนไม่ apply กับ algo นั้น

หน้า Backtest — items 13, 14, 25, 26, 27:
- 13: sharpe annualize ผิด timeframe (sqrt(252*24) กับ H4) → ตรวจจับ bar interval จาก timestamp
  แล้วใช้ตัวคูณให้ตรง (H4 = 6 bars/วัน) + label บอกว่า annualized จาก timeframe ไหน
- 14: backtest fail ทับ meta สมบูรณ์เดิมด้วย {status: failed} → เขียน fail ลง key แยก
  หรือเก็บ meta เดิมไว้ (เช่น เขียน <model>_live_bt.failed.json แทน)
- 25: PF การ์ดค้างเมื่อค่าเป็น 'inf' → format เป็น '∞' หรือ cap แสดง '>99' และแก้ heuristic _update_stat
- 26: report/equity render ตาม model ที่เลือกอยู่ตอนงานจบ ไม่ใช่ตัวที่ถูกเทส →
  จำ model name ตอน start ไว้กับ run แล้วใช้ตัวนั้นตอน _handle_done
- 27: backtest 0 เทรด → เคลียร์การ์ด+รูปเก่า แสดงข้อความ "0 trades" ชัดเจน

หน้า Regime/Quality — items 16, 17, 20:
- 16: chart HTML fetch JSON ไม่ได้บน file:// → embed JSON ลงใน HTML เลย (inline <script>)
  ทำทั้ง regime_single และ regime_compare
- 17: dict key ชื่อ event ซ้ำทับกัน → เก็บเป็น list หรือ key = (name, date)
- 20: Use-as-Train-Cutoff ใช้ breakpoints จาก detection ก่อนหน้ากับ CSV อื่น →
  จำ csv path ที่ detection รันไว้ แล้ว block/เตือนถ้า CSV ปัจจุบันไม่ตรง + กันผล dataset ว่าง

หน้า Models — items 29, 30, 31, 32, 33:
- 29: Params column เช็ค marker string ที่มีทุกไฟล์ → เช็คว่ามี params จริง (ไม่ใช่ no-op stub)
- 30: refresh signature ไม่ครอบไฟล์ที่ row ใช้ → เพิ่ม best/**.zip + meta files เข้า signature
- 31: 'M1 ✓' ทั้งที่ resolve ได้ส่วนน้อย → แสดงสัดส่วน เช่น 'M1 62%' และ ✓ เฉพาะเมื่อ
  ambiguous ที่เหลือ < 10%
- 32: Return%/MaxDD hardcode $10k → อ่าน balance จาก backtest meta
- 33: แถว 'env backtest' หยิบ equity/Realism ของ live backtest มาปน → แยก source ให้ชัด
  หรือไม่แสดงรูป/Realism สำหรับแถว env

หน้า Pipeline — items 34, 35:
- 34: pipe_bt_csv.set() ไม่ refresh M1 hint → เรียก _update_pipe_m1_hint() หลัง set เสมอ
- 35: pipeline ไม่ส่ง --stop_slippage → ส่งค่า default เดียวกับหน้า Backtest (0.01% = 0.0001)

Widget — item 39: ScrollableOptionMenu popup ไม่ปิดเมื่อย้าย/ย่อหน้าต่าง/สลับแอป →
bind <Configure> ของ toplevel หลัก + <FocusOut>/<Unmap> ให้เรียก _close_popup

Export — item 15: sanitize deploy name ฝั่ง export_to_onnx.py (MQL5 identifier:
[A-Za-z_][A-Za-z0-9_]*) — แปลงอักขระอื่นเป็น _ พร้อมเตือน และกันชนกับ validation
ฝั่ง GUI ที่ทำไปแล้วใน Batch 1 item 10

เทสขั้นต่ำ: headless GUI ทุกหน้าที่แตะ + สร้าง meta/JSON ปลอมใน %TEMP% เพื่อเทสเคส
27/29/31/32; รัน backtest_live กับ model จริง 1 ครั้ง (uj_h4 sample) เช็ค sharpe ใหม่ + meta fail path

[แนบ "กติกาการทำงาน" จากหัวไฟล์นี้ต่อท้ายด้วย]
```

## Batch 4 — LOW (items 41–43, 45, 47)

```text
อยู่ในโปรเจค C:\Users\omesb\Documents\claude code\pycaret_trainer

เก็บงาน LOW ที่เหลือจาก docs/bug_audit_2026-07-12.md:
- 41: std epsilon ถูกบวกซ้ำสองชั้น (rl_train เขียน std+1e-8 ลง norm.csv แล้ว EA/backtest
  บวกอีก 1e-8) → เลือกชั้นเดียว: ให้ rl_train เขียน std ดิบ แล้วผู้บริโภคเป็นคนบวก
  (ระวัง: model เก่า norm เก่ายังเป็น std+1e-8 — ต้อง backward compatible, ห้ามทำให้
  ผล inference ของ model เก่าเปลี่ยน; ถ้าเสี่ยงให้แค่ document ไว้ใน AGENTS.md แทน)
- 42: PELT breakpoint off-by-one เทียบ HMM/K-Means → รายงานวันแรกของ regime ใหม่ให้เหมือนกัน
- 43: path ค้างฝั่ง CLI ของ regime_compare/gemini_labeler (default CSV หาย, known_events.json
  เขียน CWD แต่อ่าน script dir) → ใช้ script dir ทั้งคู่
- 45: ช่อง Mix Ratio label "(old %)" แต่รับ 0-1 → รับทั้งสองแบบ (ถ้า >1 ให้หาร 100)
  แบบเดียวกับ _parse_train_pct แล้วแก้ label เป็น "(0-1 หรือ %)"
- 47: pythonw.exe crash เพราะ sys.stdout เป็น None → guard: if sys.stdout is not None ก่อน wrap

[แนบ "กติกาการทำงาน" จากหัวไฟล์นี้ต่อท้ายด้วย]
```

## หลังทุก batch เสร็จ

```text
เช็คสุดท้ายหลังปิดครบทุก batch:
1. ทุก checkbox ใน docs/bug_audit_2026-07-12.md ต้องเป็น [x] หรือมีหมายเหตุว่าทำไมข้าม
2. อัปเดต AGENTS.md: เพิ่ม gotchas ที่เจอระหว่างแก้ + อัปเดต section ที่พฤติกรรมเปลี่ยน
   (เช่น Backtest ส่ง --max_hold จาก meta แล้ว, WF verdict 4 ระดับ)
3. รัน smoke ทั้งแอป: .venv\Scripts\python.exe -c "import rl_app; app = rl_app.RLTradingStudio();
   [app.show_page(k) for k in list(app.PAGE_TITLES)]; app.destroy(); print('all pages ok')"
4. แจ้งรายการที่ user ต้องทำเอง: recompile EA template ใน MetaEditor,
   re-export packages ที่ได้รับผลจาก item 1/15, model ไหนควร backtest ซ้ำ
```
