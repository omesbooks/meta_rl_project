# Live visual — รูปแบบการแสดงผลและต้นทุนทรัพยากร (2026-10-11)

> คำถาม: ภาพ 3D ตอนนี้กินทรัพยากรที่สุดหรือไม่ และมีรูปแบบอื่นอะไรบ้าง
> วัดจริงในหน้าต่าง pywebview (Edge WebView2) ขนาด 1600×920 กับ snapshot ของโมเดล `rl_test` (200k steps)
> สถานะ: **ทำครบทั้ง 3 โหมด + Auto idle** (ดู "ผลหลังทำ" และ "Dashboard" ท้ายไฟล์)
> CPU = % ของ 1 core (เครื่องนี้มี 20 logical cores → 100% ≈ 5% ของทั้งเครื่อง) · วัด 10 วินาทีหลังหน้าโหลดเสร็จ

## ผลวัด

| รูปแบบ | fps ที่วาดได้จริง | CPU | หน่วยความจำ |
|---|---:|---:|---:|
| หน้าเปล่า (WebView2 อย่างเดียว) | – | 0.5% | 416 MB |
| 3D cinematic (ค่าเริ่มต้นตอนนี้) · cap 30 | 30 | 61–65% | ~700 MB |
| 3D cinematic · cap 60 | **37** (ไปไม่ถึง 60) | 138–142% | ~735 MB |
| 3D ไม่มี bloom · cap 30 | 30 | 39% | ~650 MB |
| 3D lite (ไม่มี bloom, synapse 4, ไม่มีจุดแสง, ความละเอียด 1x) · cap 30 | 30 | 35–37% | ~650 MB |
| 3D lite · cap 15 | 15 | 21% | ~630 MB |
| 3D cinematic หยุดเล่น (กล้องส่ายอย่างเดียว) · cap 30 | 30 | 56% | ~680 MB |
| 2D canvas (ต้นแบบแรก) · ไม่จำกัด | 59 | **187%** | ~575 MB |

## สิ่งที่ตัวเลขบอก

1. **3D ไม่ใช่แบบที่หนักที่สุด** — WebGL ส่งงานวาดให้ GPU · ต้นแบบ 2D canvas ที่วาดเส้นหลายพันเส้น + แสงเรือง กิน CPU มากกว่า 3x เพราะวาดบน CPU
2. **ต้นทุนมาจาก "การวาดทุกเฟรม" ไม่ใช่ข้อมูล** — หยุดเล่นแต่ยังวาดอยู่ยังกิน 56% · ข้อมูลใหม่มาแค่ครั้งละ 1–2 นาที (ทุก eval)
3. **bloom คิดเป็นราว 40% ของต้นทุน 3D** (65% → 39%)
4. **fps แปรผันตรงกับ CPU** (lite 30 → 15 fps: 37% → 21%)
5. หน่วยความจำ ~650–740 MB ส่วนใหญ่คือ WebView2 เอง (หน้าเปล่า 416 MB) — ลดได้ไม่มาก
6. 60 fps ไม่คุ้ม: กิน CPU 2.2x แต่ได้จริงแค่ 37 fps

## รูปแบบที่เสนอ (ปุ่มเลือกในหน้าต่าง + ค่าเริ่มต้นจากหน้า Train)

| โหมด | แสดงอะไร | วิธีวาด | CPU (ประมาณจากผลวัด) | ใช้เมื่อ |
|---|---|---|---:|---|
| 🎬 Cinematic | แบบปัจจุบันครบ: bloom, จุดแสงวิ่ง, synapse 8–16 | WebGL 30 fps | ~60% | อัดวิดีโอ / โชว์ |
| ⚡ Lite (**ค่าเริ่มต้นระหว่างเทรน**) | ฉาก 3D เดิม ไม่มี bloom, synapse 4, ไม่มีจุดแสง, ความละเอียด 1x | WebGL 15–20 fps | ~20–25% | ดูระหว่างเทรนทั่วไป |
| 📊 Dashboard | กราฟ reward ทุกรอบ eval, เส้น equity ซ้อนทุกรอบ, สัดส่วน action, ตารางเทรด, heatmap activation รอบล่าสุด | SVG/HTML วาดใหม่เฉพาะตอน snapshot มา (ไม่มี animation loop) | ~0–2% | เทรนนาน / โน้ตบุ๊ก / เปิดทิ้งไว้ทั้งวัน |
| 💤 Auto idle (ใช้กับทุกโหมด) | หยุดวาดเมื่อย่อหน้าต่าง หรือเล่น snapshot จบรอบแล้ว · ตื่นเมื่อ snapshot ใหม่มาหรือขยับเมาส์ | – | ~0.5% ตอนพัก | ทุกกรณี |

ทางเลือกเสริมใน Lite/Cinematic: **เล่นรอบเดียวต่อ eval** — เล่น 300 แท่ง (~2 นาทีที่ speed 6 ใกล้เคียงระยะห่างระหว่าง eval) แล้วพักจน snapshot ใหม่มา

## งานที่ต้องทำถ้าจะทำครบ

- viewer: ตัวเลือกโหมดใน HUD + `mode=` ใน URL (knob `bloom/k/pulses/dpr/play/fps` มีแล้วตั้งแต่การวัดนี้) · auto idle (visibilitychange + จบรอบ) · โหมด Dashboard เป็น layout แยกในไฟล์เดียวกัน
- GUI: ตัวเลือกโหมดข้าง checkbox "Live 3D visual" (ค่าเริ่มต้น Lite) ส่งผ่าน `live_viewer.py --query`
- บทเรียนจากการวัด: animation loop ต้อง `requestAnimationFrame` เป็นบรรทัดแรก — ต้นแบบ 2D ที่เรียกไว้ท้ายฟังก์ชันหยุดวาดถาวรใน WebView2 เพราะเฟรมแรกพังตอนหน้าต่างยังไม่มีขนาด (viewer 3D ตัวจริงทำถูกแล้ว)

## ผลหลังทำ Lite + Auto idle (2026-10-11, หน้าต่าง pywebview จริง, snapshot `rl_test`)

| สถานะ | CPU | fps ที่วาดจริง |
|---|---:|---:|
| Lite ขณะเล่น (ค่าเริ่มต้นของโหมด live) | 21–26% | 20 |
| เล่น snapshot จบ → idle เอง | 1.2% | 0 |
| ขยับเมาส์ → เล่นใหม่ตั้งแต่ต้น | 21% | 20 |
| ย่อหน้าต่าง → idle | 2.4% | 0 |
| คืนหน้าต่าง → เล่นต่อ | 23% | 20 |
| Cinematic | 63% | 30 |
| idle แล้วมี eval ใหม่เข้ามา | ตื่นเอง โหลด snapshot ใหม่ภายใน ~3 วินาที | 20 |

การทำงาน
- `mode=lite|cinematic` (URL / เมนู mode ในหน้าต่าง / เมนูข้าง checkbox ในหน้า Train → `live_viewer.py --mode`) · ค่าเริ่มต้น live = Lite, หน้า static = Cinematic
- live: เล่นแต่ละ snapshot รอบเดียวแล้ว idle (`loop=1` เพื่อวนต่อ) · หน้า static วนตามเดิม
- ทุกหน้า: idle เมื่อ pause แล้วไม่มี input 8 วินาที, เมื่อ `visibilitychange` เป็น hidden, และเมื่อหน้าต่าง pywebview ถูกย่อ
  (WebView2 วาดต่อแม้ย่อหน้าต่าง — `live_viewer.py` ส่ง event minimized/restored เข้าไปให้หน้าเว็บ)
- ตอน idle loop หยุด `requestAnimationFrame` จริง (ภาพสุดท้ายค้างไว้บนจอ) · ตื่นเมื่อ snapshot ใหม่มา ขยับเมาส์ คลิก scroll กดปุ่ม หรือเปลี่ยน control

## Dashboard (2026-10-11)

ไฟล์ `tools/viz/dashboard.js` (ES module, ไม่มี animation loop) · เลือกด้วย `mode=dashboard` / เมนู mode ในหน้าต่าง /
"Dashboard (lowest CPU)" ในหน้า Train

แผง: KPI 8 ช่อง (step, eval reward, best, trades, win rate, equity, max DD, avg trade) · learning curve (reward ต่อ eval + best so far) ·
action mix ต่อ eval · equity ของทุก eval ซ้อนกัน (ทอง = ล่าสุด) · ตารางเทรดล่าสุด · replay แท่งเทียน + จุดเข้า/ออก + equity ของ eval ที่เลือก ·
heatmap นิวรอน × แท่ง ของทุก layer (Actor/Critic, สูงสุด 64 นิวรอนต่อ layer) + แถบความน่าจะเป็นของ action + แถบ V(s)

ข้อมูล: `index.json` เก็บสรุปต่อ eval เพิ่ม (`snapshot.summary()`: equity ย่อ 120 จุด, จำนวน action, trades/wins/win rate, max DD)
→ Dashboard ไม่ต้องโหลดทุก snapshot · run เก่าที่ไม่มีสรุป Dashboard โหลด snapshot แล้วคำนวณเองครั้งเดียว (cache)

| สถานะ (หน้าต่าง WebView2 จริง, run `rl_test` 7 eval) | CPU |
|---|---:|
| Dashboard นิ่ง | 1–2% |
| เลือก eval ใน timeline (โหลด snapshot + วาดทั้งหน้า) | ~21% เฉลี่ย 3 วินาที (ครั้งเดียว) |
| resize แล้ววาดใหม่ | ~5% เฉลี่ย 3 วินาที |
| สลับไป Lite / กลับ Dashboard | Lite วาด 20 fps ตามปกติ · กลับ Dashboard loop หยุดทันที |

หมายเหตุการวัด: ตัวเลข Lite/Cinematic ขึ้นกับสภาพเครื่อง — วัดซ้ำตอนบ่าย Lite ได้ 37–44% และ Cinematic ไปได้แค่ 24 fps
(เช้า 21–26% / 30 fps) · A/B ไฟล์ viewer รุ่นก่อนกับรุ่นนี้ในสภาพเดียวกัน: 44–45% vs 39–41% → ไม่ได้เกิดจากโค้ด
