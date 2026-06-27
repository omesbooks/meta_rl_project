# Codex Git Update Prompts

ไฟล์นี้รวม prompt สำหรับสั่ง Codex ให้ช่วยดูแล repo บนเครื่องของผู้ใช้เองอย่างปลอดภัย
โดยเน้น 2 งานหลัก:

- ดึงอัพเดทล่าสุดจาก GitHub เข้ามาในโปรเจค
- ตรวจงานที่แก้ แล้ว commit/push กลับขึ้น GitHub

ใช้ได้กับเครื่องที่ clone repo ไว้แล้ว และเปิด Codex อยู่ที่ root ของโปรเจคนี้
เช่น `C:\Users\omesb\Documents\claude code\pycaret_trainer`

## 1. Prompt: ดึงอัพเดทจาก GitHub อย่างปลอดภัย

ใช้ prompt นี้เมื่อต้องการให้อีกเครื่องอัพเดทโปรเจคให้เป็นเวอร์ชันล่าสุดจาก `origin/main`
และติดตั้ง dependencies ที่เปลี่ยนตาม repo

```text
ช่วยอัพเดทโปรเจคนี้จาก GitHub และติดตั้ง dependencies ล่าสุดแบบปลอดภัย

ขั้นตอน:
1. เช็คว่าอยู่ใน repo ถูกต้อง:
   git remote -v
   git branch --show-current
   git log -1 --oneline

2. เช็ค local changes:
   git status --short

3. ถ้า worktree สะอาด ให้ update จาก remote:
   git fetch --all --prune
   git pull --ff-only origin main

4. ถ้ามี local changes ห้าม reset / checkout / ทับไฟล์เอง
   ให้หยุดแล้วสรุปไฟล์ที่เปลี่ยนก่อน เพื่อให้เลือกว่าจะ commit / stash / ยกเลิกการ pull

5. หลัง pull แล้ว ให้ติดตั้ง dependencies ตามไฟล์ของโปรเจค:
   - ถ้ามี requirements.txt ให้รัน .venv\Scripts\python.exe -m pip install -r requirements.txt
   - ถ้ามี package-lock.json ให้รัน npm install
   - ถ้ามี pyproject.toml ให้ดู README หรือไฟล์ config ว่าควรใช้ pip / poetry / uv

6. เช็ค dependency conflict:
   .venv\Scripts\python.exe -m pip check

7. รัน smoke test เบื้องต้นของโปรเจค เช่น:
   .venv\Scripts\python.exe -B -c "import rl_app; print('app import ok')"

8. สรุปผล:
   - ก่อนอัพเดทอยู่ commit ไหน
   - หลังอัพเดทอยู่ commit ไหน
   - dependencies ติดตั้ง/ตรวจผ่านไหม
   - มี error/warning อะไรเหลือไหม
   - git status หลังทำเสร็จเป็นยังไง
```

## 2. Prompt: ดึงอัพเดท แล้ว commit/push งานกลับ GitHub

ใช้ prompt นี้เมื่อต้องการให้ Codex ช่วยทั้งอัพเดท repo และถ้ามีงานที่ตั้งใจแก้ไว้
ให้ตรวจสอบ, commit, และ push กลับขึ้น GitHub ด้วย

```text
ช่วยอัพเดทโปรเจคนี้จาก GitHub และถ้ามีการแก้ไขที่ตั้งใจไว้ ให้ commit/push กลับขึ้น GitHub แบบปลอดภัย

1. เช็คว่าอยู่ใน repo ถูกต้อง:
   git remote -v
   git branch --show-current
   git log -1 --oneline

2. เช็ค local changes:
   git status --short

3. ถ้า worktree สะอาด ให้ดึงอัพเดท:
   git fetch --all --prune
   git pull --ff-only origin main

4. ถ้ามี local changes ก่อน pull:
   ห้าม reset / checkout / ทับไฟล์เอง
   ให้สรุปไฟล์ที่เปลี่ยนก่อน แล้วถามว่าจะ commit / stash / ยกเลิกการ pull

5. หลัง pull ให้ติดตั้ง dependencies ล่าสุดตามโปรเจค:
   - ถ้ามี requirements.txt: .venv\Scripts\python.exe -m pip install -r requirements.txt
   - ถ้ามี package-lock.json: npm install
   - ถ้ามี pyproject.toml: ตรวจว่าควรใช้ pip / poetry / uv ตาม README หรือไฟล์ config

6. เช็ค dependency conflict และ smoke test:
   .venv\Scripts\python.exe -m pip check
   แล้วรัน test/import/build เบื้องต้นตามโปรเจค เช่น import main app

7. ถ้ามีการแก้ไขไฟล์หลังจากอัพเดทหรือลง dependencies ให้เช็คก่อน commit:
   git status --short
   git diff --stat
   git diff --check

8. ตรวจว่าไม่มีไฟล์ที่ไม่ควร commit เช่น:
   .venv, __pycache__, .env, api_keys, secrets, model .zip, .onnx, logs, generated output ที่ gitignore ควรจัดการ

9. Stage เฉพาะไฟล์ที่เกี่ยวกับงานนี้เท่านั้น
   ห้ามใช้ git add . ถ้ายังไม่ได้ตรวจรายการไฟล์ทั้งหมด

10. Commit ด้วยข้อความสั้น ๆ ที่อธิบายงานจริง

11. Push ขึ้น GitHub:
    git push origin main

12. สรุปผล:
    - ก่อนอัพเดทอยู่ commit ไหน
    - หลังอัพเดทอยู่ commit ไหน
    - dependencies ติดตั้ง/ตรวจผ่านไหม
    - มีไฟล์อะไรถูก commit/push
    - commit hash ที่ push
    - git status หลังทำเสร็จ
```

## 3. PowerShell แบบรันเองสำหรับโปรเจคนี้

ถ้าไม่ใช้ Codex และต้องการรันเองบน Windows:

```powershell
cd "C:\Users\omesb\Documents\claude code\pycaret_trainer"

git status --short
git fetch --all --prune
git pull --ff-only origin main

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B -c "import rl_app; print('app import ok')"

git status --short
git log -1 --oneline
```

## 4. ข้อควรระวัง

- อย่าใช้ `git reset --hard` ถ้าไม่ได้ตั้งใจลบ local changes จริง ๆ
- อย่าใช้ `git add .` ถ้ายังไม่ได้ตรวจว่ามีไฟล์ generated หรือ secrets ปนไหม
- ถ้าเจอ conflict ให้หยุดและอ่าน diff ก่อนแก้
- สำหรับโปรเจคนี้ ควรใช้ `.venv\Scripts\python.exe` เสมอ ไม่ใช้ `python` เปล่า
- ไฟล์ data/model/export/log ส่วนใหญ่เป็น runtime artifact และไม่ควร commit ยกเว้นตั้งใจทำเป็น example
