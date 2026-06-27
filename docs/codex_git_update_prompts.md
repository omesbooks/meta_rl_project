# Codex Git Update Prompt

ไฟล์นี้เป็น prompt สำหรับให้ผู้ใช้สั่ง Codex บนเครื่องของตัวเอง
เพื่อดึงอัพเดทล่าสุดจาก GitHub และติดตั้ง dependencies ตาม repo อย่างปลอดภัย

ใช้ได้กับเครื่องที่ clone repo ไว้แล้ว และเปิด Codex อยู่ที่ root ของโปรเจคนี้
เช่น `C:\Users\omesb\Documents\claude code\pycaret_trainer`

## Prompt: ดึงอัพเดทจาก GitHub อย่างปลอดภัย

คัดลอก prompt นี้ไปวางใน Codex:

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
   ให้หยุดแล้วสรุปไฟล์ที่เปลี่ยนก่อน เพื่อให้เลือกว่าจะเก็บงาน local ไว้อย่างไร

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

## PowerShell แบบรันเองสำหรับโปรเจคนี้

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

## ข้อควรระวัง

- อย่าใช้ `git reset --hard` ถ้าไม่ได้ตั้งใจลบ local changes จริง ๆ
- อย่าใช้ `git add .` หรือ commit/push งานจากเครื่องผู้ใช้ ถ้าไม่ได้รับคำสั่งชัดเจนจากผู้ดูแล repo
- ถ้าเจอ conflict ให้หยุดและอ่าน diff ก่อนแก้
- สำหรับโปรเจคนี้ ควรใช้ `.venv\Scripts\python.exe` เสมอ ไม่ใช้ `python` เปล่า
- ไฟล์ data/model/export/log ส่วนใหญ่เป็น runtime artifact และไม่ควร commit
