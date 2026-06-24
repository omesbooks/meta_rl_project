"""
Quarterly Update Orchestrator
=============================
รันทุก 3 เดือนด้วยคำสั่งเดียว — ทำทุกขั้นตอน + รายงานผลครบ

Steps:
  1. Pull latest data from MT5
  2. Calc features + relabel
  3. Fine-tune existing model
  4. Walk-forward validate
  5. Decision gate: deploy หรือไม่
  6. (optional) auto-deploy
  7. Send notification

Usage:
    python quarterly_update.py
    python quarterly_update.py --auto-deploy   # auto-promote ถ้าผ่าน WF
    python quarterly_update.py --dry-run       # simulate ไม่จริง
"""
import sys
import io
import argparse
import subprocess
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def log(msg, level="info"):
    icons = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "err": "❌", "step": "🔄"}
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icons.get(level, '')} {msg}")


def run_step(name, cmd, dry_run=False):
    """รัน sub-command พร้อม log + measure time"""
    log(f"STEP: {name}", "step")
    log(f"  $ {' '.join(cmd)}", "info")
    if dry_run:
        log(f"  [DRY RUN] skipped", "warn")
        return True

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        log(f"  ok ({elapsed:.0f}s)", "ok")
        return True
    else:
        log(f"  FAILED ({elapsed:.0f}s)", "err")
        log(f"  stderr: {result.stderr[:500]}", "err")
        return False


def parse_wf_results(csv_path: Path):
    """อ่านผล walk-forward → คืน {pass_count, total, mean_pf}"""
    if not csv_path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(csv_path)
    pass_count = (df["profit_factor"] > 1.0).sum()
    return {
        "pass_count": int(pass_count),
        "total": len(df),
        "mean_pf": float(df["profit_factor"].mean()),
        "min_pf": float(df["profit_factor"].min()),
    }


def notify(title, msg, success=True):
    """ส่ง notification — สามารถใส่ Telegram/Discord webhook ที่นี่"""
    icon = "✅" if success else "❌"
    print(f"\n{'='*60}")
    print(f"  {icon} {title}")
    print(f"{'='*60}")
    print(msg)
    print(f"{'='*60}\n")
    # TODO: ใส่ Telegram bot / Discord webhook
    # requests.post(WEBHOOK_URL, json={"text": f"{title}\n{msg}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-model", default="rl_v3",
                    help="ชื่อ model ที่ใช้อยู่ในปัจจุบัน")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--mix-ratio", type=float, default=0.3)
    ap.add_argument("--auto-deploy", action="store_true",
                    help="ติดตั้ง model ใหม่อัตโนมัติถ้าผ่าน WF gate")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate steps ไม่รันจริง")
    ap.add_argument("--wf-pass-min", type=int, default=4,
                    help="WF windows ต้องผ่านขั้นต่ำ (default 4/5)")
    ap.add_argument("--wf-min-pf", type=float, default=1.0,
                    help="ทุก window ต้องมี PF >= ค่านี้")
    args = ap.parse_args()

    # === Setup ===
    today = datetime.now()
    quarter = (today.month - 1) // 3 + 1
    new_name = f"{args.current_model}_q{today.year}{quarter}"  # rl_v3_q20262

    log("=" * 60, "info")
    log(f"Quarterly Update — {today:%Y-%m-%d}", "info")
    log(f"  Current model: {args.current_model}", "info")
    log(f"  New model name: {new_name}", "info")
    log(f"  Symbol: {args.symbol}", "info")
    log(f"  Auto-deploy: {args.auto_deploy}", "info")
    log("=" * 60, "info")

    if args.dry_run:
        log("DRY RUN MODE — no actual changes", "warn")

    # === Verify base model exists ===
    base_zip = Path(f"{args.current_model}.zip")
    base_norm = Path(f"{args.current_model}_norm.csv")
    if not base_zip.exists() and not args.dry_run:
        log(f"Base model not found: {base_zip}", "err")
        notify("Quarterly Update FAILED",
               f"Base model {args.current_model}.zip ไม่พบ", success=False)
        return 1

    # === Step 1: Pull latest data ===
    # NOTE: ต้องเขียน pull_mt5_data.py แยก (ใช้ MetaTrader5 Python API)
    # ตัวอย่าง:
    #   end = today
    #   start = today - timedelta(days=90)  # last quarter
    end = today
    start = today - timedelta(days=90)
    csv_new = Path(f"data_{args.symbol}_{start:%Y%m%d}_{end:%Y%m%d}.csv")

    log(f"\n📥 Step 1: Pull data {start:%Y-%m-%d} → {end:%Y-%m-%d}", "step")
    if not args.dry_run:
        # placeholder — real version uses pull_mt5_data.py
        if not csv_new.exists():
            log(f"  data file not found: {csv_new}", "warn")
            log("  TODO: implement pull_mt5_data.py", "warn")
            log("  ใช้ existing data เพื่อ demo", "warn")
            csv_new = Path("training_data_v2.csv")
            if not csv_new.exists():
                log(f"  ก็ไม่มี — abort", "err")
                return 1

    # === Step 2: Relabel ===
    csv_relabeled = csv_new.with_stem(csv_new.stem + "_relabeled")
    log(f"\n🔧 Step 2: Relabel data", "step")
    # NOTE: relabel.py interactive — สำหรับ auto ต้องเรียกฟังก์ชันตรง
    # ตัวอย่างนี้สมมติว่า relabel แล้ว
    if not csv_relabeled.exists() and not args.dry_run:
        log(f"  ⚠️ ต้อง relabel manually: python relabel.py {csv_new}", "warn")
        log(f"  (หรือ refactor relabel.py ให้รับ args ตรง)", "warn")

    # === Step 3: Fine-tune ===
    log(f"\n🤖 Step 3: Fine-tune {args.current_model} → {new_name}", "step")
    cmd = [
        sys.executable, "rl_finetune.py", args.current_model,
        "--old_csv", "training_data_v2_relabeled.csv",  # original old data
        "--new_csv", str(csv_relabeled),
        "--steps", str(args.steps),
        "--mix_ratio", str(args.mix_ratio),
        "--name", new_name,
    ]
    if not run_step("Fine-tune PPO", cmd, dry_run=args.dry_run):
        notify("Quarterly Update FAILED",
               f"Fine-tune step failed", success=False)
        return 1

    # === Step 4: Walk-forward validate ===
    log(f"\n🔬 Step 4: Walk-forward validate", "step")
    wf_name = f"wf_{new_name}"
    cmd = [
        sys.executable, "rl_walkforward.py", str(csv_relabeled),
        "--windows", "5",
        "--steps", "30000",  # short WF for speed
        "--window", "10",
        "--name", wf_name,
    ]
    if not run_step("Walk-forward", cmd, dry_run=args.dry_run):
        log("WF failed — skipping deploy", "warn")
        return 1

    # === Step 5: Parse WF results + decision gate ===
    log(f"\n🎯 Step 5: Decision Gate", "step")
    wf_csv = Path(f"{wf_name}_results.csv")

    if args.dry_run:
        log("  [DRY RUN] simulated WF result: PASSED", "ok")
        decision = "DEPLOY"
    else:
        results = parse_wf_results(wf_csv)
        if not results:
            log(f"  ไม่พบไฟล์ผล WF: {wf_csv}", "err")
            return 1

        log(f"  WF: {results['pass_count']}/{results['total']} windows passed", "info")
        log(f"  Mean PF: {results['mean_pf']:.2f} | Min PF: {results['min_pf']:.2f}", "info")

        # Gate logic
        if (results['pass_count'] >= args.wf_pass_min and
            results['min_pf'] >= args.wf_min_pf):
            log(f"  ✅ PASS — meets gate criteria", "ok")
            decision = "DEPLOY"
        else:
            log(f"  ❌ FAIL — does not meet gate", "err")
            log(f"  Need: pass >= {args.wf_pass_min}, min PF >= {args.wf_min_pf}", "warn")
            decision = "HOLD"

    # === Step 6: Deploy (if approved) ===
    log(f"\n🚀 Step 6: Deployment", "step")
    if decision == "DEPLOY" and args.auto_deploy:
        # Backup current model
        archive_name = f"{args.current_model}_archive_{today:%Y%m%d}"
        log(f"  Archiving {args.current_model} → {archive_name}", "info")
        if not args.dry_run:
            shutil.copy(f"{args.current_model}.zip", f"{archive_name}.zip")
            shutil.copy(f"{args.current_model}_norm.csv", f"{archive_name}_norm.csv")

        # Promote new model to "current" name
        log(f"  Promoting {new_name} → {args.current_model}", "info")
        if not args.dry_run:
            shutil.copy(f"{new_name}.zip", f"{args.current_model}.zip")
            shutil.copy(f"{new_name}_norm.csv", f"{args.current_model}_norm.csv")
        log(f"  ✅ Deployed!", "ok")

    elif decision == "DEPLOY":
        log(f"  Auto-deploy disabled — model พร้อมแล้ว แต่ต้องสั่ง deploy เอง:", "info")
        log(f"    cp {new_name}.zip {args.current_model}.zip", "info")
    else:
        log(f"  Skipped — gate ไม่ผ่าน", "warn")

    # === Step 7: Notify ===
    summary = f"""
Date         : {today:%Y-%m-%d}
Base model   : {args.current_model}
New model    : {new_name}
Decision     : {decision}
Auto-deploy  : {args.auto_deploy}

Walk-forward results in: {wf_name}_results.csv
Chart: {wf_name}_chart.png
"""
    notify(
        f"Quarterly Update — {decision}",
        summary,
        success=(decision == "DEPLOY")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
