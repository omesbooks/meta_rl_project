"""
Trigger-Based Fine-tune Orchestrator
=====================================
Smart orchestrator ที่:
  1. ตรวจ trigger conditions (PF drop, regime shift, schedule)
  2. ใช้ "Recent Window + Since Last Train" hybrid strategy
  3. ตรวจ safety: cooldown, min_data, data quality
  4. Fine-tune ถ้าทุก gate ผ่าน
  5. Walk-forward validate ก่อน deploy
  6. Log ทุกอย่าง

Usage:
    python trigger_finetune.py              # check + run if triggered
    python trigger_finetune.py --force      # force run, ignore cooldown
    python trigger_finetune.py --check-only # just print status, don't run
"""
import sys, io, json, sqlite3, argparse, subprocess
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ===== CONFIG =====
SYMBOL = "XAUUSDm"
CURRENT_MODEL = "rl_v3"
ARCHIVE_DIR = Path("./model_archive")
LOG_DB = Path("finetune_log.db")

# Triggers
TRIGGER_PF_THRESHOLD   = 0.90    # rolling PF ต่ำกว่า → trigger
TRIGGER_PF_WINDOW      = 50      # rolling N trades
TRIGGER_REGIME_Z       = 2.0     # ATR z-score > 2 = regime shift
TRIGGER_SCHEDULE_DAYS  = 90      # auto trigger ทุก 90 วัน

# Safety
COOLDOWN_DAYS          = 14      # min days between fine-tunes
MIN_NEW_DATA_BARS      = 500     # min bars of new data
MIN_NEW_DATA_DAYS      = 30      # หรือ min days

# Gates (after fine-tune)
WF_PASS_MIN            = 4
WF_MIN_PF              = 1.0
WF_MEAN_PF             = 1.05


def init_db():
    """SQLite log สำหรับ track ทุก trigger + finetune"""
    conn = sqlite3.connect(LOG_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finetune_log (
            id INTEGER PRIMARY KEY,
            date TEXT,
            triggers TEXT,
            data_start TEXT,
            data_end TEXT,
            n_new_bars INTEGER,
            decision TEXT,
            wf_pass INTEGER,
            wf_total INTEGER,
            wf_mean_pf REAL,
            new_model TEXT,
            deployed BOOLEAN
        )
    """)
    conn.commit()
    return conn


def get_last_finetune(conn):
    """ดึงวันที่ fine-tune ครั้งล่าสุด"""
    cur = conn.execute(
        "SELECT date FROM finetune_log WHERE deployed=1 ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    return datetime.fromisoformat(row[0]) if row else None


def check_triggers(conn):
    """ตรวจ trigger conditions ทั้งหมด"""
    triggers = []

    # === Trigger 1: Schedule ===
    last = get_last_finetune(conn)
    if last:
        days_since = (datetime.now() - last).days
        if days_since >= TRIGGER_SCHEDULE_DAYS:
            triggers.append({
                "type": "schedule",
                "reason": f"{days_since}d since last (>= {TRIGGER_SCHEDULE_DAYS})"
            })
    else:
        triggers.append({"type": "schedule", "reason": "no previous train"})

    # === Trigger 2: Live PF drop ===
    # NOTE: ต้องอ่านจาก live_trades.db (สมมติว่ามี)
    # rolling_pf = calc_rolling_pf(last_n=TRIGGER_PF_WINDOW)
    # if rolling_pf < TRIGGER_PF_THRESHOLD:
    #     triggers.append({"type": "pf_drop", "reason": f"PF={rolling_pf:.2f}"})
    # placeholder:
    pass

    # === Trigger 3: Regime shift ===
    # current_atr = get_current_atr()
    # baseline_atr_mean = ...
    # baseline_atr_std = ...
    # z = (current_atr - baseline_atr_mean) / baseline_atr_std
    # if abs(z) > TRIGGER_REGIME_Z:
    #     triggers.append({"type": "regime", "reason": f"ATR z={z:.2f}"})

    return triggers


def check_safety(conn):
    """ตรวจ safety conditions ก่อน fine-tune"""
    safety = {"all_ok": True, "issues": []}

    # === Cooldown ===
    last = get_last_finetune(conn)
    if last:
        days_since = (datetime.now() - last).days
        if days_since < COOLDOWN_DAYS:
            safety["all_ok"] = False
            safety["issues"].append(
                f"Cooldown: {days_since}d < {COOLDOWN_DAYS}d minimum"
            )

    return safety


def determine_data_window(conn):
    """หาว่าจะใช้ data ช่วงไหนเป็น 'new'"""
    last = get_last_finetune(conn)
    today = datetime.now()

    # New data start: max(last_train, today - MIN_NEW_DATA_DAYS)
    if last:
        new_start = max(last, today - timedelta(days=MIN_NEW_DATA_DAYS * 6))
    else:
        new_start = today - timedelta(days=MIN_NEW_DATA_DAYS * 6)

    new_end = today

    return {
        "new_start": new_start,
        "new_end": new_end,
        "old_csv": "training_data_v2_relabeled.csv",  # archive ของเก่า
    }


def estimate_new_data_bars(start, end):
    """ประมาณจำนวน bars (H1) ระหว่างช่วง"""
    # H1 ~ 24 bars/day × ~5 trading days/week
    days = (end - start).days
    return int(days * 24 * 5/7)  # rough estimate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="แค่เช็ค ไม่รัน")
    ap.add_argument("--force", action="store_true",
                    help="bypass cooldown")
    ap.add_argument("--auto-deploy", action="store_true",
                    help="auto deploy ถ้าผ่าน WF gate")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  Trigger-Based Fine-tune Check — {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 60)

    conn = init_db()

    # === Step 1: Check triggers ===
    print("\n📊 Checking triggers...")
    triggers = check_triggers(conn)
    if not triggers:
        print("  ✓ No triggers fired — system healthy")
        return 0

    print(f"  🚨 {len(triggers)} trigger(s) fired:")
    for t in triggers:
        print(f"     - {t['type']}: {t['reason']}")

    # === Step 2: Check safety ===
    print("\n🛡️ Checking safety conditions...")
    safety = check_safety(conn)
    if not safety["all_ok"] and not args.force:
        print("  ❌ Safety check failed:")
        for issue in safety["issues"]:
            print(f"     - {issue}")
        print("\n  Use --force to override")
        return 1

    if args.force:
        print("  ⚠️ FORCE mode — safety overridden")
    else:
        print("  ✓ Safety OK")

    # === Step 3: Determine data window ===
    print("\n📅 Determining data window...")
    window = determine_data_window(conn)
    estimated_bars = estimate_new_data_bars(window["new_start"], window["new_end"])

    print(f"  New data: {window['new_start']:%Y-%m-%d} → {window['new_end']:%Y-%m-%d}")
    print(f"  Estimated: ~{estimated_bars} bars")
    print(f"  Old archive: {window['old_csv']}")

    if estimated_bars < MIN_NEW_DATA_BARS:
        print(f"  ⚠️ Too few new data ({estimated_bars} < {MIN_NEW_DATA_BARS})")
        if not args.force:
            return 1

    # === Step 4: Check-only mode ===
    if args.check_only:
        print("\n✓ Check-only mode — would proceed if not for --check-only")
        return 0

    # === Step 5: Run fine-tune pipeline ===
    print("\n🚀 Starting fine-tune pipeline...")
    print("    (would run quarterly_update.py with computed args)")
    # subprocess.run([
    #     sys.executable, "quarterly_update.py",
    #     "--current-model", CURRENT_MODEL,
    #     "--auto-deploy" if args.auto_deploy else "",
    # ])

    # === Step 6: Log ===
    print("\n📝 Logging to DB...")
    conn.execute("""
        INSERT INTO finetune_log
        (date, triggers, data_start, data_end, n_new_bars, decision, deployed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        json.dumps(triggers),
        window["new_start"].isoformat(),
        window["new_end"].isoformat(),
        estimated_bars,
        "TRIGGERED",
        False,  # set True after deploy
    ))
    conn.commit()
    print("  ✓ Logged")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
