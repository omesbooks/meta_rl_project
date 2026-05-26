"""
Re-label dataset ให้ class balance ดีขึ้น — ไม่ต้องไปรัน MT4 ใหม่
-----------------------------------------------------------------
วิธีใช้:
    python relabel.py EURUSD_H1.csv

จะถามให้เลือกวิธี label:
    1) Quantile split (33/33/33) — สมดุลสุด
    2) Fixed threshold — ใส่ threshold เอง
    3) Binary (UP/DOWN เท่านั้น) — ง่ายสุดสำหรับ ML

output: EURUSD_H1_relabeled.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", help="CSV file with future_return column")
    ap.add_argument("--mode", default=None,
                    choices=["quantile", "fixed", "binary"],
                    help="non-interactive relabel mode")
    ap.add_argument("--threshold", type=float, default=0.001,
                    help="threshold for --mode fixed")
    args = ap.parse_args()

    src = args.csv_file
    df = pd.read_csv(src)

    if "future_return" not in df.columns:
        print("ERROR: ไฟล์ไม่มีคอลัมน์ future_return")
        sys.exit(1)

    print(f"\nloaded: {src} ({len(df):,} rows)")
    print("\nClass distribution (เดิม):")
    print(df["target"].value_counts())
    print("\nfuture_return stats:")
    print(df["future_return"].describe())

    if args.mode:
        choice = {"quantile": "1", "fixed": "2", "binary": "3"}[args.mode]
    else:
        print("\n" + "=" * 50)
        print("เลือกวิธี re-label:")
        print("  1) Quantile (33/33/33) — balance ดีสุด [แนะนำ]")
        print("  2) Fixed threshold — เลือกเอง")
        print("  3) Binary UP/DOWN — ข้าม FLAT")
        choice = input("เลือก (1/2/3) [default=1]: ").strip() or "1"

    fr = df["future_return"]

    if choice == "1":
        q_low = fr.quantile(0.33)
        q_high = fr.quantile(0.67)
        print(f"\n  quantile 33% = {q_low:.6f}")
        print(f"  quantile 67% = {q_high:.6f}")
        df["target"] = pd.cut(
            fr,
            bins=[-float("inf"), q_low, q_high, float("inf")],
            labels=["DOWN", "FLAT", "UP"],
        )

    elif choice == "2":
        if args.mode:
            t = args.threshold
        else:
            t = input("  threshold (เช่น 0.001): ").strip()
            t = float(t) if t else 0.001
        print(f"\n  using +/- {t}")
        df["target"] = "FLAT"
        df.loc[fr >= t, "target"] = "UP"
        df.loc[fr <= -t, "target"] = "DOWN"

    elif choice == "3":
        # Binary: drop bars ที่ flat มาก
        t = fr.abs().median()  # ตัด bar ที่ผันผวนน้อยกว่า median ออก
        print(f"\n  keep only bars with |future_return| > {t:.6f}")
        df = df[fr.abs() > t].copy()
        df["target"] = "DOWN"
        df.loc[df["future_return"] > 0, "target"] = "UP"
        print(f"  rows หลัง filter: {len(df):,}")

    else:
        print("เลือกไม่ถูกต้อง")
        sys.exit(1)

    print("\nClass distribution (ใหม่):")
    vc = df["target"].value_counts()
    print(vc)
    print("\nPercentages:")
    print((vc / len(df) * 100).round(1))

    # save
    out = Path(src).with_stem(Path(src).stem + "_relabeled")
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    print("\nไปเทรนใหม่ด้วยไฟล์นี้ได้เลย!")


if __name__ == "__main__":
    main()
