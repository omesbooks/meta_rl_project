"""
วิเคราะห์ว่า model มั่นใจแค่ไหนแล้วทายแม่น?
ใช้ดูว่ามี confidence threshold ที่ทำให้ win rate > 50% ไหม

Usage:
    python analyze_confidence.py <model> <csv>
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOLD_BARS = 5
SPREAD_PCT = 0.0002
COMMISSION = 0.0001


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_confidence.py <model> <csv>")
        return

    model_name = sys.argv[1].replace(".pkl", "")
    csv_path = sys.argv[2]

    from pycaret.classification import load_model, predict_model
    print(f"[load] {model_name}.pkl")
    model = load_model(model_name)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    leaky = [c for c in df.columns if any(k in c.lower() for k in ("future_", "forward_", "next_"))]
    df_pred = df.drop(columns=leaky) if leaky else df

    print(f"[predict] {len(df):,} rows ...")
    pred = predict_model(model, data=df_pred)
    df["signal"] = pred["prediction_label"]
    df["conf"] = pred["prediction_score"] if "prediction_score" in pred.columns else 1.0

    # คำนวณผลของ trade แต่ละ bar
    rows = []
    for i in range(len(df) - HOLD_BARS - 1):
        sig = df.at[i, "signal"]
        if sig == "FLAT":
            continue
        entry = df.at[i, "close"]
        exit = df.at[i + HOLD_BARS, "close"]
        if sig == "UP":
            ret = (exit - entry) / entry - SPREAD_PCT - COMMISSION
        else:
            ret = (entry - exit) / entry - SPREAD_PCT - COMMISSION
        rows.append({"conf": df.at[i, "conf"], "return": ret, "win": ret > 0, "signal": sig})

    tdf = pd.DataFrame(rows)
    print(f"\nTotal non-FLAT trades: {len(tdf):,}")
    print(f"Confidence distribution:")
    print(tdf["conf"].describe().round(4))

    # Bucket analysis
    print("\n" + "=" * 70)
    print("  Performance by confidence bucket")
    print("=" * 70)
    print(f"  {'Bucket':<15} {'#trades':>10} {'WinRate':>10} {'AvgRet':>12} {'TotalRet':>12}")
    print("-" * 70)

    buckets = [(0.33, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
               (0.55, 0.60), (0.60, 0.70), (0.70, 1.00)]
    for lo, hi in buckets:
        sub = tdf[(tdf["conf"] >= lo) & (tdf["conf"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["win"].mean() * 100
        avg = sub["return"].mean() * 100
        tot = sub["return"].sum() * 100
        print(f"  [{lo:.2f}-{hi:.2f})    {len(sub):>10,} {wr:>9.2f}% {avg:>+11.4f}% {tot:>+11.2f}%")

    # หา threshold ที่ดีที่สุด
    print("\n" + "=" * 70)
    print("  หา confidence threshold ที่ดีที่สุด")
    print("=" * 70)
    print(f"  {'Min Conf':<12} {'#trades':>10} {'WinRate':>10} {'TotalRet':>12} {'ProfitFactor':>14}")
    print("-" * 70)

    best_pf = 0
    best_thr = 0
    for thr in np.arange(0.35, 0.80, 0.05):
        sub = tdf[tdf["conf"] >= thr]
        if len(sub) < 20:
            continue
        wr = sub["win"].mean() * 100
        tot = sub["return"].sum() * 100
        wins_sum = sub[sub["return"] > 0]["return"].sum()
        losses_sum = abs(sub[sub["return"] <= 0]["return"].sum())
        pf = wins_sum / losses_sum if losses_sum > 0 else float("inf")
        marker = " ⭐" if pf > best_pf else ""
        if pf > best_pf:
            best_pf = pf
            best_thr = thr
        print(f"  conf>={thr:.2f}   {len(sub):>10,} {wr:>9.2f}% {tot:>+11.2f}% {pf:>13.2f}{marker}")

    print("\n" + "=" * 70)
    if best_pf > 1.2:
        print(f"  ✓ ดี! ใช้ conf >= {best_thr:.2f} → Profit Factor = {best_pf:.2f}")
        print(f"    → เอาไปใส่ใน backtest.py เพิ่ม filter นี้")
    elif best_pf > 1.0:
        print(f"  ~ พอใช้: conf >= {best_thr:.2f} → PF = {best_pf:.2f} (แต่ไม่เยอะ)")
    else:
        print(f"  ✗ model ไม่มี edge ที่ confidence level ไหนเลย")
        print(f"    → ต้องลองวิธีอื่น: เพิ่ม features, เปลี่ยน TF, หรือเปลี่ยน target")
    print("=" * 70)


if __name__ == "__main__":
    main()
