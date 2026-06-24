"""
Signal Server — สร้าง signal ต่อเนื่องให้ MT4/MT5 อ่าน
------------------------------------------------------
วิธีทำงาน:
  1) อ่านไฟล์ features.csv (MT4 EA เขียนทุก bar ใหม่)
  2) โหลด model ทำนาย
  3) เขียนผล signal.txt ให้ MT4 อ่าน

สร้างไฟล์ 2 ตัวใน MT4/Files/:
  - features.csv  (EA เขียน: features ของ bar ล่าสุด)
  - signal.txt    (Python เขียน: UP/DOWN/FLAT + confidence)

Usage:
    python signal_server.py my_model "C:/Users/<you>/AppData/Roaming/MetaQuotes/Terminal/.../MQL4/Files"
"""

import sys
import time
from pathlib import Path
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python signal_server.py <model_name> <mt4_files_dir>")
        sys.exit(1)

    model_name = sys.argv[1].replace(".pkl", "")
    files_dir = Path(sys.argv[2])
    feat_path = files_dir / "features.csv"
    sig_path  = files_dir / "signal.txt"

    print(f"[server] loading model: {model_name}")
    from pycaret.classification import load_model, predict_model
    model = load_model(model_name)

    print(f"[server] watching: {feat_path}")
    print(f"[server] writing to: {sig_path}")
    print("[server] Ctrl+C to stop\n")

    last_mtime = 0.0
    while True:
        try:
            if feat_path.exists():
                mtime = feat_path.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        df = pd.read_csv(feat_path)
                        if len(df) == 0:
                            time.sleep(0.2)
                            continue

                        pred = predict_model(model, data=df.tail(1))
                        label = pred["prediction_label"].iloc[-1]
                        conf = pred["prediction_score"].iloc[-1] if "prediction_score" in pred.columns else 1.0

                        sig_path.write_text(f"{label},{conf:.4f}")
                        print(f"[{time.strftime('%H:%M:%S')}] {label} (conf={conf:.3f})")
                    except Exception as e:
                        print(f"[server] error: {e}")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[server] stopped")
            break


if __name__ == "__main__":
    main()
