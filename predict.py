"""
ทดสอบ model ที่เทรนไว้กับข้อมูลใหม่
-----------------------------------
วิธีใช้:
    python predict.py my_model new_data.csv

output:
    predictions.csv (เพิ่มคอลัมน์ prediction_label + prediction_score)
"""

import sys
from pathlib import Path
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_name_without_pkl> <input_csv>")
        print("Example: python predict.py my_model new_data.csv")
        return

    model_name = sys.argv[1].replace(".pkl", "")
    input_csv = sys.argv[2]

    print(f"[predict] loading model: {model_name}.pkl")
    from pycaret.classification import load_model, predict_model

    model = load_model(model_name)

    print(f"[predict] reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  rows: {len(df):,}")

    print("[predict] predicting ...")
    result = predict_model(model, data=df)

    out_path = Path(input_csv).with_stem(Path(input_csv).stem + "_predicted")
    result.to_csv(out_path, index=False)

    print(f"\n[predict] saved: {out_path}")

    # สรุป prediction
    if "prediction_label" in result.columns:
        print("\n=== Prediction summary ===")
        print(result["prediction_label"].value_counts())
        if "prediction_score" in result.columns:
            print(f"\nMean confidence: {result['prediction_score'].mean():.4f}")


if __name__ == "__main__":
    main()
