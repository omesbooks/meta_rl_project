"""
Backtest model prediction เหมือนเทรดจริง
----------------------------------------
กฎ:
  - UP signal   -> Long 1 unit, ปิดหลัง N bars
  - DOWN signal -> Short 1 unit, ปิดหลัง N bars
  - FLAT        -> ไม่ trade
  - คิด spread + commission

Usage:
    python backtest.py my_model training_data.csv
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np

# fix Thai chars on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


HOLD_BARS   = 5       # ถือ N bars แล้วปิด (ให้ตรงกับ ForwardBars ใน EA)
SPREAD_PCT  = 0.0002  # 0.02% spread
COMMISSION  = 0.0001  # 0.01% ต่อ trade
INIT_EQUITY = 10000.0


def run_backtest(model_name: str, csv_path: str):
    from pycaret.classification import load_model, predict_model

    print(f"[bt] loading model: {model_name}.pkl")
    model = load_model(model_name)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    print(f"[bt] data rows: {len(df):,}")

    # Leakage guard — ห้ามส่ง future column เข้า model
    leaky = [c for c in df.columns if any(k in c.lower()
             for k in ("future_", "forward_", "next_"))]
    if leaky:
        print(f"[bt] WARNING: dropping leaky columns before predict: {leaky}")
        df_pred = df.drop(columns=leaky)
    else:
        df_pred = df

    print("[bt] predicting ...")
    pred = predict_model(model, data=df_pred)
    df["signal"] = pred["prediction_label"]
    if "prediction_score" in pred.columns:
        df["conf"] = pred["prediction_score"]
    else:
        df["conf"] = 1.0

    # ----------- simulate -----------
    equity = INIT_EQUITY
    trades = []
    curve = [equity]

    i = 0
    while i < len(df) - HOLD_BARS - 1:
        sig = df.at[i, "signal"]
        entry_price = df.at[i, "close"]
        exit_price  = df.at[i + HOLD_BARS, "close"]

        if sig == "UP":
            pnl_pct = (exit_price - entry_price) / entry_price - SPREAD_PCT - COMMISSION
        elif sig == "DOWN":
            pnl_pct = (entry_price - exit_price) / entry_price - SPREAD_PCT - COMMISSION
        else:
            i += 1
            curve.append(equity)
            continue

        pnl = equity * pnl_pct
        equity += pnl
        trades.append({
            "bar": i,
            "signal": sig,
            "conf": df.at[i, "conf"],
            "entry": entry_price,
            "exit": exit_price,
            "pnl_pct": pnl_pct,
            "pnl": pnl,
            "equity": equity,
        })
        curve.append(equity)
        i += HOLD_BARS  # ข้ามช่วงที่ถืออยู่

    # ----------- stats -----------
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("\n⚠ ไม่มี trade เลย — model ออก FLAT หมด")
        return

    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]

    total_pnl = tdf["pnl"].sum()
    win_rate = len(wins) / len(tdf)
    profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) > 0 else float("inf")
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

    # max drawdown
    eq = np.array(curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # sharpe (ง่ายๆ)
    rets = tdf["pnl_pct"].values
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    print("\n" + "=" * 50)
    print(f"  Backtest result: {model_name}")
    print("=" * 50)
    print(f"  Total trades      : {len(tdf):,}")
    print(f"    UP  trades      : {(tdf['signal']=='UP').sum():,}")
    print(f"    DOWN trades     : {(tdf['signal']=='DOWN').sum():,}")
    print(f"  Win rate          : {win_rate:.2%}")
    print(f"  Profit factor     : {profit_factor:.2f}")
    print(f"  Avg win           : {avg_win:+.2f}")
    print(f"  Avg loss          : {avg_loss:+.2f}")
    print(f"  Total P&L         : {total_pnl:+.2f}")
    print(f"  Final equity      : {equity:,.2f} (เริ่ม {INIT_EQUITY:,.0f})")
    print(f"  Return            : {(equity/INIT_EQUITY - 1):+.2%}")
    print(f"  Max drawdown      : {max_dd:.2%}")
    print(f"  Sharpe (~annual)  : {sharpe:.2f}")
    print("=" * 50)

    # save trades
    out = Path(csv_path).with_stem(Path(csv_path).stem + "_trades")
    tdf.to_csv(out, index=False)
    print(f"\n[bt] trades -> {out}")

    # equity curve chart (ถ้ามี matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(curve)
        plt.title(f"Equity Curve — {model_name}")
        plt.xlabel("Bar")
        plt.ylabel("Equity")
        plt.grid(True)
        chart_path = Path(csv_path).with_stem(Path(csv_path).stem + "_equity").with_suffix(".png")
        plt.savefig(chart_path, dpi=100, bbox_inches="tight")
        print(f"[bt] equity chart -> {chart_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python backtest.py <model_name> <csv_file>")
        sys.exit(1)
    run_backtest(sys.argv[1].replace(".pkl", ""), sys.argv[2])
