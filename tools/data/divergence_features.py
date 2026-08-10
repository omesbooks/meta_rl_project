"""
Divergence Signal Collector — เก็บสัญญาณ divergence เป็น features
==================================================================
ตรวจ divergence ระหว่างราคา (swing high/low) กับ oscillator (RSI/MACD hist)
แบบ non-repainting: pivot ต้องรอแท่งขวายืนยันก่อน สัญญาณประทับที่ "แท่งยืนยัน"
(pivot_bar + right) เท่านั้น — ไม่มี look-ahead

ชนิดสัญญาณ (regular):
  - Bullish  : ราคาทำ Lower Low  แต่ oscillator ทำ Higher Low  (กลับตัวขึ้น)
  - Bearish  : ราคาทำ Higher High แต่ oscillator ทำ Lower High (กลับตัวลง)
ชนิดสัญญาณ (hidden, เปิดด้วย --hidden):
  - Bullish  : ราคาทำ Higher Low  แต่ oscillator ทำ Lower Low  (ไปต่อขึ้น)
  - Bearish  : ราคาทำ Lower High แต่ oscillator ทำ Higher High (ไปต่อลง)

คอลัมน์ที่เพิ่ม (ต่อ oscillator):
  <osc>_div_bull / <osc>_div_bear          flag 0/1 ณ แท่งยืนยัน
  <osc>_div_hbull / <osc>_div_hbear        (เฉพาะ --hidden)
รวมทุก oscillator:
  div_bull_age / div_bear_age              อายุสัญญาณล่าสุด (แท่ง, cap แล้วหาร cap
                                           → 0=สดใหม่ .. 1=ไม่มีสัญญาณเร็วๆ นี้)

Usage:
    python tools/data/divergence_features.py uj_h4_85_percent.csv
    python tools/data/divergence_features.py data.csv --osc rsi_14 macd_hist --hidden
    python tools/data/divergence_features.py data.csv --edge   # + สถิติ forward return

Output: <input>_div.csv
"""
import sys, io, argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def find_confirmed_pivots(values: np.ndarray, left: int, right: int, kind: str):
    """Return list of (pivot_idx, confirm_idx) for confirmed swing points.

    pivot ยืนยันเมื่อเดินมาถึงแท่ง pivot_idx + right แล้วเท่านั้น (causal)
    kind='low'  : values[p] ต่ำกว่าเพื่อนซ้ายแบบเด็ดขาด และไม่แพ้เพื่อนขวา
    kind='high' : กลับด้านกัน
    """
    n = len(values)
    pivots = []
    for p in range(left, n - right):
        v = values[p]
        if np.isnan(v):
            continue
        left_win = values[p - left:p]
        right_win = values[p + 1:p + 1 + right]
        if kind == "low":
            if np.all(v < left_win) and np.all(v <= right_win):
                pivots.append((p, p + right))
        else:
            if np.all(v > left_win) and np.all(v >= right_win):
                pivots.append((p, p + right))
    return pivots


def collect_divergences(df: pd.DataFrame, osc_cols: list[str], left: int, right: int,
                        min_span: int, max_span: int, include_hidden: bool,
                        age_cap: int) -> tuple[pd.DataFrame, dict]:
    n = len(df)
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)

    pivot_lows = find_confirmed_pivots(lows, left, right, "low")
    pivot_highs = find_confirmed_pivots(highs, left, right, "high")

    out = {}
    counts = {}
    any_bull = np.zeros(n, dtype=bool)
    any_bear = np.zeros(n, dtype=bool)

    for osc in osc_cols:
        osc_v = df[osc].to_numpy(dtype=float)
        bull = np.zeros(n, dtype=np.int8)
        bear = np.zeros(n, dtype=np.int8)
        hbull = np.zeros(n, dtype=np.int8)
        hbear = np.zeros(n, dtype=np.int8)

        # เทียบ swing low คู่ล่าสุดที่ "ยืนยันแล้วทั้งคู่" ณ เวลานั้น
        for (p1, c1), (p2, c2) in zip(pivot_lows, pivot_lows[1:]):
            span = p2 - p1
            if span < min_span or span > max_span:
                continue
            if np.isnan(osc_v[p1]) or np.isnan(osc_v[p2]):
                continue
            price_ll = lows[p2] < lows[p1]
            price_hl = lows[p2] > lows[p1]
            osc_hl = osc_v[p2] > osc_v[p1]
            osc_ll = osc_v[p2] < osc_v[p1]
            if price_ll and osc_hl:
                bull[c2] = 1
            elif include_hidden and price_hl and osc_ll:
                hbull[c2] = 1

        for (p1, c1), (p2, c2) in zip(pivot_highs, pivot_highs[1:]):
            span = p2 - p1
            if span < min_span or span > max_span:
                continue
            if np.isnan(osc_v[p1]) or np.isnan(osc_v[p2]):
                continue
            price_hh = highs[p2] > highs[p1]
            price_lh = highs[p2] < highs[p1]
            osc_lh = osc_v[p2] < osc_v[p1]
            osc_hh = osc_v[p2] > osc_v[p1]
            if price_hh and osc_lh:
                bear[c2] = 1
            elif include_hidden and price_lh and osc_hh:
                hbear[c2] = 1

        out[f"{osc}_div_bull"] = bull
        out[f"{osc}_div_bear"] = bear
        counts[f"{osc}_div_bull"] = int(bull.sum())
        counts[f"{osc}_div_bear"] = int(bear.sum())
        if include_hidden:
            out[f"{osc}_div_hbull"] = hbull
            out[f"{osc}_div_hbear"] = hbear
            counts[f"{osc}_div_hbull"] = int(hbull.sum())
            counts[f"{osc}_div_hbear"] = int(hbear.sum())
        any_bull |= bull.astype(bool)
        any_bear |= bear.astype(bool)

    # อายุสัญญาณล่าสุด (รวมทุก oscillator, เฉพาะ regular)
    def age_series(flags: np.ndarray) -> np.ndarray:
        age = np.full(n, age_cap, dtype=float)
        last = -age_cap
        for i in range(n):
            if flags[i]:
                last = i
            age[i] = min(i - last, age_cap)
        return age / age_cap

    out["div_bull_age"] = age_series(any_bull)
    out["div_bear_age"] = age_series(any_bear)
    counts["any_bull"] = int(any_bull.sum())
    counts["any_bear"] = int(any_bear.sum())
    return pd.DataFrame(out, index=df.index), counts


def edge_report(df: pd.DataFrame, feat: pd.DataFrame, horizons=(5, 10, 20)):
    """สถิติ forward return หลังสัญญาณ (ใช้อนาคตเพื่อ 'ประเมิน' เท่านั้น ไม่ใช่ feature)"""
    close = df["close"].to_numpy(dtype=float)
    print("\n[edge check] mean forward return (%) หลังสัญญาณ เทียบ baseline ทุกแท่ง")
    header = "signal".ljust(24) + "n".rjust(6) + "".join(f"  fwd{h}".rjust(9) for h in horizons)
    print(header)
    rows = [("baseline (ทุกแท่ง)", np.ones(len(df), dtype=bool))]
    for col in feat.columns:
        if col.endswith(("_age",)):
            continue
        rows.append((col, feat[col].to_numpy(dtype=bool)))
    for name, mask in rows:
        n_sig = int(mask.sum())
        cells = []
        for h in horizons:
            fwd = np.full(len(close), np.nan)
            fwd[:-h] = (close[h:] - close[:-h]) / close[:-h] * 100
            vals = fwd[mask]
            vals = vals[~np.isnan(vals)]
            cells.append(f"{np.mean(vals):+.3f}" if len(vals) else "   -")
        print(name.ljust(24) + str(n_sig).rjust(6) + "".join(c.rjust(9) for c in cells))


def main():
    ap = argparse.ArgumentParser(description="เก็บสัญญาณ divergence เป็น features (non-repainting)")
    ap.add_argument("csv")
    ap.add_argument("--osc", nargs="+", default=["rsi_14", "macd_hist"],
                    help="คอลัมน์ oscillator ที่ใช้ตรวจ (default: rsi_14 macd_hist)")
    ap.add_argument("--left", type=int, default=3, help="แท่งซ้ายของ pivot (default 3)")
    ap.add_argument("--right", type=int, default=3,
                    help="แท่งขวายืนยัน pivot (default 3) — สัญญาณช้าลงเท่านี้แต่ไม่ repaint")
    ap.add_argument("--min_span", type=int, default=5, help="ระยะ swing คู่ขั้นต่ำ (แท่ง)")
    ap.add_argument("--max_span", type=int, default=60, help="ระยะ swing คู่สูงสุด (แท่ง)")
    ap.add_argument("--hidden", action="store_true", help="เก็บ hidden divergence ด้วย")
    ap.add_argument("--age_cap", type=int, default=50, help="เพดานอายุสัญญาณ (default 50)")
    ap.add_argument("--edge", action="store_true", help="พิมพ์สถิติ forward return หลังสัญญาณ")
    args = ap.parse_args()

    # guard rails: negative right = look-ahead, age_cap 0 = division by zero
    if args.left < 1 or args.right < 1:
        ap.error("--left and --right must be >= 1 (right < 1 would look ahead)")
    if args.min_span < 1 or args.max_span < args.min_span:
        ap.error("--min_span must be >= 1 and --max_span >= --min_span")
    if args.age_cap < 1:
        ap.error("--age_cap must be >= 1")

    path = Path(args.csv)
    df = pd.read_csv(path)
    missing = [c for c in ["high", "low", "close"] + args.osc if c not in df.columns]
    if missing:
        print(f"[error] ไม่พบคอลัมน์: {missing}")
        sys.exit(1)

    feat, counts = collect_divergences(
        df, args.osc, args.left, args.right,
        args.min_span, args.max_span, args.hidden, args.age_cap)

    print(f"[data] {path.name}: {len(df):,} แท่ง")
    print(f"[pivot] left={args.left} right={args.right} "
          f"(สัญญาณยืนยันช้า {args.right} แท่ง — non-repainting)")
    print("[collected]")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if args.edge:
        edge_report(df, feat)

    out_df = pd.concat([df, feat], axis=1)
    out_path = path.with_name(path.stem + "_div.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[save] -> {out_path.name}  ({len(out_df.columns)} คอลัมน์, "
          f"เพิ่ม {len(feat.columns)} คอลัมน์)")

    # Carry the params sidecar to the new stem — train/export forward only the
    # exact output stem's .params.json, so dropping it would silently reset the
    # EA's feature settings to defaults. Also record DIV_* keys (same names the
    # DataCollector writes) so the EA computes divergence with these settings.
    sidecar_src = path.with_name(path.stem + ".params.json")
    params = {}
    if sidecar_src.exists():
        params = json.loads(sidecar_src.read_text(encoding="utf-8-sig"))
    else:
        print(f"[warn] ไม่พบ {sidecar_src.name} — sidecar ใหม่จะมีเฉพาะ DIV_* keys")
    for osc in args.osc:
        m = re.fullmatch(r"rsi_(\d+)", osc)
        if m:
            params["DIV_RSI_PERIOD"] = int(m.group(1))
            break
    if "macd_hist" in args.osc:
        params["DIV_MACD_FAST"] = int(params.get("MACD_FAST", 12))
        params["DIV_MACD_SLOW"] = int(params.get("MACD_SLOW", 26))
        params["DIV_MACD_SIGNAL"] = int(params.get("MACD_SIGNAL", 9))
    params["DIV_PIVOT_LEFT"] = args.left
    params["DIV_PIVOT_RIGHT"] = args.right
    params["DIV_MIN_SPAN"] = args.min_span
    params["DIV_MAX_SPAN"] = args.max_span
    params["DIV_AGE_CAP"] = args.age_cap
    sidecar_out = out_path.with_name(out_path.stem + ".params.json")
    sidecar_out.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(f"[save] -> {sidecar_out.name}  (params sidecar สำหรับ export/EA parity)")


if __name__ == "__main__":
    main()
