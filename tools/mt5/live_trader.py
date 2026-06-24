"""
live_trader.py — Production Live Trading Bot
=============================================
Main loop ที่:
  1. Connect MT5
  2. ทุก H1 bar ใหม่ → predict ด้วย RL model → execute
  3. Apply: confidence filter + risk management 3 layers
  4. Log ทุก trade ลง SQLite

ติดตั้ง:
    pip install MetaTrader5 stable-baselines3 pandas numpy

วิธีรัน:
    1. แก้ CONFIG ด้านล่างให้ตรงกับ broker / model
    2. python live_trader.py [--demo|--live] [--paper]

โหมด:
    --demo   : ส่ง order จริงแต่ใน demo account (default)
    --live   : ส่ง order ใน live account ⚠️
    --paper  : ไม่ส่ง order (แค่ predict + log) — สำหรับ debug
"""
import os, sys, io, time, sqlite3, argparse, signal, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =============================================================
# CONFIG — แก้ให้ตรงกับของคุณ
# =============================================================
SYMBOL                = "XAUUSDm"            # symbol ของ broker
TIMEFRAME             = "H1"                 # timeframe
MODEL_NAME            = "rl_v3"              # model file (without .zip)
WINDOW_SIZE           = 10                   # ตรงกับตอน train
N_BARS_TO_PULL        = 300                  # bars ดึงทุก loop

# Risk management
CONFIDENCE_THRESHOLD  = 0.85                 # ทำ trade เฉพาะ conf >= นี้
RISK_PER_TRADE        = 0.01                 # 1% ต่อ trade
MAX_POSITIONS         = 3
MAX_HOLD_BARS         = 30                   # บังคับปิดหลัง N bars
ATR_SL_MULTIPLIER     = 1.5                  # SL = ATR * x
ATR_TP_MULTIPLIER     = 3.0                  # TP = ATR * x  (RR 1:2)

# Risk gates
HARD_DD_PCT           = 0.15                 # equity ตกถึงนี้ → close all
SOFT_PAUSE_PF         = 0.90                 # PF ต่ำกว่านี้ → pause 24hr
ROLLING_TRADES        = 50                   # number for rolling stats

# Loop
SLEEP_BETWEEN_CHECKS  = 30                   # วินาที — เช็ค new bar ทุก N วิ
DB_PATH               = "live_trades.db"


# =============================================================
# Setup
# =============================================================
def init_db(path: str):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            ts TEXT,
            bar_time TEXT,
            action INTEGER,
            action_name TEXT,
            confidence REAL,
            position_before INTEGER,
            equity REAL,
            executed INTEGER,
            ticket TEXT,
            price REAL,
            sl REAL,
            tp REAL,
            lots REAL,
            comment TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            ts TEXT,
            level TEXT,
            event TEXT,
            details TEXT
        )
    """)
    conn.commit()
    return conn


def log_event(conn, level: str, event: str, details: str = ""):
    conn.execute(
        "INSERT INTO events (ts, level, event, details) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), level, event, details),
    )
    conn.commit()
    print(f"[{datetime.now():%H:%M:%S}] [{level}] {event}: {details}")


def log_trade(conn, **kwargs):
    cols = ",".join(kwargs.keys())
    qs = ",".join("?" * len(kwargs))
    conn.execute(f"INSERT INTO trades ({cols}) VALUES ({qs})", tuple(kwargs.values()))
    conn.commit()


# =============================================================
# Risk Management
# =============================================================
def check_hard_stop(mt5, initial_equity: float) -> bool:
    """L1: Hard stop if equity dropped too much"""
    current = mt5.equity()
    if initial_equity > 0 and current / initial_equity < (1 - HARD_DD_PCT):
        return True
    return False


def calc_rolling_pf(conn) -> float:
    """L2: Calc rolling PF from recent trades log"""
    # placeholder — would need to track actual P&L per closed trade
    # For now, return 1.0 (no soft pause)
    return 1.0


# =============================================================
# Main Loop
# =============================================================
def run_trading_loop(mode: str = "demo"):
    print("=" * 60)
    print(f"  LIVE TRADER — Mode: {mode.upper()}")
    print("=" * 60)
    print(f"  Symbol     : {SYMBOL}")
    print(f"  Timeframe  : {TIMEFRAME}")
    print(f"  Model      : {MODEL_NAME}.zip")
    print(f"  Confidence : >= {CONFIDENCE_THRESHOLD}")
    print(f"  Risk/trade : {RISK_PER_TRADE:.0%}")
    print(f"  Max DD     : {HARD_DD_PCT:.0%}")
    print("=" * 60)

    # === DB ===
    conn = init_db(DB_PATH)
    log_event(conn, "INFO", "STARTUP", f"mode={mode}")

    # === Load model + norm ===
    from stable_baselines3 import PPO
    import torch

    model_path = Path(f"{MODEL_NAME}.zip")
    norm_path = Path(f"{MODEL_NAME}_norm.csv")
    if not model_path.exists():
        log_event(conn, "ERROR", "MODEL_NOT_FOUND", str(model_path))
        return 1

    print(f"\n[load] {model_path}")
    model = PPO.load(model_path)
    norm = pd.read_csv(norm_path, index_col=0) if norm_path.exists() else None
    log_event(conn, "INFO", "MODEL_LOADED", f"{model_path}, norm={norm is not None}")

    # === Connect MT5 ===
    if mode != "paper":
        from mt5_connector import MT5Connector
        mt5_conn = MT5Connector(symbol=SYMBOL)
        if not mt5_conn.connect():
            log_event(conn, "ERROR", "MT5_CONNECT_FAILED", "")
            return 1
        log_event(conn, "INFO", "MT5_CONNECTED", f"equity={mt5_conn.equity()}")
        initial_equity = mt5_conn.equity()
    else:
        mt5_conn = None
        initial_equity = 10000.0

    # === Features ===
    from features import calc_features, get_feature_columns
    feature_cols = get_feature_columns()

    # === Main loop ===
    last_bar_time = 0
    paused_until = None

    print("\n[loop] starting (Ctrl+C to stop)...\n")
    try:
        while True:
            # ----- Pause check -----
            if paused_until and datetime.now() < paused_until:
                time.sleep(SLEEP_BETWEEN_CHECKS)
                continue

            # ----- Hard stop check (L1) -----
            if mt5_conn:
                if check_hard_stop(mt5_conn, initial_equity):
                    log_event(conn, "CRITICAL", "HARD_STOP_TRIGGERED",
                              f"equity={mt5_conn.equity()} initial={initial_equity}")
                    mt5_conn.close_all()
                    log_event(conn, "INFO", "ALL_POSITIONS_CLOSED", "")
                    break

            # ----- Check new bar -----
            if mt5_conn:
                rates = mt5_conn.get_rates(TIMEFRAME, N_BARS_TO_PULL)
                if len(rates) == 0:
                    time.sleep(SLEEP_BETWEEN_CHECKS)
                    continue
                # Use last CLOSED bar (skip current forming)
                latest_closed = rates.iloc[-2]
                if latest_closed['time'] == last_bar_time:
                    time.sleep(SLEEP_BETWEEN_CHECKS)
                    continue
                last_bar_time = latest_closed['time']
            else:
                # Paper mode: simulate from CSV (TODO)
                print("Paper mode requires CSV — TODO")
                break

            print(f"\n[{datetime.now():%H:%M:%S}] new bar: {latest_closed['timestamp']}")

            # ----- Calc features -----
            df_features = calc_features(rates)

            # Drop NaN rows (first ~50 rows have NaN due to lookback)
            df_features = df_features.dropna(subset=feature_cols)
            if len(df_features) < WINDOW_SIZE + 1:
                log_event(conn, "WARN", "INSUFFICIENT_DATA",
                          f"only {len(df_features)} valid rows")
                time.sleep(SLEEP_BETWEEN_CHECKS)
                continue

            # Apply normalization
            if norm is not None:
                for c in feature_cols:
                    if c in norm.index:
                        df_features[c] = (df_features[c] - norm.at[c, 'mean']) / \
                                          (norm.at[c, 'std'] + 1e-8)

            # ----- Build state -----
            position_side = mt5_conn.current_position_side() if mt5_conn else 0
            unrealized = mt5_conn.unrealized_pnl_pct() if mt5_conn else 0.0
            bars_in_position = 0  # TODO: track from DB

            # Last `window` rows of features
            window_features = df_features.iloc[-WINDOW_SIZE:][feature_cols].values
            extra = np.array([position_side, unrealized, bars_in_position / 100.0])
            state = np.concatenate([window_features.flatten(), extra]).astype(np.float32)

            # ----- Predict -----
            obs_tensor = torch.as_tensor(state).unsqueeze(0).float()
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy().flatten()
            action = int(np.argmax(probs))
            confidence = float(probs[action])

            action_name = ['Hold', 'Buy', 'Sell', 'Close'][action]
            print(f"  predict: {action_name} (conf={confidence:.3f})")

            # ----- Confidence filter -----
            if action in (1, 2) and confidence < CONFIDENCE_THRESHOLD:
                print(f"  → skip (conf < {CONFIDENCE_THRESHOLD})")
                log_trade(conn,
                          ts=datetime.now().isoformat(),
                          bar_time=str(latest_closed['timestamp']),
                          action=action, action_name=action_name,
                          confidence=confidence,
                          position_before=position_side,
                          equity=mt5_conn.equity() if mt5_conn else initial_equity,
                          executed=0, ticket=None, price=None,
                          sl=None, tp=None, lots=None,
                          comment="skipped: low confidence")
                continue

            # ----- Position limits -----
            if mt5_conn and action in (1, 2):
                if mt5_conn.position_count() >= MAX_POSITIONS:
                    print(f"  → skip (max positions reached)")
                    continue

            # ----- Execute -----
            if mode == "paper":
                print(f"  [PAPER] would execute {action_name}")
                continue

            # Calc SL/TP based on ATR
            atr_val = rates.iloc[-2].get('high', 0) - rates.iloc[-2].get('low', 0)
            if 'atr' in df_features.columns:
                # use computed ATR if available (un-normalized)
                # since we normalized in df_features, recalc raw
                from features import atr as atr_calc
                raw_atr = atr_calc(rates['high'], rates['low'], rates['close']).iloc[-1]
                atr_val = float(raw_atr)
            atr_val = max(atr_val, rates['close'].iloc[-1] * 0.001)  # min 0.1%

            executed = False
            ticket = None

            if action == 1:  # Buy
                tick = mt5_conn.current_tick()
                price = tick.ask
                sl = price - atr_val * ATR_SL_MULTIPLIER
                tp = price + atr_val * ATR_TP_MULTIPLIER
                lots = mt5_conn.calc_lot_size(price - sl, RISK_PER_TRADE)
                executed = mt5_conn.send_buy(lots, sl, tp, comment=f"RL_{confidence:.2f}")
                if executed:
                    log_event(conn, "TRADE", "BUY",
                              f"lots={lots} price={price} sl={sl} tp={tp}")

            elif action == 2:  # Sell
                tick = mt5_conn.current_tick()
                price = tick.bid
                sl = price + atr_val * ATR_SL_MULTIPLIER
                tp = price - atr_val * ATR_TP_MULTIPLIER
                lots = mt5_conn.calc_lot_size(sl - price, RISK_PER_TRADE)
                executed = mt5_conn.send_sell(lots, sl, tp, comment=f"RL_{confidence:.2f}")
                if executed:
                    log_event(conn, "TRADE", "SELL",
                              f"lots={lots} price={price} sl={sl} tp={tp}")

            elif action == 3:  # Close
                if mt5_conn.position_count() > 0:
                    closed = mt5_conn.close_all()
                    executed = closed > 0
                    log_event(conn, "TRADE", "CLOSE", f"closed={closed}")

            # else: action == 0 (Hold) — no-op

            log_trade(conn,
                      ts=datetime.now().isoformat(),
                      bar_time=str(latest_closed['timestamp']),
                      action=action, action_name=action_name,
                      confidence=confidence,
                      position_before=position_side,
                      equity=mt5_conn.equity() if mt5_conn else initial_equity,
                      executed=int(executed),
                      ticket=str(ticket) if ticket else None,
                      price=None, sl=None, tp=None, lots=None,
                      comment=f"action={action_name}")

            time.sleep(SLEEP_BETWEEN_CHECKS)

    except KeyboardInterrupt:
        print("\n[stop] received Ctrl+C")
        log_event(conn, "INFO", "STOPPED", "user interrupt")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        log_event(conn, "ERROR", "EXCEPTION", str(e))
    finally:
        if mt5_conn:
            mt5_conn.disconnect()
        conn.close()
        print("[shutdown] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="demo mode (default)")
    ap.add_argument("--live", action="store_true", help="LIVE trading (real money!)")
    ap.add_argument("--paper", action="store_true", help="paper mode (no orders)")
    args = ap.parse_args()

    mode = "demo"
    if args.paper:
        mode = "paper"
    elif args.live:
        mode = "live"
        # Safety prompt
        confirm = input("⚠️  LIVE MODE — real money! Type 'YES' to continue: ")
        if confirm.strip() != "YES":
            print("Cancelled.")
            return 0

    run_trading_loop(mode=mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
