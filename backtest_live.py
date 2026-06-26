"""
backtest_live.py — Backtest using LIVE_TRADER logic on historical data
========================================================================
Simulate live_trader.py loop ทีละ bar — แต่ใช้ historical CSV แทน MT5

ทำไมต้องมี?
  - rl_backtest.py    → ใช้ TradingEnv (ตอน train)
  - live_trader.py    → ใช้ production logic (filter + risk + ATR SL/TP)
  - backtest_live.py  → backtest ด้วย logic ของ live_trader → ผลตรง 100%

ความต่างจาก rl_backtest.py:
  ✓ Confidence filter (skip ถ้า conf < threshold)
  ✓ Position size calculation (% risk per trade)
  ✓ ATR-based SL/TP (dynamic)
  ✓ Max positions limit
  ✓ Hard stop ถ้า equity ตกถึง threshold
  ✓ Force close หลัง max_hold_bars

Usage:
    # Backtest บน test set (2021-2026)
    python backtest_live.py rl_prod_v1 test_2021_2026.csv

    # ปรับ confidence threshold
    python backtest_live.py rl_prod_v1 test.csv --conf 0.90

    # แสดงรายละเอียดทุก trade
    python backtest_live.py rl_prod_v1 test.csv --verbose
"""
import sys, io, argparse, sqlite3, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from artifact_paths import (
    backtest_meta_path,
    backtests_dir,
    equity_path as artifact_equity_path,
    find_model_path,
    find_norm_path,
    trades_path as artifact_trades_path,
)

TRADE_COLUMNS = [
    "side", "entry", "sl", "tp", "lots", "open_idx", "open_time",
    "exit", "exit_time", "close_idx", "bars_held", "pnl_pct",
    "pnl_dollars", "reason",
]


def _period(df):
    if "timestamp" not in df.columns or len(df) == 0:
        return None
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return None
    return {"start": ts.iloc[0].isoformat(), "end": ts.iloc[-1].isoformat()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================
# Simulated Account (แทน MT5)
# =============================================================
class SimAccount:
    """จำลอง broker account สำหรับ backtest"""

    def __init__(self, initial_balance=10000.0, spread_pct=0.0002, commission=0.0001):
        self.initial = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.peak = initial_balance
        self.spread = spread_pct
        self.commission = commission
        self.positions = []  # list of dicts
        self.trade_history = []
        self.equity_curve = [initial_balance]

    def get_position_side(self):
        if not self.positions:
            return 0
        return 1 if self.positions[0]['side'] == 'long' else -1

    def position_count(self):
        return len(self.positions)

    def unrealized_pnl_pct(self, current_price):
        if not self.positions:
            return 0.0
        total = 0.0
        for p in self.positions:
            sign = 1 if p['side'] == 'long' else -1
            change = (current_price - p['entry']) / p['entry'] * sign
            total += change * p['lots']
        return total / max(self.equity, 1.0)

    def calc_lot_size(self, sl_distance, risk_pct=0.01):
        """Risk-based lot sizing"""
        if sl_distance <= 0:
            return 0.01
        risk_amount = self.balance * risk_pct
        # Simplified: lot = risk_amount / sl_distance (assumes $1/pip mini-lot proxy)
        # In live, this uses tick value from MT5. Here we approximate.
        lots = risk_amount / max(sl_distance, 0.0001) / 100
        return max(0.01, min(1.0, round(lots, 2)))

    def open_position(self, side: str, price: float, lots: float,
                       sl: float, tp: float, bar_idx: int, bar_time):
        # Apply spread (cost on entry)
        if side == 'long':
            price = price * (1 + self.spread)
        else:
            price = price * (1 - self.spread)

        self.positions.append({
            'side': side,
            'entry': price,
            'sl': sl,
            'tp': tp,
            'lots': lots,
            'open_idx': bar_idx,
            'open_time': bar_time,
        })

    def close_position(self, idx: int, exit_price: float, bar_idx: int,
                       bar_time, reason: str = "signal"):
        """Close position by index in self.positions"""
        if idx >= len(self.positions):
            return None
        p = self.positions[idx]
        sign = 1 if p['side'] == 'long' else -1

        # Apply spread on exit (opposite of entry)
        if p['side'] == 'long':
            exit_price = exit_price * (1 - self.spread)
        else:
            exit_price = exit_price * (1 + self.spread)

        pnl_pct = (exit_price - p['entry']) / p['entry'] * sign
        net_pnl_pct = pnl_pct - self.commission
        pnl_dollars = self.balance * net_pnl_pct * p['lots']
        self.balance += pnl_dollars

        trade = {
            **p,
            'exit': exit_price,
            'exit_time': bar_time,
            'close_idx': bar_idx,
            'bars_held': bar_idx - p['open_idx'],
            'pnl_pct': net_pnl_pct,
            'pnl_dollars': pnl_dollars,
            'reason': reason,
        }
        self.trade_history.append(trade)
        del self.positions[idx]
        return trade

    def close_all(self, exit_price, bar_idx, bar_time, reason="signal"):
        closed = 0
        while self.positions:
            self.close_position(0, exit_price, bar_idx, bar_time, reason)
            closed += 1
        return closed

    def update_equity(self, current_price):
        """Mark-to-market update"""
        unrealized = 0.0
        for p in self.positions:
            sign = 1 if p['side'] == 'long' else -1
            change = (current_price - p['entry']) / p['entry'] * sign
            unrealized += self.balance * change * p['lots']
        self.equity = self.balance + unrealized
        self.peak = max(self.peak, self.equity)
        self.equity_curve.append(self.equity)

    def drawdown_pct(self):
        return (self.equity - self.peak) / self.peak

    def check_sl_tp(self, bar_high: float, bar_low: float, bar_idx: int, bar_time):
        """Check if any position hit SL/TP within this bar"""
        for i in range(len(self.positions) - 1, -1, -1):
            p = self.positions[i]
            if p['side'] == 'long':
                # Long: SL below, TP above
                if p['sl'] > 0 and bar_low <= p['sl']:
                    self.close_position(i, p['sl'], bar_idx, bar_time, "SL")
                elif p['tp'] > 0 and bar_high >= p['tp']:
                    self.close_position(i, p['tp'], bar_idx, bar_time, "TP")
            else:
                # Short: SL above, TP below
                if p['sl'] > 0 and bar_high >= p['sl']:
                    self.close_position(i, p['sl'], bar_idx, bar_time, "SL")
                elif p['tp'] > 0 and bar_low <= p['tp']:
                    self.close_position(i, p['tp'], bar_idx, bar_time, "TP")


# =============================================================
# Backtest Engine
# =============================================================
def run_backtest_live(args):
    print("=" * 60)
    print("  BACKTEST WITH LIVE_TRADER LOGIC")
    print("=" * 60)
    print(f"  Model       : {args.model}.zip")
    print(f"  Data        : {args.csv}")
    print(f"  Confidence  : >= {args.conf}")
    print(f"  Risk/trade  : {args.risk:.0%}")
    print(f"  Max positions: {args.max_positions}")
    print(f"  Max hold    : {args.max_hold} bars")
    print(f"  Hard DD     : {args.hard_dd:.0%}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(args.csv)
    leaky = [c for c in df.columns
             if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))]
    if leaky:
        print(f"\n[data] dropping leaky cols: {leaky}")
        df = df.drop(columns=leaky)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
    data_rows_before_start = len(df)
    data_period_before_start = _period(df)

    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    print(f"[data] rows: {len(df):,} | features: {len(feature_cols)}")

    # Apply normalization
    norm_path = find_norm_path(args.model)
    if norm_path and norm_path.exists():
        norm = pd.read_csv(norm_path, index_col=0)
        for c in feature_cols:
            if c in norm.index:
                df[c] = (df[c] - norm.at[c, 'mean']) / (norm.at[c, 'std'] + 1e-8)
        print(f"[norm] applied from {norm_path}")
    else:
        print(f"[norm] not found for {args.model}; using raw feature scale")
    df = df.fillna(0).reset_index(drop=True)

    # Apply start fraction (use last X% as test)
    start_idx = 0
    if args.start > 0:
        start_idx = int(len(df) * args.start)
        df = df.iloc[start_idx:].reset_index(drop=True)
        print(f"[split] using rows {start_idx}.. ({len(df):,} rows)")
    backtest_period = _period(df)

    out_dir = backtests_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = backtest_meta_path(args.model)
    meta = {
        "model": args.model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_csv": args.csv,
        "data_rows_before_start": data_rows_before_start,
        "data_period_before_start": data_period_before_start,
        "start_fraction": args.start,
        "start_index": start_idx,
        "backtest_rows": len(df),
        "backtest_period": backtest_period,
        "feature_count": len(feature_cols),
        "model_path": None,
        "norm_path": str(norm_path) if norm_path else "",
        "artifacts_dir": str(out_dir),
        "artifacts": {
            "trades": str(artifact_trades_path(args.model)),
            "equity": str(artifact_equity_path(args.model)),
            "backtest_meta": str(meta_path),
        },
        "settings": {
            "conf": args.conf,
            "risk": args.risk,
            "max_positions": args.max_positions,
            "max_hold": args.max_hold,
            "hard_dd": args.hard_dd,
            "atr_sl": args.atr_sl,
            "atr_tp": args.atr_tp,
            "mode": args.mode,
            "balance": args.balance,
            "spread": args.spread,
            "commission": args.commission,
            "window": args.window,
        },
    }

    # Load model
    from stable_baselines3 import PPO
    import torch
    model_path = find_model_path(args.model, "final")
    if model_path is None:
        print(f"\nERROR: model not found: {args.model}")
        _write_json(meta_path, {**meta, "status": "failed", "error": "model not found"})
        return 1
    meta["model_path"] = str(model_path)
    print(f"\n[load] {model_path}")
    model = PPO.load(str(model_path))

    # Setup account
    account = SimAccount(initial_balance=args.balance,
                          spread_pct=args.spread,
                          commission=args.commission)

    # Stats counters
    n_signals = {0: 0, 1: 0, 2: 0, 3: 0}
    n_executed = {1: 0, 2: 0, 3: 0}
    n_skipped_conf = 0
    n_skipped_pos = 0
    n_force_close = 0
    n_hard_stop = 0

    # Auto-detect window from the model's observation space:
    #   obs_dim = window * n_features + 3  ->  window = (obs_dim - 3) / n_features
    # No need to pass --window; the model knows it. --window only overrides.
    obs_dim = int(model.observation_space.shape[0])
    n_feat = len(feature_cols)
    detected_window = (obs_dim - 3) // n_feat if n_feat > 0 else 0

    if detected_window > 0 and (obs_dim - 3) % n_feat == 0:
        if args.window > 0 and args.window != detected_window:
            print(f"[window] --window {args.window} != model's {detected_window}; "
                  f"using model's {detected_window}")
        else:
            print(f"[window] auto-detected {detected_window} "
                  f"(obs_dim={obs_dim}, features={n_feat})")
        warmup = detected_window
    else:
        warmup = args.window if args.window > 0 else 10
        print(f"[window] could not auto-detect (obs_dim={obs_dim}, "
              f"features={n_feat}); using {warmup}. "
              f"Check that backtest features match training.")

    # Ensure we have enough warmup
    if len(df) <= warmup + 1:
        print("ERROR: not enough data after warmup")
        return 1

    # ==========================================================
    # Main backtest loop
    # ==========================================================
    print(f"\n[backtest] starting on {len(df) - warmup} bars ...")
    paused = False

    for i in range(warmup, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        price = row['close']
        bar_time = row.get('timestamp', i)

        # Update equity (mark-to-market)
        account.update_equity(price)

        # Check SL/TP for open positions (using bar OHLC)
        account.check_sl_tp(row['high'], row['low'], i, bar_time)

        # ---- Hard stop check (L1) ----
        if account.drawdown_pct() <= -args.hard_dd:
            if not paused:
                if args.verbose:
                    print(f"  [{bar_time}] 🚨 HARD STOP triggered (DD={account.drawdown_pct():.2%})")
                account.close_all(price, i, bar_time, "hard_stop")
                n_hard_stop += 1
                # ⭐ Reset peak so agent can resume after hard stop
                # (Without this, drawdown stays negative forever → paused forever)
                account.peak = account.equity
                paused = True
            continue
        else:
            paused = False

        # ---- Force close after max_hold ----
        for j in range(len(account.positions) - 1, -1, -1):
            p = account.positions[j]
            if i - p['open_idx'] >= args.max_hold:
                account.close_position(j, price, i, bar_time, "max_hold")
                n_force_close += 1

        # ---- Predict ----
        # Build state: window of features + position_info
        if i < warmup:
            continue
        window = df.iloc[i - warmup + 1:i + 1][feature_cols].values
        pos_side = account.get_position_side()
        unrealized = account.unrealized_pnl_pct(price)
        bars_in_pos = 0
        if account.positions:
            bars_in_pos = i - account.positions[0]['open_idx']
        extra = np.array([pos_side, unrealized, bars_in_pos / 100.0])
        state = np.concatenate([window.flatten(), extra]).astype(np.float32)

        obs_tensor = torch.as_tensor(state).unsqueeze(0).float()
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.numpy().flatten()
        action = int(np.argmax(probs))
        confidence = float(probs[action])
        n_signals[action] += 1

        # ---- Apply confidence filter ----
        if action in (1, 2) and confidence < args.conf:
            n_skipped_conf += 1
            continue

        # ---- Position limit ----
        if action in (1, 2) and account.position_count() >= args.max_positions:
            n_skipped_pos += 1
            continue

        # ---- Calc ATR for SL/TP ----
        # Use raw ATR if available, else compute from recent prices
        # Note: features are normalized — need raw ATR
        recent_high = df.iloc[max(0, i - 14):i + 1]['high']
        recent_low = df.iloc[max(0, i - 14):i + 1]['low']
        atr_proxy = (recent_high - recent_low).mean()
        atr_proxy = max(atr_proxy, price * 0.001)  # min 0.1%

        # ---- Execute ----
        # Pure agent mode: disable SL/TP (set to 0 = no level)
        sltp_enabled = args.mode == "agent_sltp" and args.atr_sl > 0

        if action == 1:  # Buy
            if sltp_enabled:
                sl = price - atr_proxy * args.atr_sl
                tp = price + atr_proxy * args.atr_tp
                lots = account.calc_lot_size(price - sl, args.risk)
            else:
                sl = 0  # no SL
                tp = 0  # no TP
                # In pure mode, use fixed lot proportional to risk%
                lots = max(0.01, args.risk * 100)
            account.open_position('long', price, lots, sl, tp, i, bar_time)
            n_executed[1] += 1
            if args.verbose:
                print(f"  [{bar_time}] BUY  conf={confidence:.2f} lots={lots} "
                      f"price={price:.2f} sl={sl:.2f} tp={tp:.2f}")

        elif action == 2:  # Sell
            if sltp_enabled:
                sl = price + atr_proxy * args.atr_sl
                tp = price - atr_proxy * args.atr_tp
                lots = account.calc_lot_size(sl - price, args.risk)
            else:
                sl = 0
                tp = 0
                lots = max(0.01, args.risk * 100)
            account.open_position('short', price, lots, sl, tp, i, bar_time)
            n_executed[2] += 1
            if args.verbose:
                print(f"  [{bar_time}] SELL conf={confidence:.2f} lots={lots} "
                      f"price={price:.2f} sl={sl:.2f} tp={tp:.2f}")

        elif action == 3:  # Close
            if account.position_count() > 0:
                closed = account.close_all(price, i, bar_time, "signal")
                n_executed[3] += closed
                if args.verbose:
                    print(f"  [{bar_time}] CLOSE conf={confidence:.2f} closed={closed}")

    # Final close any remaining positions
    if account.positions:
        final_price = df.iloc[-1]['close']
        account.close_all(final_price, len(df) - 1, df.iloc[-1].get('timestamp'), "end")

    # ==========================================================
    # Stats
    # ==========================================================
    trades = account.trade_history
    if not trades:
        print("\n⚠️  No trades executed!")
        meta["status"] = "complete"
        meta["result"] = {
            "total_trades": 0,
            "final_balance": account.balance,
            "return_pct": account.balance / args.balance - 1,
        }
        if args.save:
            out_csv = artifact_trades_path(args.model)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=TRADE_COLUMNS).to_csv(out_csv, index=False)
            print(f"[save] empty trades -> {out_csv}")
            _write_json(meta_path, meta)
            print(f"[meta] -> {meta_path}")
        return 0

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    pnls = [t['pnl_pct'] for t in trades]
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    total_pnl = sum(t['pnl_dollars'] for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    sum_wins = sum(t['pnl_pct'] for t in wins)
    sum_losses = abs(sum(t['pnl_pct'] for t in losses))
    pf = sum_wins / sum_losses if sum_losses > 0 else float('inf')

    eq = np.array(account.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Sharpe
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9) * np.sqrt(252 * 24)  # H1
    else:
        sharpe = 0

    print(f"  Total trades       : {len(trades):,}")
    print(f"    Long  trades     : {sum(1 for t in trades if t['side']=='long'):,}")
    print(f"    Short trades     : {sum(1 for t in trades if t['side']=='short'):,}")
    print(f"  Win rate           : {win_rate:.2%}")
    print(f"  Profit factor      : {pf:.2f}")
    print(f"  Avg win            : {avg_win:+.4%}")
    print(f"  Avg loss           : {avg_loss:+.4%}")
    print(f"  Total P&L          : ${total_pnl:+,.2f}")
    print(f"  Final balance      : ${account.balance:,.2f} (start ${args.balance:,.0f})")
    print(f"  Return             : {(account.balance / args.balance - 1):+.2%}")
    print(f"  Max drawdown       : {max_dd:.2%}")
    print(f"  Sharpe (annualized): {sharpe:.2f}")

    # Exit reason breakdown
    print(f"\n  Exit reasons:")
    from collections import Counter
    reasons = Counter(t['reason'] for t in trades)
    for r, n in reasons.most_common():
        pct = n / len(trades) * 100
        print(f"    {r:<10}: {n:>5,} ({pct:5.1f}%)")

    # Signal breakdown
    print(f"\n  Signal stats:")
    names = {0: 'Hold', 1: 'Buy', 2: 'Sell', 3: 'Close'}
    total_signals = sum(n_signals.values())
    for a in [0, 1, 2, 3]:
        pct = n_signals[a] / total_signals * 100 if total_signals else 0
        executed = n_executed.get(a, 0) if a > 0 else '-'
        print(f"    {names[a]:<6}: {n_signals[a]:>6,} signals "
              f"({pct:5.1f}%) | executed: {executed}")

    print(f"\n  Filter stats:")
    print(f"    Skipped (low confidence): {n_skipped_conf:,}")
    print(f"    Skipped (max positions) : {n_skipped_pos:,}")
    print(f"    Force closed (max hold) : {n_force_close:,}")
    print(f"    Hard stops triggered    : {n_hard_stop:,}")
    print("=" * 60)

    # Save trades + equity curve
    meta["status"] = "complete"
    meta["result"] = {
        "total_trades": len(trades),
        "long_trades": sum(1 for t in trades if t['side'] == 'long'),
        "short_trades": sum(1 for t in trades if t['side'] == 'short'),
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
        "final_balance": account.balance,
        "return_pct": account.balance / args.balance - 1,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "exit_reasons": dict(reasons),
        "signals": n_signals,
        "executed": n_executed,
        "skipped_low_confidence": n_skipped_conf,
        "skipped_max_positions": n_skipped_pos,
        "force_closed_max_hold": n_force_close,
        "hard_stops": n_hard_stop,
    }
    if args.save:
        out_csv = artifact_trades_path(args.model)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trades).to_csv(out_csv, index=False)
        print(f"\n[save] -> {out_csv}")

        # Chart
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
            axes[0].plot(eq, color='#2563eb', linewidth=1.2)
            axes[0].axhline(args.balance, color='gray', linestyle='--', alpha=0.5)
            axes[0].fill_between(range(len(eq)), args.balance, eq,
                                  where=(eq >= args.balance), color='#10b981', alpha=0.15)
            axes[0].fill_between(range(len(eq)), args.balance, eq,
                                  where=(eq < args.balance), color='#ef4444', alpha=0.15)
            axes[0].set_title(f"Live Backtest Equity — {args.model} (PF={pf:.2f}, WR={win_rate:.1%})")
            axes[0].set_ylabel("Equity ($)")
            axes[0].grid(True, alpha=0.3)

            axes[1].fill_between(range(len(dd)), 0, dd * 100, color='#ef4444', alpha=0.5)
            axes[1].set_ylabel("DD (%)")
            axes[1].set_xlabel("Bar")
            axes[1].grid(True, alpha=0.3)

            chart_path = artifact_equity_path(args.model)
            plt.tight_layout()
            plt.savefig(chart_path, dpi=110, bbox_inches='tight')
            print(f"[save] -> {chart_path}")
        except Exception as e:
            print(f"[chart] skipped: {e}")
            meta["chart_error"] = str(e)
        _write_json(meta_path, meta)
        print(f"[meta] -> {meta_path}")

    return 0


# =============================================================
# CLI
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model name (without .zip)")
    ap.add_argument("csv", help="historical data CSV")

    # Filter & risk (เหมือน live_trader.py CONFIG)
    ap.add_argument("--conf", type=float, default=0.85,
                    help="confidence threshold (default 0.85)")
    ap.add_argument("--risk", type=float, default=0.01,
                    help="risk per trade (default 1%%)")
    ap.add_argument("--max_positions", type=int, default=3)
    ap.add_argument("--max_hold", type=int, default=30,
                    help="force close after N bars")
    ap.add_argument("--hard_dd", type=float, default=0.15,
                    help="hard stop drawdown (default 15%%)")
    ap.add_argument("--atr_sl", type=float, default=1.5,
                    help="SL = ATR × multiplier (set 0 to disable)")
    ap.add_argument("--atr_tp", type=float, default=3.0,
                    help="TP = ATR × multiplier (set 0 to disable)")
    ap.add_argument("--mode", default="agent_sltp",
                    choices=["pure_agent", "agent_sltp"],
                    help="pure_agent = no SL/TP (matches training) · agent_sltp = with SL/TP")

    # Account
    ap.add_argument("--balance", type=float, default=10000,
                    help="initial balance")
    ap.add_argument("--spread", type=float, default=0.0002,
                    help="spread %% (0.02%% default)")
    ap.add_argument("--commission", type=float, default=0.0001)

    # Data split
    ap.add_argument("--start", type=float, default=0.0,
                    help="start fraction (0.8 = use last 20%%)")
    ap.add_argument("--window", type=int, default=0,
                    help="state window size (0 = auto-detect from model; optional override)")

    # Output
    ap.add_argument("--verbose", action="store_true",
                    help="print every trade")
    ap.add_argument("--save", action="store_true", default=True,
                    help="save trades CSV + equity chart")

    args = ap.parse_args()
    return run_backtest_live(args)


if __name__ == "__main__":
    sys.exit(main())
