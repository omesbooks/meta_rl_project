"""
mt5_connector.py — MT5 Operations Wrapper
==========================================
Encapsulate MT5 API calls + error handling

ติดตั้ง:
    pip install MetaTrader5

Usage:
    from mt5_connector import MT5Connector

    mt5 = MT5Connector(symbol="XAUUSDm")
    mt5.connect()
    bars = mt5.get_rates(timeframe='H1', n_bars=250)
    mt5.send_buy(lots=0.01, sl=2050.0, tp=2080.0)
    mt5.close_all()
"""
import time
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 not installed. Run: pip install MetaTrader5")


class MT5Connector:
    """MT5 connection + trading operations wrapper"""

    TIMEFRAMES = {
        'M1':  1,    'M5':  5,    'M15': 15,   'M30': 30,
        'H1':  16385,  # mt5.TIMEFRAME_H1
        'H4':  16388,  # mt5.TIMEFRAME_H4
        'D1':  16408,  # mt5.TIMEFRAME_D1
        'W1':  32769,  # mt5.TIMEFRAME_W1
    }

    def __init__(self, symbol: str = "XAUUSDm", magic: int = 20260424,
                 deviation: int = 20):
        self.symbol = symbol
        self.magic = magic
        self.deviation = deviation
        self.connected = False

    # ---------------------------------------------------------
    # Connection
    # ---------------------------------------------------------
    def connect(self) -> bool:
        """Initialize MT5 connection"""
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 module not installed")

        if not mt5.initialize():
            print(f"[MT5] initialize() failed: {mt5.last_error()}")
            return False

        # Check symbol available
        info = mt5.symbol_info(self.symbol)
        if info is None:
            print(f"[MT5] symbol {self.symbol} not found")
            return False

        # Enable in MarketWatch if needed
        if not info.visible:
            mt5.symbol_select(self.symbol, True)

        self.connected = True
        account = mt5.account_info()
        print(f"[MT5] Connected. Account: {account.login} ({account.server})")
        print(f"[MT5] Balance: {account.balance} {account.currency}")
        return True

    def disconnect(self):
        """Shutdown MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    def get_rates(self, timeframe: str = 'H1', n_bars: int = 250) -> pd.DataFrame:
        """Pull recent bars"""
        tf = self.TIMEFRAMES[timeframe]
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, n_bars)
        if rates is None:
            print(f"[MT5] copy_rates_from_pos failed: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_rates_range(self, timeframe: str, start: datetime,
                        end: datetime) -> pd.DataFrame:
        """Pull historical range"""
        tf = self.TIMEFRAMES[timeframe]
        rates = mt5.copy_rates_range(self.symbol, tf, start, end)
        if rates is None:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        return df

    def current_tick(self):
        """Get current bid/ask"""
        return mt5.symbol_info_tick(self.symbol)

    def is_new_bar(self, last_time: int, timeframe: str = 'H1') -> bool:
        """Check if new bar formed since last_time (unix timestamp)"""
        rates = self.get_rates(timeframe, n_bars=2)
        if len(rates) < 2:
            return False
        return rates.iloc[-1]['time'] > last_time

    # ---------------------------------------------------------
    # Orders
    # ---------------------------------------------------------
    def send_buy(self, lots: float, sl: float = 0.0, tp: float = 0.0,
                 comment: str = "RL"):
        """Send market buy order"""
        tick = mt5.symbol_info_tick(self.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lots),
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "sl": float(sl) if sl > 0 else 0.0,
            "tp": float(tp) if tp > 0 else 0.0,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return self._handle_result(result, "BUY")

    def send_sell(self, lots: float, sl: float = 0.0, tp: float = 0.0,
                  comment: str = "RL"):
        """Send market sell order"""
        tick = mt5.symbol_info_tick(self.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lots),
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "sl": float(sl) if sl > 0 else 0.0,
            "tp": float(tp) if tp > 0 else 0.0,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return self._handle_result(result, "SELL")

    def close_all(self) -> int:
        """Close all positions for this magic+symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return 0
        closed = 0
        for pos in positions:
            if pos.magic != self.magic:
                continue
            if self._close_position(pos):
                closed += 1
        return closed

    def _close_position(self, pos) -> bool:
        """Close single position"""
        tick = mt5.symbol_info_tick(self.symbol)
        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == 0 else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    def _handle_result(self, result, op: str):
        if result is None:
            print(f"[MT5] {op} failed (no result)")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] {op} failed: retcode={result.retcode} {result.comment}")
            return False
        print(f"[MT5] {op} ok ticket={result.order} price={result.price}")
        return True

    # ---------------------------------------------------------
    # Account / Positions
    # ---------------------------------------------------------
    def get_account(self):
        """Account info"""
        return mt5.account_info()

    def equity(self) -> float:
        info = self.get_account()
        return info.equity if info else 0.0

    def balance(self) -> float:
        info = self.get_account()
        return info.balance if info else 0.0

    def get_positions(self):
        """Open positions for this magic+symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return []
        return [p for p in positions if p.magic == self.magic]

    def position_count(self) -> int:
        return len(self.get_positions())

    def current_position_side(self) -> int:
        """Returns +1 (long), -1 (short), 0 (flat)"""
        positions = self.get_positions()
        if not positions:
            return 0
        # If multiple, return net (+1 long, -1 short, 0 if balanced)
        net = sum(1 if p.type == 0 else -1 for p in positions)
        return 1 if net > 0 else (-1 if net < 0 else 0)

    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as % of equity"""
        positions = self.get_positions()
        if not positions:
            return 0.0
        total_profit = sum(p.profit for p in positions)
        return total_profit / max(self.equity(), 1.0)

    # ---------------------------------------------------------
    # Risk helpers
    # ---------------------------------------------------------
    def calc_lot_size(self, sl_distance_price: float,
                      risk_pct: float = 0.01) -> float:
        """Calculate lot size based on % risk per trade"""
        info = mt5.symbol_info(self.symbol)
        if info is None or sl_distance_price <= 0:
            return 0.01

        balance = self.balance()
        risk_amount = balance * risk_pct
        tick_value = info.trade_tick_value
        tick_size = info.trade_tick_size

        if tick_size == 0 or tick_value == 0:
            return 0.01

        ticks_at_risk = sl_distance_price / tick_size
        risk_per_lot = ticks_at_risk * tick_value
        lots = risk_amount / max(risk_per_lot, 1e-9)

        # Round to step + clip to limits
        lots = max(info.volume_min, min(info.volume_max, lots))
        step = info.volume_step
        lots = round(lots / step) * step
        return round(lots, 2)


if __name__ == "__main__":
    # Quick test
    if not MT5_AVAILABLE:
        print("Install MetaTrader5 first")
    else:
        c = MT5Connector(symbol="XAUUSDm")
        if c.connect():
            print(f"Equity: {c.equity()}")
            df = c.get_rates('H1', 5)
            print(df.tail())
            c.disconnect()
