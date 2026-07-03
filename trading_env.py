"""
TradingEnv — Gymnasium environment สำหรับ forex/stock trading

State (observation):
    [feature_window flatten] + [position, unrealized_pnl, bars_in_position]
    ⭐ V3: window includes current bar t (bars [t-W+1..t])
       = ตรงกับ backtest_live.py / live_trader.py

Action (discrete):
    0 = Hold  (ไม่ทำอะไร)
    1 = Buy   (เปิด long ถ้ายังไม่มี position)
    2 = Sell  (เปิด short ถ้ายังไม่มี position)
    3 = Close (ปิด position ปัจจุบัน)

Reward (Balanced profile, V4 — Honest):
    + Net-PnL × 50 ตอนปิด (real $ dominates reward)
    + +0.01 bonus เฉพาะถ้า net pnl > 0.5% (meaningful win)
    + incremental unrealized × 0.1 (tiny direction hint)
    - give-back penalty -0.001 (peak/2 reached, threshold 0.2%)
    - trade penalty -0.005 ตอนเปิด (heavy: ป้องกัน over-trading)
    - hold idle penalty -0.0001 ต่อ bar (ตอน flat)
    - time-decay -0.0002 หลังถือเกิน 70% max_hold
    → Break-even WR ~53% (ตรงกับ real $ break-even)

Other presets come from reward_profiles.py and change these weights.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from reward_formula import compile_reward_formula, eval_reward_formula
from reward_profiles import get_reward_profile
from action_profiles import action_ids_by_name, get_action_profile


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        window_size: int = 10,
        spread_pct: float = 0.0002,   # 0.02%
        commission: float = 0.0001,   # 0.01%
        max_steps: int = None,
        max_hold_bars: int = 30,      # บังคับปิดถ้าถือเกิน
        reward_mode: str = "realized", # "mtm" (เก่า/buy-hold trap) หรือ "realized" (ดีกว่า)
        reward_profile: str | dict = "balanced",
        reward_formula: str = "",
        action_profile: str | dict = "basic_4",
        action_params: dict | None = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.spread = spread_pct
        self.commission = commission
        self.n_features = len(feature_cols)
        self.max_steps = max_steps if max_steps else len(df) - window_size - 2
        self.max_hold_bars = max_hold_bars
        self.reward_mode = reward_mode
        self.reward_profile_name, self.reward_cfg = get_reward_profile(reward_profile)
        self.reward_formula = (reward_formula or "").strip()
        self._reward_formula_code = (
            compile_reward_formula(self.reward_formula)
            if self.reward_formula else None
        )
        self.action_profile_name, self.action_profile = get_action_profile(action_profile, action_params)
        self.action_params = self.action_profile["params"]
        self.action_ids = action_ids_by_name(self.action_profile)

        # Action profile decides output dimension.
        self.action_space = spaces.Discrete(len(self.action_profile["actions"]))

        # Obs: window flat + [position, unrealized_pnl, bars_in_position]
        obs_dim = self.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Pre-compute feature window matrix for speed
        self._features = self.df[feature_cols].values.astype(np.float32)
        self._closes = self.df["close"].values.astype(np.float32)
        self._highs = self.df["high"].values.astype(np.float32) if "high" in self.df.columns else self._closes
        self._lows = self.df["low"].values.astype(np.float32) if "low" in self.df.columns else self._closes

    # --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ⭐⭐ V5 FIX: Random episode start position ⭐⭐
        # ─────────────────────────────────────────────────────────────
        # Bug ก่อนหน้านี้: self.t = window-1 ทุกครั้ง → episode เริ่ม
        # ตำแหน่งเดิมเสมอ → agent เห็นเฉพาะ bars 0..max_steps ตลอด
        # ผลคือ memorize ส่วนต้นของข้อมูล แทนที่จะ generalize
        #
        # Fix: random start → agent เห็นทุกส่วนของข้อมูล
        # → coverage ครบ + ป้องกัน overfitting อย่างแรง
        # ─────────────────────────────────────────────────────────────
        # Window-1 still applies (window includes current bar t)
        min_start = self.window_size - 1
        # max start: ต้องเหลือพื้นที่ให้ episode รันได้ครบ max_steps
        max_start = len(self.df) - self.max_steps - 2
        if max_start > min_start:
            # Random uniform start within valid range
            self.t = int(self.np_random.integers(min_start, max_start + 1))
        else:
            # Fallback: data ไม่พอ → ใช้ตำแหน่งเริ่มต้น
            self.t = min_start
        self.start_t = self.t
        self.position = 0          # -1 short, 0 flat, +1 long
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.bars_in_position = 0
        self.equity = 1.0          # normalized; multiply by initial capital later
        self.peak_equity = 1.0
        self.trades = []
        self._curve = [self.equity]
        # track unrealized P&L per bar for incremental reward shaping
        self._prev_unrealized = 0.0
        self._peak_unrealized = 0.0   # for "give-back" penalty
        return self._get_obs(), self._info()

    # --------------------------------------------------------------
    def _get_obs(self):
        # ⭐ V3 FIX: window = bars [t-W+1 .. t] (รวม bar ปัจจุบัน)
        # เดิม: [t-W .. t-1] → agent ตัดสินใจด้วย info เก่า 1 bar
        # ใหม่: [t-W+1 .. t] → agent เห็น bar ปัจจุบันด้วย (ตรงกับ live_trader)
        window = self._features[self.t - self.window_size + 1 : self.t + 1].flatten()
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            price = self._closes[self.t]
            unrealized = (price - self.entry_price) / self.entry_price * self.position
        extra = np.array(
            [self.position, unrealized, self.bars_in_position / 100.0],
            dtype=np.float32,
        )
        return np.concatenate([window, extra]).astype(np.float32)

    def _info(self):
        return {
            "equity": self.equity,
            "position": self.position,
            "n_trades": len(self.trades),
        }

    def _atr_proxy(self):
        period = int(self.action_params.get("trail_atr_period", 14))
        period = max(2, period)
        start = max(0, self.t - period + 1)
        ranges = self._highs[start:self.t + 1] - self._lows[start:self.t + 1]
        atr = float(np.mean(np.maximum(ranges, 0.0))) if len(ranges) else 0.0
        price = float(self._closes[self.t])
        return max(atr, price * 0.001)

    def _current_unrealized(self, price):
        if self.position == 0 or self.entry_price <= 0:
            return 0.0
        return (price - self.entry_price) / self.entry_price * self.position

    def _close_current_position(self, price, reason="signal"):
        if self.position == 0:
            return None
        exit_p = price * (1.0 - self.spread) if self.position == 1 else price * (1.0 + self.spread)
        pnl_pct = (exit_p - self.entry_price) / self.entry_price * self.position
        net_pnl = pnl_pct - self.commission
        self.equity *= (1 + net_pnl)
        self.trades.append({
            "entry": self.entry_price,
            "exit": exit_p,
            "side": self.position,
            "bars": self.bars_in_position,
            "pnl": net_pnl,
            "reason": reason,
            "stop": self.stop_price,
        })
        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.bars_in_position = 0
        return net_pnl

    def _apply_breakeven_stop(self, price):
        if self.position == 0 or self.entry_price <= 0:
            return False
        if self._current_unrealized(price) < float(self.action_params["breakeven_min_profit"]):
            return False
        old_stop = self.stop_price
        if self.position == 1:
            self.stop_price = max(self.stop_price, self.entry_price)
        else:
            self.stop_price = min(self.stop_price if self.stop_price > 0 else float("inf"), self.entry_price)
        return self.stop_price != old_stop

    def _apply_trailing_stop(self, price):
        if self.position == 0 or self.entry_price <= 0:
            return False
        if self._current_unrealized(price) < float(self.action_params["trail_min_profit"]):
            return False
        atr = self._atr_proxy()
        dist = atr * float(self.action_params["trail_atr_mult"])
        old_stop = self.stop_price
        if self.position == 1:
            candidate = price - dist
            self.stop_price = max(self.stop_price, candidate)
        else:
            candidate = price + dist
            self.stop_price = min(self.stop_price if self.stop_price > 0 else float("inf"), candidate)
        return self.stop_price != old_stop

    def _check_managed_stop(self, price):
        if self.position == 0 or self.stop_price <= 0:
            return None
        if self.position == 1 and price <= self.stop_price:
            return self._close_current_position(self.stop_price, "managed_sl")
        if self.position == -1 and price >= self.stop_price:
            return self._close_current_position(self.stop_price, "managed_sl")
        return None

    # --------------------------------------------------------------
    def step(self, action):
        action = int(action)
        prev_equity = self.equity
        price = self._closes[self.t]
        trade_closed_pnl = None
        just_opened = False   # ⭐ Phase 2: flag สำหรับ trade penalty (แทน cost > 0)
        management_action = False
        stop_adjusted = False
        invalid_action = False
        buy_id = self.action_ids.get("buy")
        sell_id = self.action_ids.get("sell")
        close_id = self.action_ids.get("close")
        breakeven_id = self.action_ids.get("move_sl_breakeven")
        trail_id = self.action_ids.get("trail_sl_atr")

        # ---- execute action ----
        # ⭐⭐ Phase 2: apply spread to entry/exit prices (match backtest_live.py)
        # Long:  buy at higher price (market + spread), sell at lower (market - spread)
        # Short: sell at lower price (market - spread), buy at higher (market + spread)
        if action == buy_id and self.position == 0:      # BUY
            self.position = 1
            self.entry_price = price * (1.0 + self.spread)   # ⭐ shift up
            self.stop_price = 0.0
            self.bars_in_position = 0
            just_opened = True

        elif action == sell_id and self.position == 0:   # SELL
            self.position = -1
            self.entry_price = price * (1.0 - self.spread)   # ⭐ shift down
            self.stop_price = 0.0
            self.bars_in_position = 0
            just_opened = True

        elif action == close_id and self.position != 0:  # CLOSE
            trade_closed_pnl = self._close_current_position(price, "signal")

        elif action == breakeven_id:
            management_action = True
            stop_adjusted = self._apply_breakeven_stop(price)
            invalid_action = not stop_adjusted

        elif action == trail_id:
            management_action = True
            stop_adjusted = self._apply_trailing_stop(price)
            invalid_action = not stop_adjusted

        stop_closed_pnl = self._check_managed_stop(price)
        if stop_closed_pnl is not None:
            trade_closed_pnl = stop_closed_pnl

        # ---- force close if held too long (avoid buy-and-hold trap) ----
        if self.position != 0 and self.bars_in_position >= self.max_hold_bars:
            trade_closed_pnl = self._close_current_position(price, "max_hold")

        # ---- update mark-to-market equity ----
        if self.position != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price * self.position
            current_equity = self.equity * (1 + pnl_pct)
            self.bars_in_position += 1
        else:
            current_equity = self.equity

        # ---- reward shaping ----
        if self.reward_mode == "realized":
            cfg = self.reward_cfg
            reward = 0.0
            current_unrealized = 0.0
            unrealized_delta = 0.0
            giveback = False

            # 1) Tiny unrealized hint (just direction signal; no gaming)
            if self.position != 0 and self.entry_price > 0:
                current_unrealized = (price - self.entry_price) / self.entry_price * self.position
                unrealized_delta = current_unrealized - self._prev_unrealized

                # Give-back penalty.
                self._peak_unrealized = max(self._peak_unrealized, current_unrealized)
                giveback = (
                    self._peak_unrealized > cfg["giveback_trigger"] and
                    current_unrealized < self._peak_unrealized * cfg["giveback_fraction"]
                )

                self._prev_unrealized = current_unrealized
            else:
                self._prev_unrealized = 0.0
                self._peak_unrealized = 0.0
            trade_closed = trade_closed_pnl is not None
            pnl_for_formula = 0.0 if trade_closed_pnl is None else float(trade_closed_pnl)
            is_idle = action == 0 and self.position == 0
            time_decay_active = (
                self.position != 0 and
                self.bars_in_position > self.max_hold_bars * cfg["time_decay_start"]
            )

            if self._reward_formula_code is not None:
                variables = dict(cfg)
                variables.update({
                    "action": int(action),
                    "position": float(self.position),
                    "just_opened": bool(just_opened),
                    "is_idle": bool(is_idle),
                    "in_position": bool(self.position != 0),
                    "trade_closed": bool(trade_closed),
                    "trade_closed_pnl": pnl_for_formula,
                    "current_unrealized": float(current_unrealized),
                    "unrealized_delta": float(unrealized_delta),
                    "peak_unrealized": float(self._peak_unrealized),
                    "giveback": bool(giveback),
                    "bars_in_position": float(self.bars_in_position),
                    "max_hold_bars": float(self.max_hold_bars),
                    "time_decay_active": bool(time_decay_active),
                    "prev_equity": float(prev_equity),
                    "current_equity": float(current_equity),
                    "equity_delta": float((current_equity - prev_equity) / max(prev_equity, 1e-9)),
                    "management_action": bool(management_action),
                    "stop_adjusted": bool(stop_adjusted),
                    "invalid_action": bool(invalid_action),
                    "stop_price": float(self.stop_price),
                })
                reward = eval_reward_formula(self._reward_formula_code, variables)
            else:
                reward += unrealized_delta * cfg["unrealized_scale"]
                if giveback:
                    reward -= cfg["giveback_penalty"]

                # 2) Strong close reward; let real P/L dominate.
                if trade_closed:
                    close_mult = cfg["close_pnl_mult"]
                    if trade_closed_pnl < 0:
                        close_mult *= cfg["loss_penalty_mult"]
                    reward += trade_closed_pnl * close_mult

                    if trade_closed_pnl > cfg["profit_bonus_threshold"]:
                        reward += cfg["profit_bonus"]

                # 3) Opening a trade must be worth the extra cost.
                if just_opened:
                    reward -= cfg["trade_penalty"]

                # 4) Penalize doing nothing while flat.
                if is_idle:
                    reward -= cfg["idle_penalty"]

                # 5) Mild time-decay while in position (encourage timely exits)
                if time_decay_active:
                    reward -= cfg["time_decay_penalty"]

                if management_action:
                    if stop_adjusted:
                        reward -= float(self.action_params["management_action_penalty"])
                    else:
                        reward -= float(self.action_params["invalid_action_penalty"])

            if trade_closed:
                self._prev_unrealized = 0.0
                self._peak_unrealized = 0.0
        else:
            cfg = self.reward_cfg
            # legacy MTM mode; spread is captured via equity drop on entry.
            reward = (current_equity - prev_equity) / max(prev_equity, 1e-9)
            if trade_closed_pnl is not None and trade_closed_pnl > 0:
                reward += cfg["mtm_profit_bonus"]
            if self.position != 0 and self.bars_in_position > cfg["mtm_long_hold_bars"]:
                reward -= cfg["mtm_long_hold_penalty"]
            if management_action:
                reward -= float(
                    self.action_params["management_action_penalty"]
                    if stop_adjusted else self.action_params["invalid_action_penalty"]
                )

        # ---- step forward ----
        self.t += 1
        self._curve.append(current_equity)

        # ---- termination ----
        terminated = False
        truncated = False
        if self.t >= self.start_t + self.max_steps:
            truncated = True
            # auto-close at end — ⭐ Phase 2: apply exit spread for consistency
            if self.position != 0:
                exit_p = price * (1.0 - self.spread) if self.position == 1 else price * (1.0 + self.spread)
                pnl_pct = (exit_p - self.entry_price) / self.entry_price * self.position
                net_pnl = pnl_pct - self.commission
                self.equity *= (1 + net_pnl)
                self.trades.append({
                    "entry": self.entry_price,
                    "exit": exit_p,
                    "side": self.position,
                    "bars": self.bars_in_position,
                    "pnl": net_pnl,
                })
                self.position = 0
                self.stop_price = 0.0

        if self.equity < 0.5:  # blew up half the capital
            terminated = True
            reward -= self.reward_cfg["blowup_penalty"]  # big penalty

        self.peak_equity = max(self.peak_equity, current_equity)

        return self._get_obs(), float(reward), terminated, truncated, self._info()

    # --------------------------------------------------------------
    def render(self):
        print(
            f"t={self.t} equity={self.equity:.4f} pos={self.position} "
            f"trades={len(self.trades)}"
        )

    def get_stats(self):
        """สรุปผลหลังจบ episode"""
        if not self.trades:
            return {"trades": 0, "return": self.equity - 1, "max_dd": 0}
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        eq = np.array(self._curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return {
            "trades": len(self.trades),
            "return": self.equity - 1,
            "win_rate": len(wins) / len(self.trades) if self.trades else 0,
            "avg_win": np.mean([t["pnl"] for t in wins]) if wins else 0,
            "avg_loss": np.mean([t["pnl"] for t in losses]) if losses else 0,
            "max_dd": float(dd.min()),
            "profit_factor": (
                sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
                if losses else float("inf")
            ),
        }
