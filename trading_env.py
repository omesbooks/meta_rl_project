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

Reward (V4 — Honest):
    + Net-PnL × 50 ตอนปิด (real $ dominates reward)
    + +0.01 bonus เฉพาะถ้า net pnl > 0.5% (meaningful win)
    + incremental unrealized × 0.1 (tiny direction hint)
    - give-back penalty -0.001 (peak/2 reached, threshold 0.2%)
    - trade penalty -0.005 ตอนเปิด (heavy: ป้องกัน over-trading)
    - hold idle penalty -0.0001 ต่อ bar (ตอน flat)
    - time-decay -0.0002 หลังถือเกิน 70% max_hold
    → Break-even WR ~53% (ตรงกับ real $ break-even)
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


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

        # Action: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)

        # Obs: window flat + [position, unrealized_pnl, bars_in_position]
        obs_dim = self.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Pre-compute feature window matrix for speed
        self._features = self.df[feature_cols].values.astype(np.float32)
        self._closes = self.df["close"].values.astype(np.float32)

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

    # --------------------------------------------------------------
    def step(self, action):
        prev_equity = self.equity
        price = self._closes[self.t]
        trade_closed_pnl = None
        just_opened = False   # ⭐ Phase 2: flag สำหรับ trade penalty (แทน cost > 0)

        # ---- execute action ----
        # ⭐⭐ Phase 2: apply spread to entry/exit prices (match backtest_live.py)
        # Long:  buy at higher price (market + spread), sell at lower (market - spread)
        # Short: sell at lower price (market - spread), buy at higher (market + spread)
        if action == 1 and self.position == 0:           # BUY
            self.position = 1
            self.entry_price = price * (1.0 + self.spread)   # ⭐ shift up
            self.bars_in_position = 0
            just_opened = True

        elif action == 2 and self.position == 0:         # SELL
            self.position = -1
            self.entry_price = price * (1.0 - self.spread)   # ⭐ shift down
            self.bars_in_position = 0
            just_opened = True

        elif action == 3 and self.position != 0:         # CLOSE
            # ⭐ apply exit spread (long: -, short: +)
            exit_p = price * (1.0 - self.spread) if self.position == 1 else price * (1.0 + self.spread)
            pnl_pct = (exit_p - self.entry_price) / self.entry_price * self.position
            net_pnl = pnl_pct - self.commission
            self.equity *= (1 + net_pnl)
            trade_closed_pnl = net_pnl
            self.trades.append({
                "entry": self.entry_price,
                "exit": exit_p,                            # ⭐ store shifted price
                "side": self.position,
                "bars": self.bars_in_position,
                "pnl": net_pnl,
            })
            self.position = 0
            self.entry_price = 0.0
            self.bars_in_position = 0

        # ---- force close if held too long (avoid buy-and-hold trap) ----
        if self.position != 0 and self.bars_in_position >= self.max_hold_bars:
            # ⭐ apply exit spread for forced close too
            exit_p = price * (1.0 - self.spread) if self.position == 1 else price * (1.0 + self.spread)
            pnl_pct = (exit_p - self.entry_price) / self.entry_price * self.position
            net_pnl = pnl_pct - self.commission
            self.equity *= (1 + net_pnl)
            trade_closed_pnl = net_pnl
            self.trades.append({
                "entry": self.entry_price, "exit": exit_p,
                "side": self.position, "bars": self.bars_in_position,
                "pnl": net_pnl, "forced": True,
            })
            self.position = 0
            self.entry_price = 0.0
            self.bars_in_position = 0

        # ---- update mark-to-market equity ----
        if self.position != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price * self.position
            current_equity = self.equity * (1 + pnl_pct)
            self.bars_in_position += 1
        else:
            current_equity = self.equity

        # ---- reward shaping ----
        if self.reward_mode == "realized":
            # ⭐⭐⭐ V4: Honest Reward (Real $ dominates)
            # ─────────────────────────────────────────────────────────
            # ปัญหาใน V3: profit bonus capped 0.005 → ทุก win เล็กก็ได้ bonus เต็ม
            #            → agent เรียน "เก็บ small wins" → cost กิน
            #
            # V4 หลักการ:
            #   1. close_pnl multiplier 5 → 50  (real $ เด่นใน reward)
            #   2. profit bonus เฉพาะ pnl > 0.5% (10× cost)
            #   3. trade penalty 0.002 → 0.005 (ป้องกัน over-trading จริงจัง)
            #   4. unrealized scale 0.3 → 0.1 (ลด dense reward gaming)
            #   5. give-back penalty 0.0002 → 0.001 (เร็วขึ้น 5×)
            # → Break-even WR คาดว่า ~53% (สมจริง)
            # ─────────────────────────────────────────────────────────
            reward = 0.0

            # 1) Tiny unrealized hint (just direction signal — no gaming)
            if self.position != 0 and self.entry_price > 0:
                current_unrealized = (price - self.entry_price) / self.entry_price * self.position
                unrealized_delta = current_unrealized - self._prev_unrealized
                reward += unrealized_delta * 0.1   # ⭐ V4: ลด 0.3 → 0.1

                # Give-back penalty (more aggressive)
                self._peak_unrealized = max(self._peak_unrealized, current_unrealized)
                if (self._peak_unrealized > 0.002 and       # ⭐ V4: 0.001 → 0.002 (เกณฑ์)
                        current_unrealized < self._peak_unrealized * 0.5):
                    reward -= 0.001                          # ⭐ V4: 0.0002 → 0.001 (× 5)

                self._prev_unrealized = current_unrealized
            else:
                self._prev_unrealized = 0.0
                self._peak_unrealized = 0.0

            # 2) ⭐⭐ STRONG close reward — let real $ dominate
            if trade_closed_pnl is not None:
                # ⭐ V4: multiplier 5 → 50 — real pnl เด่นมาก
                reward += trade_closed_pnl * 50.0

                # ⭐ V4: profit bonus เฉพาะ MEANINGFUL wins (> 0.5% net)
                # cost ต่อ trade = 0.05% → bonus ต้องการ pnl 10× ของ cost
                if trade_closed_pnl > 0.005:
                    reward += 0.01

                # reset trackers
                self._prev_unrealized = 0.0
                self._peak_unrealized = 0.0

            # 3) ⭐ HEAVY Trade Penalty (was 0.002 → 0.005)
            #    ตอนเปิด trade agent ต้อง "มั่นใจมาก" จึงจะคุ้ม
            if just_opened:
                reward -= 0.005   # ⭐ V4: 0.002 → 0.005 (× 2.5)

            # 4) ลงโทษ "Hold ตลอด" (no position, doing nothing)
            if action == 0 and self.position == 0:
                reward -= 0.0001

            # 5) Mild time-decay while in position (encourage timely exits)
            if self.position != 0 and self.bars_in_position > self.max_hold_bars * 0.7:
                reward -= 0.0002
        else:
            # legacy MTM mode — Phase 2: spread auto-captured via equity drop on entry
            reward = (current_equity - prev_equity) / max(prev_equity, 1e-9)
            if trade_closed_pnl is not None and trade_closed_pnl > 0:
                reward += 0.001
            if self.position != 0 and self.bars_in_position > 50:
                reward -= 0.0005

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

        if self.equity < 0.5:  # blew up half the capital
            terminated = True
            reward -= 1.0  # big penalty

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
