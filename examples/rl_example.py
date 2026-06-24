"""
RL Trading Example — ใช้ PPO เทรด EUR/USD
------------------------------------------
ต้อง: pip install stable-baselines3[extra] gymnasium

ขั้นตอน:
1) สร้าง Environment (ตลาดจำลอง)
2) Train Agent (PPO) ให้ interact กับ env
3) Evaluate บนข้อมูล out-of-sample
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ==================================================================
# 1) Trading Environment — ตลาดจำลอง
# ==================================================================
class TradingEnv(gym.Env):
    """
    State:  [features of current bar + position + unrealized_pnl]
    Action: 0=Hold, 1=Buy (go long), 2=Sell (go short), 3=Close
    Reward: change in equity per step
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, feature_cols: list, window_size: int = 10):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.n_features = len(feature_cols)

        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)

        # Observation: features (window_size * n_features) + 2 (position + unrealized_pnl)
        obs_dim = window_size * self.n_features + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0          # -1=short, 0=flat, +1=long
        self.entry_price = 0.0
        self.equity = 10000.0
        self.initial_equity = self.equity
        self.trade_count = 0
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        """สร้าง state: window ของ features + position info"""
        window = self.df.loc[
            self.current_step - self.window_size : self.current_step - 1,
            self.feature_cols,
        ].values.flatten()

        unrealized = 0.0
        if self.position != 0:
            price = self.df.at[self.current_step, "close"]
            unrealized = (price - self.entry_price) / self.entry_price * self.position

        return np.concatenate([window, [self.position, unrealized]]).astype(np.float32)

    def step(self, action):
        price = self.df.at[self.current_step, "close"]
        prev_equity = self.equity
        cost = 0.0

        # Execute action
        if action == 1 and self.position == 0:           # BUY
            self.position = 1
            self.entry_price = price
            self.trade_count += 1
            cost = 0.0002  # spread + commission
        elif action == 2 and self.position == 0:         # SELL
            self.position = -1
            self.entry_price = price
            self.trade_count += 1
            cost = 0.0002
        elif action == 3 and self.position != 0:         # CLOSE
            pnl_pct = (price - self.entry_price) / self.entry_price * self.position
            self.equity *= (1 + pnl_pct - 0.0002)
            self.position = 0
            self.entry_price = 0.0

        # Update unrealized equity (mark-to-market)
        if self.position != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price * self.position
            current_equity = self.equity * (1 + pnl_pct)
        else:
            current_equity = self.equity

        # Reward = change in equity
        reward = (current_equity - prev_equity) / prev_equity
        # Penalty for cost
        reward -= cost

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        # Auto-close if done
        if done and self.position != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price * self.position
            self.equity *= (1 + pnl_pct)

        truncated = False
        self.history.append(current_equity)
        return self._get_obs(), reward, done, truncated, {}


# ==================================================================
# 2) Train + Evaluate
# ==================================================================
def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    # --- load data ---
    df = pd.read_csv("EURUSD_H1.csv")
    feature_cols = ["rsi", "ema_fast", "ema_slow", "atr", "return_1", "hl_range"]

    # Normalize features (RL ชอบ scaled input)
    for c in feature_cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    # Split by time
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # --- create env ---
    env = DummyVecEnv([lambda: TradingEnv(train_df, feature_cols)])

    # --- train PPO ---
    print("\n[RL] Training PPO ... (ใช้เวลา 10-30 นาที)")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./rl_logs/",
    )
    model.learn(total_timesteps=100_000)
    model.save("rl_trader")

    # --- evaluate on test set ---
    print("\n[RL] Evaluating on out-of-sample ...")
    test_env = TradingEnv(test_df, feature_cols)
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env.step(action)

    print(f"  Start equity: {test_env.initial_equity:,.2f}")
    print(f"  Final equity: {test_env.equity:,.2f}")
    print(f"  Return:       {(test_env.equity / test_env.initial_equity - 1):+.2%}")
    print(f"  Trades:       {test_env.trade_count}")


if __name__ == "__main__":
    main()
