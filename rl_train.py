"""
RL Trainer — PPO + TradingEnv
------------------------------
เทรน RL agent บนข้อมูลประวัติศาสตร์ EURUSD H1

ติดตั้ง:
    pip install stable-baselines3[extra] gymnasium

วิธีใช้:
    python rl_train.py EURUSD_H1.csv --steps 200000 --name rl_v1

output:
    rl_v1.zip          (PPO model)
    rl_v1_logs/        (tensorboard logs - optional)
"""
import sys
import io
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Windows console UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from trading_env import TradingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file with OHLC + features")
    ap.add_argument("--steps", type=int, default=200_000, help="training steps")
    ap.add_argument("--name", default="rl_v1", help="model name (no .zip)")
    ap.add_argument("--window", type=int, default=10, help="state window size")
    ap.add_argument("--train_pct", type=float, default=0.8, help="train/test split")
    ap.add_argument("--ep_len", type=int, default=2000, help="bars per episode")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "a2c"])
    ap.add_argument("--reward_mode", default="realized",
                    choices=["realized", "mtm"],
                    help="realized=ปิดแล้วถึงให้ reward (แนะนำ), mtm=ทุก step (buy-hold trap)")
    ap.add_argument("--max_hold", type=int, default=30,
                    help="บังคับปิด position ถ้าถือเกินกี่ bars")
    ap.add_argument("--net_arch", default="auto",
                    help="NN layers e.g. '128,64' or 'auto' (scale by window)")

    # Advanced PPO hyperparameters
    ap.add_argument("--learning_rate", type=float, default=3e-4,
                    help="learning rate (default 3e-4) — ลดถ้า kl/clip สูง")
    ap.add_argument("--clip_range", type=float, default=0.2,
                    help="PPO clip range (default 0.2)")
    ap.add_argument("--ent_coef", type=float, default=0.01,
                    help="entropy coefficient (default 0.01) — เพิ่มถ้า exploration หาย")
    # Extended PPO hyperparameters (rollout / optim / value)
    ap.add_argument("--n_steps", type=int, default=2048,
                    help="rollout buffer size per env (default 2048)")
    ap.add_argument("--n_epochs", type=int, default=10,
                    help="PPO update epochs per rollout (default 10)")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="minibatch size (default 64) — ต้อง <= n_steps")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="discount factor (default 0.99)")
    ap.add_argument("--gae_lambda", type=float, default=0.95,
                    help="GAE lambda — advantage smoothing (default 0.95)")
    ap.add_argument("--vf_coef", type=float, default=0.5,
                    help="value function loss coefficient (default 0.5)")
    args = ap.parse_args()

    # Auto-scale NN by window size
    if args.net_arch == "auto":
        if args.window >= 50:
            net_arch = [512, 256, 128]   # window ใหญ่ -> NN ใหญ่
        elif args.window >= 20:
            net_arch = [256, 128, 64]
        else:
            net_arch = [128, 64]
    else:
        net_arch = [int(x) for x in args.net_arch.split(",")]
    print(f"[net] arch = {net_arch}")

    print("=" * 60)
    print("  RL Trainer — TradingEnv + PPO")
    print("=" * 60)

    # ---------- load data ----------
    print(f"\n[load] {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  rows: {len(df):,}")

    # drop leaky columns + non-feature columns
    leaky = [c for c in df.columns
             if any(k in c.lower() for k in ("future_", "forward_", "next_", "target"))]
    if leaky:
        print(f"  drop leaky/target: {leaky}")
        df = df.drop(columns=leaky)

    # detect feature columns (numeric, exclude OHLC + ids)
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  features ({len(feature_cols)}): {feature_cols}")

    # parse timestamp + sort
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # normalize features (RL ชอบ scaled input)
    print("[normalize] z-score features ...")
    feat_mean = df[feature_cols].mean()
    feat_std = df[feature_cols].std() + 1e-8
    df[feature_cols] = (df[feature_cols] - feat_mean) / feat_std
    # fill NaN
    df = df.fillna(0).reset_index(drop=True)

    # save normalization stats so we can use same in inference
    norm_path = Path(f"{args.name}_norm.csv")
    pd.DataFrame({"mean": feat_mean, "std": feat_std}).to_csv(norm_path)
    print(f"  saved norm stats -> {norm_path}")

    # ---------- split ----------
    split = int(len(df) * args.train_pct)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    print(f"\n[split] train: {len(train_df):,} | test: {len(test_df):,}")

    # ---------- create environments ----------
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_train_env():
        return Monitor(TradingEnv(
            train_df, feature_cols,
            window_size=args.window,
            max_steps=args.ep_len,
            reward_mode=args.reward_mode,
            max_hold_bars=args.max_hold,
        ))

    def make_eval_env():
        return Monitor(TradingEnv(
            test_df, feature_cols,
            window_size=args.window,
            max_steps=len(test_df) - args.window - 2,
            reward_mode=args.reward_mode,
            max_hold_bars=args.max_hold,
        ))

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    # ---------- create model ----------
    print(f"\n[model] {args.algo.upper()}")
    log_dir = f"./{args.name}_logs/"

    if args.algo == "ppo":
        # Sanity: batch_size must be <= n_steps and n_steps % batch_size == 0
        if args.batch_size > args.n_steps:
            print(f"[warn] batch_size({args.batch_size}) > n_steps({args.n_steps}); clamping to n_steps")
            args.batch_size = args.n_steps

        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,           # encourage exploration
            vf_coef=args.vf_coef,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=net_arch),
        )
        print(f"[hyper] lr={args.learning_rate}, clip={args.clip_range}, ent={args.ent_coef}")
        print(f"[hyper] n_steps={args.n_steps}, batch={args.batch_size}, epochs={args.n_epochs}")
        print(f"[hyper] gamma={args.gamma}, gae_lambda={args.gae_lambda}, vf_coef={args.vf_coef}")
    elif args.algo == "dqn":
        model = DQN(
            "MlpPolicy", train_env,
            learning_rate=1e-4,
            buffer_size=50_000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=500,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=net_arch),
        )
    else:  # a2c
        model = A2C("MlpPolicy", train_env, learning_rate=7e-4,
                    verbose=1, tensorboard_log=log_dir)

    # ---------- train ----------
    print(f"\n[train] {args.steps:,} steps ...")
    print("(progress bar will appear; tensorboard --logdir ./{}_logs to monitor)".format(args.name))

    from stable_baselines3.common.callbacks import EvalCallback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./{args.name}_best/",
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=True)

    # ---------- save ----------
    save_path = f"{args.name}.zip"
    model.save(save_path)
    print(f"\n[save] -> {save_path}")

    # ---------- quick test ----------
    print("\n[eval] quick run on test set ...")
    test_env_raw = TradingEnv(test_df, feature_cols, window_size=args.window,
                              max_steps=len(test_df) - args.window - 2,
                              reward_mode=args.reward_mode,
                              max_hold_bars=args.max_hold)
    obs, _ = test_env_raw.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_raw.step(int(action))
        done = terminated or truncated

    stats = test_env_raw.get_stats()
    print("\n" + "=" * 50)
    print("  Quick test on out-of-sample")
    print("=" * 50)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v}")
    print("=" * 50)
    print(f"\nรัน: python rl_backtest.py {args.name} {args.csv}  เพื่อดู backtest เต็ม")


if __name__ == "__main__":
    main()
