"""
RL Fine-tuning — ปรับ model เดิมด้วยข้อมูลใหม่ (Smart Mixing)
=============================================================

โหลด PPO model ที่เทรนไว้แล้ว → train ต่อด้วย data ใหม่
ใช้สำหรับ adaptive retraining ใน production

วิธีการ (Smart Fine-tune):
  1. โหลด model เก่า + normalization stats เดิม
  2. ผสมข้อมูล: old(30%) + new(70%) -> ป้องกัน catastrophic forgetting
  3. ลด learning_rate ให้เบากว่า train ปกติ (gradient ค่อยๆ adjust)
  4. Train ต่อ 50k steps (น้อยกว่า scratch 200k)
  5. Save เป็น model ใหม่ + normalization ใหม่

Modes:
  --mode pure      : fine-tune บน new data อย่างเดียว (เสี่ยง forgetting)
  --mode mixed     : ผสม old + new (default, แนะนำ)
  --mode replay    : ใช้ replay buffer (advanced)

Usage:
    python rl_finetune.py rl_v3 \\
        --old_csv training_data_v2_relabeled.csv \\
        --new_csv new_data_2026_relabeled.csv \\
        --steps 50000 \\
        --mix_ratio 0.3 \\
        --name rl_v4
"""
import sys
import io
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from trading_env import TradingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_model",
                    help="ชื่อ model เดิม (without .zip), เช่น rl_v3")
    ap.add_argument("--old_csv", required=True,
                    help="CSV เก่าที่เคย train ไว้")
    ap.add_argument("--new_csv",
                    help="CSV ใหม่ (ถ้าไม่ใส่จะใช้ old_csv อย่างเดียว = waste!)")
    ap.add_argument("--steps", type=int, default=50_000,
                    help="fine-tune steps (default 50k)")
    ap.add_argument("--mix_ratio", type=float, default=0.3,
                    help="สัดส่วน old data (0.3 = 30% old + 70% new)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate (lower than scratch's 3e-4)")
    ap.add_argument("--mode", default="mixed",
                    choices=["pure", "mixed", "replay"],
                    help="pure=new only, mixed=ผสม (default)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--max_hold", type=int, default=30)
    ap.add_argument("--ep_len", type=int, default=2000)
    ap.add_argument("--name", default=None,
                    help="ชื่อ output (default = base_model + _ft)")
    args = ap.parse_args()

    out_name = args.name or f"{args.base_model}_ft"

    print("=" * 60)
    print("  RL Fine-tuning")
    print("=" * 60)
    print(f"  Base model    : {args.base_model}.zip")
    print(f"  Mode          : {args.mode}")
    print(f"  Steps         : {args.steps:,}")
    print(f"  Mix ratio     : {args.mix_ratio:.0%} old / {1-args.mix_ratio:.0%} new")
    print(f"  Learning rate : {args.lr}")
    print(f"  Output        : {out_name}.zip")
    print("=" * 60)

    # =================================================================
    # Step 1: Verify base model exists
    # =================================================================
    base_path = Path(f"{args.base_model}.zip")
    base_norm = Path(f"{args.base_model}_norm.csv")
    if not base_path.exists():
        print(f"\n❌ Base model not found: {base_path}")
        print(f"   ต้องเทรน {args.base_model} ก่อนด้วย rl_train.py")
        return
    if not base_norm.exists():
        print(f"\n⚠️  Norm stats not found: {base_norm}")
        print(f"   จะคำนวณใหม่จาก data")

    # =================================================================
    # Step 2: Load + prepare data
    # =================================================================
    def load_csv(path):
        df = pd.read_csv(path)
        leaky = [c for c in df.columns if any(k in c.lower()
                 for k in ("future_", "forward_", "next_", "target"))]
        if leaky:
            df = df.drop(columns=leaky)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    print(f"\n[load] old_csv: {args.old_csv}")
    old_df = load_csv(args.old_csv)
    print(f"  rows: {len(old_df):,}")

    if args.new_csv:
        print(f"\n[load] new_csv: {args.new_csv}")
        new_df = load_csv(args.new_csv)
        print(f"  rows: {len(new_df):,}")
    else:
        print(f"\n⚠️  ไม่มี new_csv — จะใช้แค่ old_csv (เหมือน train ต่อบน data เดิม)")
        new_df = None

    # Detect feature columns
    skip = {"timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in old_df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(old_df[c])]
    print(f"\n[features] {len(feature_cols)} columns")

    # =================================================================
    # Step 3: Mix data based on mode
    # =================================================================
    if args.mode == "pure" and new_df is not None:
        # ใช้ new อย่างเดียว — เสี่ยง forgetting
        print(f"\n[mode] PURE — train บน new data อย่างเดียว ⚠️")
        train_df = new_df.copy()

    elif args.mode == "mixed" and new_df is not None:
        # ผสม old (sample) + new (all) — Smart!
        print(f"\n[mode] MIXED — ผสม old({args.mix_ratio:.0%}) + new({1-args.mix_ratio:.0%}) ✅")
        old_sample_n = int(len(new_df) * args.mix_ratio / (1 - args.mix_ratio))
        old_sample_n = min(old_sample_n, len(old_df))
        old_sample = old_df.sample(n=old_sample_n, random_state=42)
        # Concat แล้ว sort by timestamp ถ้ามี
        train_df = pd.concat([old_sample, new_df], ignore_index=True)
        if "timestamp" in train_df.columns:
            train_df = train_df.sort_values("timestamp").reset_index(drop=True)
        print(f"  old sampled: {len(old_sample):,} rows")
        print(f"  new (all)  : {len(new_df):,} rows")
        print(f"  total mix  : {len(train_df):,} rows")

    elif args.mode == "replay":
        # Use both, but with weighted sampling
        print(f"\n[mode] REPLAY — full buffer (old + new) ⭐")
        train_df = pd.concat([old_df, new_df if new_df is not None else pd.DataFrame()],
                              ignore_index=True)
        if "timestamp" in train_df.columns:
            train_df = train_df.sort_values("timestamp").reset_index(drop=True)
        print(f"  total: {len(train_df):,} rows")

    else:
        # Fallback: use old only
        print(f"\n[mode] fallback — using old_csv only")
        train_df = old_df.copy()

    # =================================================================
    # Step 4: Apply normalization (use BASE model's stats — important!)
    # =================================================================
    if base_norm.exists():
        print(f"\n[norm] ใช้ stats เดิมจาก {base_norm}")
        norm = pd.read_csv(base_norm, index_col=0)
        for c in feature_cols:
            if c in norm.index:
                train_df[c] = (train_df[c] - norm.at[c, "mean"]) / (norm.at[c, "std"] + 1e-8)
        # บันทึก norm เดิมเป็นของ output (สำคัญสำหรับ inference)
        norm.to_csv(f"{out_name}_norm.csv")
        print(f"  saved -> {out_name}_norm.csv (เหมือนเดิม)")
    else:
        print(f"\n[norm] คำนวณใหม่จาก train_df")
        feat_mean = train_df[feature_cols].mean()
        feat_std = train_df[feature_cols].std() + 1e-8
        train_df[feature_cols] = (train_df[feature_cols] - feat_mean) / feat_std
        pd.DataFrame({"mean": feat_mean, "std": feat_std}).to_csv(f"{out_name}_norm.csv")

    train_df = train_df.fillna(0).reset_index(drop=True)

    # =================================================================
    # Step 5: Setup PPO with loaded model
    # =================================================================
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_env():
        return Monitor(TradingEnv(
            train_df, feature_cols,
            window_size=args.window,
            max_steps=args.ep_len,
            reward_mode="realized",
            max_hold_bars=args.max_hold,
        ))

    train_env = DummyVecEnv([make_env])

    print(f"\n[load] base model: {base_path}")
    model = PPO.load(base_path, env=train_env)

    # ลด learning rate ให้นุ่มกว่า train ครั้งแรก
    print(f"[adjust] learning_rate: 3e-4 -> {args.lr} (gentler updates)")
    model.learning_rate = args.lr
    # Re-create optimizer with new lr
    for g in model.policy.optimizer.param_groups:
        g["lr"] = args.lr

    # =================================================================
    # Step 6: Fine-tune!
    # =================================================================
    print(f"\n[finetune] {args.steps:,} steps ...")
    print(f"  (~{args.steps // 6000} นาที บน CPU)")

    model.learn(
        total_timesteps=args.steps,
        progress_bar=True,
        reset_num_timesteps=False,  # สำคัญ! เก็บ counter เดิม
    )

    # =================================================================
    # Step 7: Save
    # =================================================================
    save_path = f"{out_name}.zip"
    model.save(save_path)
    print(f"\n[save] {save_path}")
    print(f"[save] {out_name}_norm.csv")

    # =================================================================
    # Step 8: Quick eval on new data (if provided)
    # =================================================================
    if new_df is not None:
        print(f"\n[eval] quick run on new data ...")
        # apply same norm
        eval_df = new_df.copy()
        if base_norm.exists():
            for c in feature_cols:
                if c in norm.index:
                    eval_df[c] = (eval_df[c] - norm.at[c, "mean"]) / (norm.at[c, "std"] + 1e-8)
        eval_df = eval_df.fillna(0).reset_index(drop=True)

        eval_env = TradingEnv(eval_df, feature_cols,
                              window_size=args.window,
                              max_steps=len(eval_df) - args.window - 2,
                              reward_mode="realized",
                              max_hold_bars=args.max_hold)
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(int(action))
            done = terminated or truncated

        stats = eval_env.get_stats()
        print("\n" + "=" * 50)
        print(f"  Quick eval on NEW data")
        print("=" * 50)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:<15}: {v:.4f}")
            else:
                print(f"  {k:<15}: {v}")
        print("=" * 50)

    print(f"\n✓ Fine-tuning เสร็จ! ลอง backtest ดูได้:")
    print(f"  python rl_backtest.py {out_name} <csv>")


if __name__ == "__main__":
    main()
