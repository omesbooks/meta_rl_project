"""
Grid search hyperparameters for backtest_live.py
"""
import sys, io, subprocess, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CSV = "training_data_v2_relabeled.csv"
MODEL = "rl_v3"

print(f"{'conf':>5} {'sl':>4} {'tp':>4} | {'trades':>8} {'WR':>7} {'PF':>5} {'Return':>8} {'DD':>7}")
print("-" * 70)

best_pf = 0
best_config = None

for conf in [0.80, 0.85, 0.90, 0.95]:
    for atr_sl in [1.5, 2.0, 2.5]:
        for atr_tp in [2.0, 3.0, 4.0]:
            cmd = [
                sys.executable, "backtest_live.py", MODEL, CSV,
                "--start", "0.8",
                "--conf", str(conf),
                "--atr_sl", str(atr_sl),
                "--atr_tp", str(atr_tp),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            output = result.stdout

            # Parse stats
            def find(pattern):
                m = re.search(pattern, output)
                return m.group(1) if m else "—"

            trades = find(r"Total trades\s+:\s+([\d,]+)")
            wr = find(r"Win rate\s+:\s+([\d.]+)%")
            pf = find(r"Profit factor\s+:\s+([\d.]+|inf)")
            ret = find(r"Return\s+:\s+([+\-][\d.]+)%")
            dd = find(r"Max drawdown\s+:\s+([\-\d.]+)%")

            print(f"{conf:>5} {atr_sl:>4} {atr_tp:>4} | "
                  f"{trades:>8} {wr:>6}% {pf:>5} {ret:>7}% {dd:>6}%")

            try:
                pf_val = float(pf)
                if pf_val > best_pf:
                    best_pf = pf_val
                    best_config = (conf, atr_sl, atr_tp, trades, wr, pf, ret, dd)
            except ValueError:
                pass

print("-" * 70)
if best_config:
    print(f"\n🏆 Best config:")
    c, s, t, tr, w, p, r, d = best_config
    print(f"   conf={c}, atr_sl={s}, atr_tp={t}")
    print(f"   trades={tr}, WR={w}%, PF={p}, Return={r}%, DD={d}%")
