"""
Training Diagnosis — อ่าน TensorBoard log + evaluations.npz ของ model
แล้ววินิจฉัยสุขภาพการเทรนอัตโนมัติ: overfit point, plateau, KL stability,
signal strength (explained variance), entropy collapse

Usage:
    python train_diagnose.py <model_name> [--json]

Outputs:
    artifacts/models/<name>/<name>_train_diag.png   (4-panel chart)
    updates <name>.train.json -> meta["train_diagnosis"]
    prints verdict lines (DIAG| prefix for GUI parsing)
"""
import sys
import io
import json
import argparse
from pathlib import Path

import numpy as np

if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from artifact_paths import logs_dir, train_meta_path, train_diag_path


def _load_scalars(run_dir: Path):
    """Read all scalar series from the newest tfevents run dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev]),
                    np.array([e.value for e in ev]))
    return out


def diagnose(model_name: str):
    ldir = logs_dir(model_name)
    runs = sorted([p for p in ldir.glob("*") if p.is_dir()],
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no TensorBoard run dirs under {ldir}")
    scalars = _load_scalars(runs[-1])

    def series(tag):
        return scalars.get(tag, (np.array([]), np.array([])))

    tr_step, tr_rew = series("rollout/ep_rew_mean")
    ev_step, ev_rew = series("eval/mean_reward")
    xv_step, xvar = series("train/explained_variance")
    kl_step, kl = series("train/approx_kl")
    en_step, ent = series("train/entropy_loss")
    cf_step, clipf = series("train/clip_fraction")

    findings = []          # (level, thai_text)  level: ok / warn / bad
    summary = {}

    # --- 1. Overfit check: eval peak vs end ---------------------------------
    if len(ev_rew) >= 5:
        peak_i = int(np.argmax(ev_rew))
        peak_step, peak_val = int(ev_step[peak_i]), float(ev_rew[peak_i])
        end_val = float(np.mean(ev_rew[-3:]))
        ev_range = float(ev_rew.max() - ev_rew.min()) or 1e-9
        total = int(ev_step[-1])
        overfit = (peak_step <= 0.75 * total
                   and (peak_val - end_val) > 0.25 * ev_range)
        summary["eval_peak_step"] = peak_step
        summary["eval_peak_value"] = peak_val
        summary["eval_end_value"] = end_val
        summary["overfit_after_peak"] = bool(overfit)
        if overfit:
            findings.append(("bad",
                f"OVERFIT หลัง step ~{peak_step:,} — eval พีค {peak_val:.2f} "
                f"แล้วร่วงเหลือ {end_val:.2f} ตอนจบ → ใช้ best_model.zip "
                f"(checkpoint ที่ ~{peak_step/total:.0%} ของการเทรน) แทน final"))
        else:
            findings.append(("ok",
                f"ไม่พบ overfit ชัดเจน — eval จบใกล้จุดพีค "
                f"(peak {peak_val:.2f} @ {peak_step:,} / end {end_val:.2f})"))
        if float(ev_rew.max()) <= 0:
            findings.append(("warn",
                "eval reward ไม่เคยเป็นบวกเลยทั้งการเทรน — สูตร/ข้อมูลชุดนี้"
                "ยังไม่เห็น edge บน out-of-sample"))

    # --- 2. Plateau / undertrained ------------------------------------------
    if len(tr_rew) >= 8:
        q = max(2, len(tr_rew) // 4)
        last_q_gain = float(tr_rew[-1] - tr_rew[-q])
        full_range = float(tr_rew.max() - tr_rew.min()) or 1e-9
        still_rising = last_q_gain > 0.10 * full_range
        summary["train_still_rising"] = bool(still_rising)
        summary["train_reward_end"] = float(tr_rew[-1])
        if still_rising:
            findings.append(("warn",
                f"train reward ยังไต่อยู่ตอนจบ (+{last_q_gain:.2f} ในช่วงท้าย) "
                f"— ยังไม่ plateau, เพิ่ม steps ได้อีก"))
        else:
            findings.append(("ok", "train reward เข้าสู่ plateau แล้ว — steps พอเหมาะ"))

    # --- 3. KL stability -----------------------------------------------------
    if len(kl) >= 4:
        kl_late = float(np.mean(kl[len(kl) // 2:]))
        summary["approx_kl_late_mean"] = kl_late
        if kl_late > 0.10:
            findings.append(("bad",
                f"approx_kl เฉลี่ยครึ่งหลัง {kl_late:.3f} สูงมาก (โซนแนะนำ <0.05) "
                f"— policy กระโดดแรงต่อ update → ลด learning rate หรือ clip_range"))
        elif kl_late > 0.05:
            findings.append(("warn",
                f"approx_kl {kl_late:.3f} สูงกว่าโซนแนะนำเล็กน้อย (<0.05) "
                f"— พอรับได้ แต่ลด lr ลงอาจเนียนขึ้น"))
        else:
            findings.append(("ok", f"approx_kl {kl_late:.3f} อยู่ในโซนเสถียร"))

    # --- 4. Signal strength (explained variance) ----------------------------
    if len(xvar) >= 4:
        xv_end = float(np.mean(xvar[-5:]))
        summary["explained_variance_end"] = xv_end
        if xv_end < 0.2:
            findings.append(("bad",
                f"explained_variance ตอนจบ {xv_end:.2f} ต่ำ — value function "
                f"ทำนาย return แทบไม่ได้ = features มี signal น้อย/reward noise สูง "
                f"(ปัญหาฝั่งข้อมูลมากกว่าการเทรน)"))
        elif xv_end < 0.5:
            findings.append(("warn",
                f"explained_variance {xv_end:.2f} ปานกลาง — มี signal แต่ไม่แรง"))
        else:
            findings.append(("ok",
                f"explained_variance {xv_end:.2f} ดี — features อธิบาย return ได้"))

    # --- 5. Entropy collapse -------------------------------------------------
    if len(ent) >= 8:
        q = max(2, len(ent) // 4)
        e0, e_q = abs(float(ent[0])), abs(float(ent[q]))
        summary["entropy_start"] = float(ent[0])
        summary["entropy_end"] = float(ent[-1])
        if e0 > 1e-9 and (e0 - e_q) / e0 > 0.7:
            findings.append(("warn",
                f"entropy หดเร็วผิดปกติในช่วงแรก ({ent[0]:.2f} → {ent[q]:.2f}) "
                f"— agent เลิกสำรวจเร็วไป ลองเพิ่ม ent_coef"))
        else:
            findings.append(("ok", "entropy ลดลงแบบค่อยเป็นค่อยไป — การสำรวจปกติ"))

    return scalars, findings, summary


def render_chart(model_name: str, scalars, summary) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(tag):
        return scalars.get(tag, (np.array([]), np.array([])))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    (ax_r, ax_xv), (ax_kl, ax_en) = axes

    tr_s, tr_v = series("rollout/ep_rew_mean")
    ev_s, ev_v = series("eval/mean_reward")
    ax_r.plot(tr_s, tr_v, color="#2563eb", label="train reward")
    if len(ev_v):
        ax_r.plot(ev_s, ev_v, color="#f59e0b", marker="o", markersize=3,
                  label="eval reward (OOS)")
        pk = summary.get("eval_peak_step")
        if pk is not None:
            ax_r.axvline(pk, color="#16a34a", linestyle="--", linewidth=1.2,
                         label=f"best eval @ {pk:,}")
            if summary.get("overfit_after_peak"):
                ax_r.axvspan(pk, tr_s[-1] if len(tr_s) else pk,
                             color="#ef4444", alpha=0.08)
    ax_r.axhline(0, color="gray", linewidth=0.6, alpha=0.6)
    ax_r.set_title("Reward: train vs eval (red zone = overfit)")
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    xv_s, xv_v = series("train/explained_variance")
    ax_xv.plot(xv_s, xv_v, color="#7c3aed")
    ax_xv.axhline(0.5, color="#16a34a", linestyle=":", linewidth=1, label="good >0.5")
    ax_xv.axhline(0.2, color="#ef4444", linestyle=":", linewidth=1, label="weak <0.2")
    ax_xv.set_ylim(-1, 1)
    ax_xv.set_title("Explained variance (signal strength)")
    ax_xv.legend(fontsize=8)
    ax_xv.grid(True, alpha=0.3)

    kl_s, kl_v = series("train/approx_kl")
    cf_s, cf_v = series("train/clip_fraction")
    ax_kl.plot(kl_s, kl_v, color="#dc2626", label="approx_kl")
    ax_kl.axhline(0.05, color="#16a34a", linestyle=":", linewidth=1,
                  label="stable <0.05")
    if len(cf_v):
        ax2 = ax_kl.twinx()
        ax2.plot(cf_s, cf_v, color="#94a3b8", alpha=0.6, label="clip_fraction")
        ax2.set_ylabel("clip fraction", color="#94a3b8", fontsize=8)
    ax_kl.set_title("PPO stability (approx_kl / clip fraction)")
    ax_kl.legend(fontsize=8, loc="upper left")
    ax_kl.grid(True, alpha=0.3)

    en_s, en_v = series("train/entropy_loss")
    ax_en.plot(en_s, en_v, color="#0891b2")
    ax_en.set_title("Entropy loss (exploration)")
    ax_en.set_xlabel("timesteps")
    ax_en.grid(True, alpha=0.3)

    fig.suptitle(f"Training Diagnosis — {model_name}", fontweight="bold")
    out = train_diag_path(model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        scalars, findings, summary = diagnose(args.model)
    except FileNotFoundError as e:
        print(f"DIAG|warn|ไม่พบ training log ของ {args.model}: {e}")
        return 1

    chart = render_chart(args.model, scalars, summary)
    print(f"[chart] -> {chart}")

    icon = {"ok": "✅", "warn": "🟡", "bad": "❌"}
    for level, text in findings:
        print(f"DIAG|{level}|{icon[level]} {text}")

    # Persist into train meta so the verdicts survive with the model
    mp = train_meta_path(args.model)
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8-sig"))
            meta["train_diagnosis"] = {
                "findings": [{"level": lv, "text": tx} for lv, tx in findings],
                **{k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in summary.items()},
            }
            mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                          encoding="utf-8")
            print(f"[meta] diagnosis saved -> {mp}")
        except Exception as e:
            print(f"[meta] skip ({e})")

    if args.json:
        print(json.dumps({"findings": [
            {"level": lv, "text": tx} for lv, tx in findings], **summary},
            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
