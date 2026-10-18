"""Backtest page helpers shared by rl_app.py and the web GUI.

Moved verbatim out of RLTradingStudio: the full report text built from
`<model>_live_bt.meta.json` (_load_backtest_report), the four stat cards, the
companion-M1 lookup (_find_matching_m1), the M1 auto-select rule (_auto_select_bt_m1)
and the Skip-% range preview (_update_backtest_skip_hint).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
NONE = "(none)"


# ------------------------------------------------------------------ M1 companion
def find_matching_m1(csv_name, work_dir=ROOT):
    """Find the companion _m1.csv for a dataset, if one exists in the project.

    Derived datasets (date-split _train/_test, _relabeled, _clean, regime
    _from_<date>, built training_data_<stem>) inherit the ORIGINAL
    collector's M1 file: M1Resolver slices by timestamp, so the full M1
    covers any time-subset — no need to split/copy the (huge) M1 file.
    Walks the name ancestry until an <ancestor>_m1.csv is found."""
    if not csv_name or csv_name == NONE:
        return None
    work_dir = Path(work_dir)
    seen = set()
    queue = [Path(csv_name).stem]
    while queue:
        s = queue.pop(0)
        if not s or s in seen:
            continue
        seen.add(s)
        cand = work_dir / f"{s}_m1.csv"
        if cand.exists():
            return cand.name
        # ancestors: strip known derived-name decorations
        if s.startswith("training_data_"):
            queue.append(s[len("training_data_"):])
        for suf in ("_train", "_test", "_relabeled", "_clean", "_enriched"):
            if s.endswith(suf):
                queue.append(s[:-len(suf)])
        m = re.match(r"^(.*)_from_\d{4}-\d{2}-\d{2}$", s)
        if m:
            queue.append(m.group(1))
    return None


def auto_select_m1(csv_name, current, previous_auto, work_dir=ROOT):
    """The M1 dropdown rule: pre-select the dataset's companion _m1.csv, but never
    override an explicit user choice. Returns (value, new_auto) — value None = keep."""
    current = (current or "").strip()
    if current not in ("", NONE) and current != previous_auto:
        return None, previous_auto              # preserve an explicit user choice
    m1 = find_matching_m1((csv_name or "").strip(), work_dir)
    if m1:
        return m1, m1
    return NONE, None


# ------------------------------------------------------------------ skip-% preview
def format_split_timestamp(value):
    try:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            if value.hour or value.minute or value.second:
                return value.strftime("%Y-%m-%d %H:%M")
            return value.strftime("%Y-%m-%d")
    except Exception:
        pass
    text = str(value)
    return text[:16] if len(text) > 16 else text


def format_range_line(label, ts, start_idx, end_idx):
    if ts is None or len(ts) == 0:
        return f"{label}: - (0 rows)"
    start_idx = max(0, int(start_idx))
    end_idx = min(len(ts) - 1, int(end_idx))
    if start_idx > end_idx or start_idx >= len(ts) or end_idx < 0:
        return f"{label}: - (0 rows)"
    rows = end_idx - start_idx + 1
    return (
        f"{label}: {format_split_timestamp(ts.iloc[start_idx])} -> "
        f"{format_split_timestamp(ts.iloc[end_idx])} "
        f"({rows:,} rows)"
    )


def skip_preview(ts, skip_raw):
    """(text, level) for the 'Skip %' hint; level 'warn' when the range is empty/invalid."""
    from ui_values import fraction
    if ts is None or len(ts) == 0:
        return "No valid timestamp rows", "warn"
    try:
        skip_pct = fraction(skip_raw, "bt_start", allow_zero=True, allow_one=False)
    except ValueError as exc:
        return str(exc), "warn"
    skip_pct = min(max(skip_pct, 0.0), 1.0)
    start_idx = max(0, min(int(len(ts) * skip_pct), len(ts)))
    skipped_line = format_range_line("Skipped", ts, 0, start_idx - 1)
    backtest_line = format_range_line("Backtest", ts, start_idx, len(ts) - 1)
    return f"{skipped_line}\n{backtest_line}", ("warn" if start_idx >= len(ts) else "")


# ------------------------------------------------------------------ report
def load_meta(model):
    from artifact_paths import backtest_meta_path
    mp = backtest_meta_path(model)
    if not mp.exists():
        return None
    return json.loads(mp.read_text(encoding="utf-8-sig"))


def _pc(v, d=2):
    return f"{v*100:.{d}f}%" if isinstance(v, (int, float)) else "—"


def stat_cards(meta):
    """Win rate / PF / return / max DD exactly as the CTk stat cards show them.
    Each entry: (text, tone) with tone 'good' | 'bad' | '' (dim)."""
    res = meta.get("result", {}) or {}
    if not res.get("total_trades"):
        return {"win_rate": ("—", "dim"), "profit_factor": ("—", "dim"), "return": ("—", "dim"), "max_dd": ("—", "dim")}
    pf = res.get("profit_factor", 0)
    ret = res.get("return_pct", 0)
    return {
        "win_rate": (_pc(res.get("win_rate"), 1), ""),
        "profit_factor": (f"{pf:.2f}", "good" if pf and pf > 1 else "bad"),
        "return": (_pc(ret), "good" if ret > 0 else "bad"),
        "max_dd": (_pc(res.get("max_drawdown")), "bad"),
    }


def report_lines(meta):
    """The full report text of the CTk Backtest page, line by line."""
    s = meta.get("settings", {})
    res = meta.get("result", {}) or {}
    pc = _pc

    L = []
    L.append(f"Model {meta.get('model','?')}  ·  {meta.get('data_csv','?')}  ·  "
             f"{meta.get('backtest_rows',0):,} rows  ·  "
             f"{(meta.get('backtest_period') or {}).get('start','?')} -> "
             f"{(meta.get('backtest_period') or {}).get('end','?')}")
    L.append(f"Settings   model={s.get('model_source','final')}  mode={s.get('mode')}  "
             f"conf={s.get('conf')}  risk={pc(s.get('risk'))}  "
             f"maxpos={s.get('max_positions')}  "
             f"SL/TP={s.get('atr_sl')}/{s.get('atr_tp')} ATR")
    L.append(f"Realism    intrabar={s.get('intrabar')}  slip={pc(s.get('stop_slippage'),3)}  "
             f"swap L/S={pc(s.get('swap_long'),4)}/{pc(s.get('swap_short'),4)}  "
             f"M1={'yes' if s.get('m1_csv') else 'no'}")
    L.append("-" * 100)
    L.append(f"Trades     {res.get('total_trades',0):,} "
             f"(L {res.get('long_trades',0):,} / S {res.get('short_trades',0):,})  ·  "
             f"WR {pc(res.get('win_rate'),1)}  ·  PF {res.get('profit_factor',0):.2f}  ·  "
             f"avg win {pc(res.get('avg_win'))} / loss {pc(res.get('avg_loss'))}")
    sharpe_val = res.get('sharpe')
    sharpe_txt = (f"{sharpe_val:.2f} [{res.get('sharpe_timeframe', 'unknown')} bars]"
                  if sharpe_val is not None else "n/a (no timestamps)")
    L.append(f"P&L        return {pc(res.get('return_pct'))}  ·  "
             f"final ${res.get('final_balance',0):,.0f}  ·  "
             f"max DD {pc(res.get('max_drawdown'))}  ·  "
             f"sharpe(ann) {sharpe_txt}")

    def num(v, d=2):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
    if any(res.get(k) is not None for k in
           ("sortino", "calmar", "ulcer_index", "equity_r2")):
        L.append(f"Risk-adj   sortino {num(res.get('sortino'))}  ·  "
                 f"calmar {num(res.get('calmar'))}  ·  "
                 f"ulcer {num(res.get('ulcer_index'))}  ·  "
                 f"UPI {num(res.get('upi'))}  ·  "
                 f"equity R2 {num(res.get('equity_r2'), 3)}")
    L.append(f"Streaks    max {res.get('max_consecutive_wins',0)} wins / "
             f"{res.get('max_consecutive_losses',0)} losses in a row  ·  "
             f"avg bars held {num(res.get('avg_trade_bars'), 1)} "
             f"(W {num(res.get('avg_winning_trade_bars'), 1)} / "
             f"L {num(res.get('avg_losing_trade_bars'), 1)})")
    if res.get("largest_win_pct") is not None:
        L.append(f"Extremes   largest win {pc(res.get('largest_win_pct'), 3)} "
                 f"({num(res.get('largest_win_bars'), 0)} bars)  ·  "
                 f"largest loss {pc(res.get('largest_loss_pct'), 3)} "
                 f"({num(res.get('largest_loss_bars'), 0)} bars)  ·  "
                 f"annual vol {pc(res.get('annual_volatility'), 1)}")
    amb = res.get('ambiguous_share_of_sl_tp')
    L.append(f"Execution  ambiguous(SL+TP same bar) {res.get('ambiguous_bars',0)}  ·  "
             f"M1-resolved {res.get('m1_resolved',0)}  ·  "
             f"by-assumption {pc(amb,1)}  ·  "
             f"swap {res.get('swap_rollovers',0)} rollovers ({pc(res.get('swap_net_impact'),3)})")
    rb = res.get("random_baseline")
    if rb:
        L.append(f"Baseline   beats {rb.get('beats')}/{rb.get('runs')} random agents "
                 f"({pc(rb.get('percentile'),0)})  ·  random median {pc(rb.get('random_return_median'))}")
        L.append(f"           -> {rb.get('verdict','')}")
    mc = res.get("monte_carlo")
    if mc:
        L.append(f"MonteCarlo DD median {pc(mc.get('dd_median'))}  ·  "
                 f"95% worst {pc(mc.get('dd_p95_worst_case'))}  ·  "
                 f"worst {pc(mc.get('dd_worst'))}  ·  "
                 f"P(hard stop) {pc(mc.get('p_hit_hard_dd'),1)}")
    sig = res.get("significance")
    if sig:
        L.append(f"Stats      mean/trade {pc(sig.get('mean_per_trade'),4)}  ·  "
                 f"t {sig.get('t_stat',0):.2f}  ·  p {sig.get('p_value_one_sided',0):.3f}  ·  "
                 f"PSR {pc(sig.get('psr'),1)}")
        L.append(f"           -> {sig.get('verdict','')}")
    bs = res.get("bootstrap")
    if bs:
        pf_b = bs.get("profit_factor") or {}
        sh_b = bs.get("sharpe_per_trade") or {}
        dd_b = bs.get("max_dd_conf") or {}
        parts = []
        if pf_b.get("ci95"):
            parts.append(f"PF {pf_b.get('point', 0):.2f} "
                         f"95% CI [{pf_b['ci95'][0]:.2f}, {pf_b['ci95'][1]:.2f}]")
        if sh_b.get("ci95"):
            parts.append(f"Sharpe/trade CI [{sh_b['ci95'][0]:.3f}, "
                         f"{sh_b['ci95'][1]:.3f}]")
        if dd_b.get("0.95") is not None:
            parts.append(f"DD 95% worst {pc(dd_b.get('0.95'))}")
        if parts:
            L.append("Bootstrap  " + "  ·  ".join(parts))
        if bs.get("verdict"):
            L.append(f"           -> {bs['verdict']}")
    ea = res.get("equity_analytics") or {}
    if ea:
        neg = sum(1 for v in (ea.get('monthly_returns') or {}).values() if v < 0)
        tot = len(ea.get('monthly_returns') or {})
        L.append(f"Analytics  underwater {ea.get('longest_underwater_bars',0):,} bars  ·  "
                 f"winners MAE {pc(ea.get('winners_avg_mae'),2)}  ·  "
                 f"losers MFE {pc(ea.get('losers_avg_mfe'),2)}  ·  "
                 f"months neg {neg}/{tot}")
    return L


def report_rows(lines):
    """The same lines split into (section, text) for a table view: 'Trades     1,544 …' ->
    ('Trades', '1,544 …'); '           -> verdict' rows keep an empty section. Text is
    not reformatted, so every number stays exactly what report_lines produced."""
    rows = []
    for i, line in enumerate(lines):
        if i == 0:
            rows.append(["Run", line[len("Model "):] if line.startswith("Model ") else line])
        elif set(line) == {"-"}:
            rows.append(["", ""])
        else:
            m = re.match(r"^(\S+)\s{2,}(.*)$", line) or re.match(r"^(MonteCarlo) (.*)$", line)
            rows.append([m.group(1), m.group(2)] if m else ["", line.strip()])
    return rows
