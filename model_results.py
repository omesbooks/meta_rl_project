"""Model comparison rows — shared by rl_app.py (Models page) and the web GUI.

Moved verbatim out of RLTradingStudio (_collect_model_rows, _metrics_from_trades,
_equity_path_for_model, the params-status check and the PDF export) so both front ends
list exactly the same rows. Caching stays with the caller.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from artifact_paths import (
    MODELS_DIR,
    backtests_dir,
    find_chart_path,
    find_equity_path,
    find_model_path,
    find_params_path,
    legacy_final_model_path,
    model_names_from_artifacts,
)

ROOT = Path(__file__).parent.resolve()

COLUMNS = ("model", "source", "trades", "win_rate", "profit_factor",
           "return", "max_dd", "realism", "chart", "params")
HEADINGS = {
    "model": "Model", "source": "Result source", "trades": "Trades", "win_rate": "Win rate",
    "profit_factor": "PF", "return": "Return", "max_dd": "Max DD", "realism": "Realism",
    "chart": "Chart", "params": "Params",
}


def fmt_pct(value):
    return "-" if value is None else f"{value:+.2f}%"


def fmt_num(value):
    if value is None:
        return "-"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def metrics_from_trades(trades_path, starting_balance=10000):
    try:
        import pandas as pd
        df = pd.read_csv(trades_path)
        if df.empty:
            return {
                "trades": 0, "win_rate": None, "profit_factor": None,
                "return_pct": None, "max_dd": None,
            }

        if "pnl_dollars" in df.columns:
            pnl = pd.to_numeric(df["pnl_dollars"], errors="coerce").dropna()
            equity = starting_balance + pnl.cumsum()
            return_pct = float(pnl.sum() / starting_balance * 100)
        elif "pnl_pct" in df.columns:
            pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").dropna()
            equity = (1 + pnl).cumprod()
            return_pct = float((equity.iloc[-1] - 1) * 100) if len(equity) else None
        elif "pnl" in df.columns:
            pnl = pd.to_numeric(df["pnl"], errors="coerce").dropna()
            if len(pnl) and pnl.abs().median() < 0.2:
                equity = (1 + pnl).cumprod()
                return_pct = float((equity.iloc[-1] - 1) * 100)
            else:
                equity = starting_balance + pnl.cumsum()
                return_pct = float(pnl.sum() / starting_balance * 100)
        else:
            pnl = pd.Series(dtype=float)
            equity = pd.Series(dtype=float)
            return_pct = None

        if "equity" in df.columns:
            eq = pd.to_numeric(df["equity"], errors="coerce").dropna()
            if len(eq):
                equity = eq
                return_pct = float((eq.iloc[-1] - starting_balance) / starting_balance * 100)

        trades = int(len(pnl)) if len(pnl) else int(len(df))
        win_rate = float((pnl > 0).mean() * 100) if len(pnl) else None
        gross_profit = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
        gross_loss = float(abs(pnl[pnl < 0].sum())) if len(pnl) else 0.0
        profit_factor = float("inf") if gross_profit > 0 and gross_loss == 0 else (
            gross_profit / gross_loss if gross_loss > 0 else None)

        max_dd = None
        if len(equity):
            peak = equity.cummax()
            dd = equity / peak - 1
            max_dd = float(dd.min() * 100)

        return {
            "trades": trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "return_pct": return_pct,
            "max_dd": max_dd,
        }
    except Exception:
        return {
            "trades": None, "win_rate": None, "profit_factor": None,
            "return_pct": None, "max_dd": None,
        }


def equity_path_for_model(model, work_dir=ROOT):
    organized = find_equity_path(model)
    if organized:
        return organized
    work_dir = Path(work_dir)
    for path in [
        work_dir / f"{model}_equity.png",
        work_dir / f"{model}_filtered_equity.png",
    ]:
        if path.exists():
            return path
    return None


def params_status(model, work_dir=ROOT):
    """'✓ sidecar' / '✓ embedded' / '— defaults' — same rule as the Models table."""
    work_dir = Path(work_dir)
    params_sidecar = find_params_path(model)
    config_candidates = [
        work_dir / "mt5_files" / "packages" / model / "MQL5" / "Include" / f"{model}_config.mqh",
        work_dir / "mt5_files" / "MQL5" / "Include" / f"{model}_config.mqh",
    ]
    config_mqh = next((p for p in config_candidates if p.exists()), config_candidates[0])
    if params_sidecar and params_sidecar.exists():
        return "✓ sidecar"
    if config_mqh.exists():
        try:
            config_text = config_mqh.read_text(encoding="utf-8", errors="ignore")
            if ("#define RL_PARAMS_EMBEDDED 1" in config_text or
                    "// Embedded from" in config_text):
                return "✓ embedded"
        except Exception:
            pass
    return "— defaults"


def collect_model_rows(work_dir=ROOT):
    work_dir = Path(work_dir)
    rows = []
    seen = set()
    trade_files = (
        list(work_dir.glob("*_live_bt_trades.csv"))
        + list(work_dir.glob("*_trades.csv"))
        + list(MODELS_DIR.glob("*/backtests/*_live_bt_trades.csv"))
        + list(MODELS_DIR.glob("*/backtests/*_trades.csv"))
        + [p for name in model_names_from_artifacts()
           for p in backtests_dir(name).glob("*_trades.csv")]
    )

    for trades_path in sorted(trade_files, key=lambda p: p.stat().st_mtime, reverse=True):
        if trades_path in seen:
            continue
        seen.add(trades_path)
        name = trades_path.name
        if name.endswith("_live_bt_trades.csv"):
            model = name[:-len("_live_bt_trades.csv")]
            source = "live backtest"
        elif name.endswith("_trades.csv"):
            model = name[:-len("_trades.csv")]
            source = "env backtest"
        else:
            continue

        meta = {}
        if (MODELS_DIR / model / "current.json").exists() and trades_path.parent != backtests_dir(model):
            continue
        if source == "live backtest":
            try:
                from artifact_paths import backtest_meta_path
                mp = backtest_meta_path(model)
                if mp.exists():
                    meta = json.loads(mp.read_text(encoding="utf-8-sig"))
            except Exception:
                meta = {}
            starting_balance = float((meta.get("settings") or {}).get("balance", 10000))
            metrics = metrics_from_trades(trades_path, starting_balance)
            result = meta.get("result") or {}
            if meta.get("status") == "complete" and result:
                metrics["return_pct"] = (float(result["return_pct"]) * 100
                                         if result.get("return_pct") is not None else None)
                metrics["max_dd"] = (float(result["max_drawdown"]) * 100
                                     if result.get("max_drawdown") is not None else None)
            chart_path = find_chart_path(model)
            equity_path = find_equity_path(model)
        else:
            metrics = metrics_from_trades(trades_path)
            chart_path = None
            equity_candidate = trades_path.with_name(f"{model}_equity.png")
            equity_path = equity_candidate if equity_candidate.exists() else None

        # Execution realism of the stored result: did it resolve intrabar
        # SL/TP with M1 data, or lean on the pessimistic/optimistic guess?
        realism = ""
        try:
            if source == "live backtest" and meta.get("status") == "complete":
                bres = meta.get("result") or {}
                amb = bres.get("ambiguous_share_of_sl_tp")
                if bres.get("m1_resolved", 0) > 0:
                    realism = "M1 ✓"
                    if amb:
                        realism += f" + guess {amb:.0%}"
                elif amb is not None:
                    realism = "ok" if amb == 0 else f"guess {amb:.0%}"
        except Exception:
            pass

        rows.append({
            "iid": str(trades_path),
            "model": model,
            "source": source,
            "trades_path": trades_path,
            "chart_path": chart_path,
            "equity_path": equity_path,
            "realism": realism,
            **metrics,
        })

    model_names = {r["model"] for r in rows}
    for model in model_names_from_artifacts():
        if model in model_names:
            continue
        zip_path = find_model_path(model, "final") or legacy_final_model_path(model)
        rows.append({
            "iid": str(zip_path),
            "model": model,
            "source": "model only",
            "trades_path": None,
            "chart_path": None,
            "equity_path": equity_path_for_model(model, work_dir),
            "trades": None,
            "win_rate": None,
            "profit_factor": None,
            "return_pct": None,
            "max_dd": None,
        })

    rows.sort(key=lambda r: (
        r.get("return_pct") is not None,
        r.get("return_pct") if r.get("return_pct") is not None else -1e9,
    ), reverse=True)
    return rows


def table_values(row, work_dir=ROOT):
    """The 10 display cells of one Models-table row (same text as the CTk Treeview)."""
    return (
        row["model"],
        row["source"],
        "-" if row.get("trades") is None else f"{row['trades']:,}",
        "-" if row.get("win_rate") is None else f"{row['win_rate']:.1f}%",
        fmt_num(row.get("profit_factor")),
        fmt_pct(row.get("return_pct")),
        fmt_pct(row.get("max_dd")),
        row.get("realism") or "-",
        "yes" if row.get("chart_path") else "-",
        params_status(row["model"], work_dir),
    )


def export_report_pdf(rows, selected, work_dir=ROOT):
    """Write pipeline_report_<ts>.pdf (table + selected equity curve); returns the path."""
    # Figure objects (not pyplot) so this works from any thread with any GUI backend.
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    work_dir = Path(work_dir)
    out = work_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    selected = selected or rows[0]

    with PdfPages(out) as pdf:
        fig = Figure(figsize=(11.69, 8.27))
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.02, 0.96, "RL Trading Pipeline Report",
                fontsize=20, weight="bold", transform=ax.transAxes)
        ax.text(0.02, 0.92, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
                fontsize=10, color="#555555", transform=ax.transAxes)

        table_rows = []
        for row in rows[:18]:
            table_rows.append([
                row["model"],
                row["source"],
                "-" if row.get("trades") is None else f"{row['trades']:,}",
                "-" if row.get("win_rate") is None else f"{row['win_rate']:.1f}%",
                fmt_num(row.get("profit_factor")),
                fmt_pct(row.get("return_pct")),
                fmt_pct(row.get("max_dd")),
            ])
        table = ax.table(
            cellText=table_rows,
            colLabels=["Model", "Source", "Trades", "WR", "PF", "Return", "Max DD"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.02, 0.08, 0.96, 0.78],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight")

        equity_path = selected.get("equity_path") or equity_path_for_model(selected["model"], work_dir)
        if equity_path and Path(equity_path).exists():
            fig = Figure(figsize=(11.69, 8.27))
            ax = fig.subplots()
            ax.axis("off")
            ax.set_title(f"Equity Curve - {selected['model']}", fontsize=16, weight="bold")
            img = mpimg.imread(equity_path)
            ax.imshow(img)
            pdf.savefig(fig, bbox_inches="tight")
    return out
