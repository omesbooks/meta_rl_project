"""
Backtest Chart Generator — Interactive Plotly visualization

Reads trades CSV + price data → produces interactive HTML chart with:
  - Candlestick price chart
  - Buy/Sell entry markers
  - Exit markers (TP / SL / signal / max_hold)
  - SL/TP horizontal lines for each trade
  - Equity curve sub-plot
  - Per-trade P&L bars

Usage:
    python backtest_chart.py rl_v3 training_data_v3.csv

    # With limited bars (for huge datasets)
    python backtest_chart.py rl_v3 training_data_v3.csv --limit 5000

Output: <model>_backtest_chart.html
"""
import sys, io, argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_chart(price_df: pd.DataFrame, trades_df: pd.DataFrame,
                 model_name: str = "model", limit: int = None) -> go.Figure:
    """Build interactive backtest chart"""
    # Parse timestamps
    if 'timestamp' in price_df.columns:
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], errors='coerce')
        price_df = price_df.sort_values('timestamp').reset_index(drop=True)

    # Parse trades timestamps first (needed for smart zoom)
    if 'open_time' in trades_df.columns:
        trades_df['open_time'] = pd.to_datetime(trades_df['open_time'], errors='coerce')
    if 'exit_time' in trades_df.columns:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'], errors='coerce')

    # Smart limit: zoom to where trades actually are
    if limit and len(price_df) > limit and not trades_df.empty and 'open_time' in trades_df.columns:
        trades_min = trades_df['open_time'].min()
        trades_max = trades_df['open_time'].max()

        # Add 5% padding before/after trades
        time_span = trades_max - trades_min
        padding = time_span * 0.05 if time_span.total_seconds() > 0 else pd.Timedelta(days=1)

        zoom_start = trades_min - padding
        zoom_end = trades_max + padding

        # Filter price to this range
        zoomed = price_df[
            (price_df['timestamp'] >= zoom_start) &
            (price_df['timestamp'] <= zoom_end)
        ]

        # If zoomed range fits in limit, use it. Else use last N from zoom range
        if len(zoomed) <= limit:
            price_df = zoomed.reset_index(drop=True)
            print(f"[chart] auto-zoom: {trades_min.date()} → {trades_max.date()} ({len(price_df):,} bars)")
        else:
            price_df = zoomed.tail(limit).reset_index(drop=True)
            print(f"[chart] auto-zoom + limit: showing last {limit:,} of zoom range")
    elif limit and len(price_df) > limit:
        # Fallback: take last N bars
        price_df = price_df.tail(limit).reset_index(drop=True)
        # Filter trades within time range
        if 'open_time' in trades_df.columns:
            min_t = price_df['timestamp'].min()
            trades_df = trades_df[trades_df['open_time'] >= min_t].reset_index(drop=True)

    # === Build figure ===
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price + Trades", "Equity Curve", "Per-trade P&L")
    )

    # === Row 1: Candlestick ===
    fig.add_trace(go.Candlestick(
        x=price_df['timestamp'],
        open=price_df['open'],
        high=price_df['high'],
        low=price_df['low'],
        close=price_df['close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        showlegend=False,
    ), row=1, col=1)

    # === Trade markers ===
    if not trades_df.empty:
        # Long entries (Buy)
        long_trades = trades_df[trades_df['side'] == 'long']
        if not long_trades.empty:
            fig.add_trace(go.Scatter(
                x=long_trades['open_time'],
                y=long_trades['entry'],
                mode='markers',
                name=f"BUY ({len(long_trades)})",
                marker=dict(
                    symbol='triangle-up', size=12,
                    color='#26a69a',
                    line=dict(color='white', width=1)),
                hovertemplate=("<b>BUY</b><br>"
                                "Time: %{x}<br>"
                                "Price: %{y:.5f}<br>"
                                "<extra></extra>"),
            ), row=1, col=1)

        # Short entries (Sell)
        short_trades = trades_df[trades_df['side'] == 'short']
        if not short_trades.empty:
            fig.add_trace(go.Scatter(
                x=short_trades['open_time'],
                y=short_trades['entry'],
                mode='markers',
                name=f"SELL ({len(short_trades)})",
                marker=dict(
                    symbol='triangle-down', size=12,
                    color='#ef5350',
                    line=dict(color='white', width=1)),
                hovertemplate=("<b>SELL</b><br>"
                                "Time: %{x}<br>"
                                "Price: %{y:.5f}<br>"
                                "<extra></extra>"),
            ), row=1, col=1)

        # Exits — colored by reason
        if 'exit_time' in trades_df.columns:
            exit_colors = {
                'TP': '#22c55e',           # green = win
                'SL': '#dc2626',           # red = loss
                'signal': '#f59e0b',       # orange = manual
                'max_hold': '#8b5cf6',     # purple = forced
                'hard_stop': '#000000',    # black = emergency
                'end': '#6b7280',          # gray
            }
            for reason, color in exit_colors.items():
                sub = trades_df[trades_df['reason'] == reason]
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub['exit_time'],
                    y=sub['exit'],
                    mode='markers',
                    name=f"Exit {reason} ({len(sub)})",
                    marker=dict(
                        symbol='x', size=10,
                        color=color,
                        line=dict(color='white', width=1)),
                    hovertemplate=("<b>EXIT (" + reason + ")</b><br>"
                                    "Time: %{x}<br>"
                                    "Price: %{y:.5f}<br>"
                                    "P&L: %{customdata:.4%}<br>"
                                    "<extra></extra>"),
                    customdata=sub['pnl_pct'] if 'pnl_pct' in sub.columns else None,
                ), row=1, col=1)

        # === Connect entry to exit lines ===
        # Draw line for each trade — green if win, red if loss
        for _, tr in trades_df.iterrows():
            if pd.isna(tr.get('exit_time')) or pd.isna(tr.get('open_time')):
                continue
            color = '#22c55e' if tr.get('pnl_pct', 0) > 0 else '#dc2626'
            fig.add_trace(go.Scatter(
                x=[tr['open_time'], tr['exit_time']],
                y=[tr['entry'], tr['exit']],
                mode='lines',
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.4,
                showlegend=False,
                hoverinfo='skip',
            ), row=1, col=1)

    # === Row 2: Equity Curve ===
    if 'pnl_dollars' in trades_df.columns and not trades_df.empty:
        # Cumulative equity
        initial = 10000.0  # default starting balance
        trades_sorted = trades_df.sort_values('exit_time').reset_index(drop=True)
        trades_sorted['cum_equity'] = initial + trades_sorted['pnl_dollars'].cumsum()

        fig.add_trace(go.Scatter(
            x=trades_sorted['exit_time'],
            y=trades_sorted['cum_equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#3b82f6', width=2),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)',
            showlegend=False,
            hovertemplate=("<b>Equity</b><br>"
                            "Time: %{x}<br>"
                            "Balance: $%{y:,.2f}<br>"
                            "<extra></extra>"),
        ), row=2, col=1)

        # Initial balance line
        fig.add_hline(y=initial, line=dict(color='#666', width=1, dash='dash'),
                       row=2, col=1)

    # === Row 3: P&L bars ===
    if not trades_df.empty and 'pnl_pct' in trades_df.columns:
        trades_sorted = trades_df.sort_values('exit_time').reset_index(drop=True)
        colors = ['#22c55e' if p > 0 else '#dc2626' for p in trades_sorted['pnl_pct']]
        fig.add_trace(go.Bar(
            x=trades_sorted['exit_time'],
            y=trades_sorted['pnl_pct'] * 100,  # to %
            name='P&L %',
            marker=dict(color=colors),
            showlegend=False,
            hovertemplate=("<b>Trade P&L</b><br>"
                            "Time: %{x}<br>"
                            "P&L: %{y:.3f}%<br>"
                            "<extra></extra>"),
        ), row=3, col=1)

        # Zero line
        fig.add_hline(y=0, line=dict(color='#666', width=1),
                       row=3, col=1)

    # === Layout ===
    fig.update_layout(
        title=dict(
            text=f"📊 Backtest Visualization — {model_name}",
            font=dict(size=22, color='#e4e4e7'),
            x=0.5),
        template='plotly_dark',
        paper_bgcolor='#0a0e14',
        plot_bgcolor='#1c2128',
        font=dict(color='#e4e4e7', size=11),
        height=900,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            bgcolor='rgba(28, 33, 40, 0.8)',
            bordercolor='#30363d',
            borderwidth=1,
        ),
        xaxis_rangeslider_visible=False,
    )

    # X-axis only show on bottom subplot
    fig.update_xaxes(showgrid=True, gridcolor='#30363d',
                     rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor='#30363d')

    # Sub-plot specific
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="P&L (%)", row=3, col=1)

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model name (without .zip)")
    ap.add_argument("csv", help="price CSV file")
    ap.add_argument("--trades", help="trades CSV (default: <model>_live_bt_trades.csv)")
    ap.add_argument("--limit", type=int, default=5000,
                    help="limit bars (default 5000) — set 0 for all")
    ap.add_argument("--output", help="output HTML path")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  Backtest Chart Generator")
    print("=" * 60)

    # Load price data
    print(f"\n[load] price: {args.csv}")
    price_df = pd.read_csv(args.csv)
    print(f"  rows: {len(price_df):,}")

    # Load trades
    trades_path = args.trades or f"{args.model}_live_bt_trades.csv"
    if not Path(trades_path).exists():
        print(f"\n⚠️  Trades file not found: {trades_path}")
        print(f"   Run backtest first: python backtest_live.py {args.model} {args.csv}")
        return 1

    print(f"[load] trades: {trades_path}")
    trades_df = pd.read_csv(trades_path)
    print(f"  trades: {len(trades_df):,}")

    if 'open_time' in trades_df.columns:
        trades_df['open_time'] = pd.to_datetime(trades_df['open_time'], errors='coerce')
    if 'exit_time' in trades_df.columns:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'], errors='coerce')

    # Build chart
    limit = args.limit if args.limit > 0 else None
    if limit:
        print(f"\n[chart] showing last {limit:,} bars (use --limit 0 for all)")
    else:
        print(f"\n[chart] showing all {len(price_df):,} bars")

    fig = build_chart(price_df, trades_df, args.model, limit=limit)

    # Save
    output = args.output or f"{args.model}_backtest_chart.html"
    fig.write_html(output, include_plotlyjs='cdn')

    print(f"\n[save] -> {output}")
    print(f"  Open in browser to view interactive chart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
