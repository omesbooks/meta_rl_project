# 🧠 Meta RL Trading Project

End-to-end Reinforcement Learning pipeline for forex trading (EUR/USD) using PPO.
จากการเก็บข้อมูล MT5 → train RL agent → backtest → deploy.

---

## 📋 Features

- 🎮 **Modern GUI** (`rl_app.py`) — All-in-one Studio with Train, Backtest, Walk-forward, Fine-tune
- 📊 **Data Pipeline** — Import from MT5, auto-clean redundant features, train/test split
- 🤖 **RL Trainer** — PPO with custom TradingEnv, V4 reward shaping (Net-PnL aware)
- 📈 **Backtest** — Production-grade simulation matching live_trader.py
- 🔬 **Feature Engineering** — Multi-timeframe + volatility regime features
- 📉 **Walk-forward Validation** — Robustness testing
- 🔄 **Smart Fine-tuning** — Continue training with mixed old+new data
- 📊 **Visualization** — Interactive Plotly charts, correlation heatmaps

---

## 🏗️ Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│  MT5 Strategy Tester│ ──▶│  DataCollector_v3.mq5│
│  (historical data)  │     │  (export CSV)        │
└─────────────────────┘     └──────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────┐
│  rl_app.py GUI                                       │
│  ┌────────────┬───────────┬──────────┬───────────┐  │
│  │ Tools      │ Train     │ Backtest │ Walk-fwd  │  │
│  │ - Import   │ - PPO     │ - Pure   │ - 5 fold  │  │
│  │ - Split    │ - V4 rwrd │ - SLTP   │ - Robust? │  │
│  │ - Clean    │ - Live    │ - Charts │           │  │
│  │ - Corr     │   metrics │          │           │  │
│  └────────────┴───────────┴──────────┴───────────┘  │
└─────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ live_trader.py   │
                            │ (production MT5) │
                            └──────────────────┘
```

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run GUI
```bash
python rl_app.py
```

### Or use CLI

**Train:**
```bash
python rl_train.py training_data.csv --steps 200000 --name rl_v1
```

**Backtest:**
```bash
python backtest_live.py rl_v1 test_data.csv --mode pure_agent
```

**Feature Engineering (Phase A):**
```bash
python feature_engineer.py training_data_h4.csv
# → adds Multi-TF (D1) features + Volatility regime
```

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| `rl_app.py` | **Main GUI** — CustomTkinter modern interface |
| `trading_env.py` | Custom Gymnasium env (V4 reward, random episode start) |
| `rl_train.py` | PPO trainer with full hyperparameter control |
| `backtest_live.py` | Production-grade backtest matching `live_trader.py` |
| `feature_engineer.py` | Multi-TF + Volatility regime feature pipeline |
| `rl_finetune.py` | Smart fine-tuning with old/new data mixing |
| `rl_walkforward.py` | Walk-forward validation (5-fold robustness) |
| `live_trader.py` | Production trader (MT5 live) |
| `DataCollector_v3.mq5` | MT5 EA for data collection |

---

## 🔬 Reward Function (V4)

The reward is engineered to align with **real $ profit** (not gameable):

```python
reward = trade_closed_pnl × 50          # real net P&L dominates
       + bonus(0.01 if pnl > 0.5%)      # only meaningful wins
       + unrealized_delta × 0.1         # tiny direction hint
       - 0.001 * give_back_penalty      # encourage closing on peak
       - 0.005 * trade_penalty          # cost-aware (no over-trading)
       - 0.0001 * idle_penalty          # avoid all-Hold
```

Spread + commission baked into entry/exit prices (matches live broker).

---

## 📊 Project Journey

| Version | Setup | PF | Notes |
|---------|-------|-----|-------|
| V2-V6 | H1, buggy random start | 0.66-0.75 | 1.8% data coverage bug |
| V7 | H1 fixed start | <0.85 | Real edge weak |
| V8 | H4 standard features | <0.85 | Same wall |
| V9 | H4 fine-tune | ~0.95 | Small improvement |
| **V10** | **H4 + Multi-TF features** | **0.99** | **Near break-even!** |

Key insight: `d1_rsi` (daily RSI) has correlation 0.138 with future returns — 3.4× stronger than any standard indicator.

---

## 🎯 Lessons Learned

1. **Random episode start is critical** — Fixed start = agent memorizes 1.8% of data
2. **Reward hacking is real** — Reward bonuses must be threshold-based, not flat
3. **Multi-timeframe features add edge** — Daily context filters H4 noise
4. **Train reward ≠ test PF** — Always backtest on out-of-sample
5. **Cost matters** — Spread + commission must be in reward signal honestly

---

## 🛠️ Tools / Scripts

- `feature_engineer.py` — Phase A feature engineering
- `generate_metrics_pptx.py` — RL metrics slide deck generator
- `generate_training_slides.py` — How-training-works visual guide
- `quarterly_update.py` — Auto-retrain pipeline
- `rl_analyze.py` — Per-trade decision analysis

---

## 📦 Data Format

Training CSV must contain (after import):
- `timestamp, symbol, open, high, low, close, volume`
- 30-50 numeric features (RSI, EMA, ATR, etc.)
- `future_return, target` (auto-generated)

Use `DataCollector_v3.mq5` to export from MT5 with standard indicator config.

---

## 🤝 Acknowledgements

Built iteratively over multiple weeks of experimentation with:
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [gymnasium](https://gymnasium.farama.org/)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [Plotly](https://plotly.com/)

---

## 📝 License

Private research project. Use at your own risk.

> ⚠️ **Disclaimer**: This is research code. Trading involves substantial risk of loss.
> Past performance does not indicate future results. Do not deploy without extensive validation.
