# Feature screen — `usdjpy_h4_2025_base.csv`

17,143 rows · 182 features · split 70%/15%/15% · IC over whole slices · z = Train IC / moving-block bootstrap SE (250-bar blocks, 200 resamples)
- Train 2014.12.31 20:00 → 2022.09.13 00:00 · Validation → 2024.05.07 00:00 · Test → 2025.12.30 16:00
- constant features dropped: candle_mathold
- non-stationary (drift_z ≥ 2, excluded from stable/ridge): ema_20, ema_50, ema_100, ema_200, ema_long, d1_ema_fast, d1_ema_slow, ex_tema, ex_dema, ex_ama, ex_env_upper, ex_env_lower, ex_obv, ex_pvt

## Summary per horizon

| horizon | |z|≥2 (chance) | |z|≥3 (chance) | stable | persistence Train→Val | Train→Test | sign agree Train/Val | ridge IC train / val / test | median |vol IC| |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 1 | 11 (7.6) | 0 (0.5) | 0 | -0.18 | -0.51 | 28% | +0.104 / -0.001 / -0.023 (α 1) | 0.05 |
| 5 | 11 (7.6) | 0 (0.5) | 0 | +0.43 | -0.09 | 44% | +0.101 / -0.000 / +0.003 (α 10000) | 0.03 |
| 20 | 9 (7.6) | 0 (0.5) | 0 | +0.06 | -0.20 | 46% | +0.218 / -0.002 / -0.002 (α 1) | 0.02 |

## Top 20 by Train stability (h=5)

| feature | IC train | z | IC val | IC test | stable | vol IC train |
|---|---:|---:|---:|---:|:-:|---:|
| body_size | +0.021 | +2.5 | -0.002 | -0.022 |  | +0.099 |
| d1_adx | +0.048 | +2.4 | -0.005 | -0.020 |  | +0.084 |
| adx_28 | +0.044 | +2.2 | +0.002 | -0.066 |  | +0.018 |
| adx_30 | +0.045 | +2.2 | +0.003 | -0.062 |  | +0.018 |
| adx_32 | +0.045 | +2.2 | +0.003 | -0.058 |  | +0.018 |
| adx_34 | +0.046 | +2.2 | +0.003 | -0.055 |  | +0.018 |
| adx_26 | +0.043 | +2.2 | +0.001 | -0.070 |  | +0.018 |
| adx_36 | +0.046 | +2.2 | +0.003 | -0.051 |  | +0.019 |
| adx_24 | +0.041 | +2.1 | -0.000 | -0.073 |  | +0.019 |
| candle_outside | -0.016 | -2.1 | +0.017 | -0.010 |  | -0.020 |
| adx_22 | +0.039 | +2.0 | -0.002 | -0.075 |  | +0.020 |
| dow | -0.032 | -1.9 | +0.019 | -0.023 |  | +0.012 |
| hl_range | +0.027 | +1.9 | -0.003 | -0.007 |  | +0.177 |
| adx_20 | +0.035 | +1.9 | -0.004 | -0.077 |  | +0.020 |
| dow_sin | +0.031 | +1.8 | -0.020 | +0.024 |  | -0.010 |
| adx_18 | +0.031 | +1.7 | -0.006 | -0.077 |  | +0.021 |
| atr_6 | +0.032 | +1.6 | -0.016 | -0.051 |  | +0.247 |
| atr_10 | +0.030 | +1.6 | +0.006 | -0.052 |  | +0.261 |
| week_start | +0.023 | +1.6 | +0.018 | +0.015 |  | -0.020 |
| atr_14 | +0.028 | +1.6 | +0.004 | -0.062 |  | +0.263 |

Stable in at least one horizon: none

Full table: `feature_screen_usdjpy_h4_2025_base.csv`
