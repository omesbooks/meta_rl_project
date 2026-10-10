# Feature screen — `xauusd_h4_2025_v2.csv`

17,593 rows · 182 features · split 70%/15%/15% · IC over whole slices · z = Train IC / moving-block bootstrap SE (250-bar blocks, 200 resamples)
- Train 2014.12.31 20:00 → 2022.09.09 12:00 · Validation → 2024.05.07 00:00 · Test → 2025.12.30 16:00
- constant features dropped: candle_mathold
- non-stationary (drift_z ≥ 2, excluded from stable/ridge): ema_20, ema_50, ema_100, ema_200, ema_long, d1_ema_fast, d1_ema_slow, ex_tema, ex_dema, ex_ama, ex_env_upper, ex_env_lower, ex_obv

## Summary per horizon

| horizon | |z|≥2 (chance) | |z|≥3 (chance) | stable | persistence Train→Val | Train→Test | sign agree Train/Val | ridge IC train / val / test | median |vol IC| |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 1 | 24 (7.7) | 15 (0.5) | 13 | +0.72 | +0.14 | 77% | +0.091 / +0.039 / +0.012 (α 10000) | 0.01 |
| 5 | 7 (7.7) | 2 (0.5) | 2 | +0.42 | -0.18 | 73% | +0.105 / -0.052 / +0.022 (α 10000) | 0.01 |
| 20 | 1 (7.7) | 0 (0.5) | 0 | +0.43 | -0.42 | 81% | +0.148 / -0.068 / +0.057 (α 10000) | 0.01 |

## Top 20 by Train stability (h=5)

| feature | IC train | z | IC val | IC test | stable | vol IC train |
|---|---:|---:|---:|---:|:-:|---:|
| hour | +0.021 | +3.8 | +0.022 | +0.022 | ✓ | -0.053 |
| session_asia | -0.018 | -3.2 | -0.016 | -0.014 | ✓ | +0.060 |
| hour_cos | -0.015 | -2.9 | -0.017 | +0.009 |  | +0.051 |
| ex_chv | +0.031 | +2.4 | -0.010 | -0.036 |  | -0.003 |
| candle_harami | -0.019 | -2.3 | +0.010 | -0.002 |  | +0.002 |
| session_ny | +0.014 | +2.3 | +0.020 | +0.031 |  | -0.045 |
| dow | +0.032 | +2.3 | +0.014 | -0.030 |  | +0.040 |
| dow_sin | -0.028 | -2.0 | +0.003 | +0.033 |  | -0.026 |
| adx_18 | +0.029 | +1.8 | +0.035 | -0.048 |  | +0.024 |
| adx_16 | +0.029 | +1.8 | +0.026 | -0.047 |  | +0.023 |
| adx_20 | +0.029 | +1.7 | +0.042 | -0.048 |  | +0.026 |
| adx_14 | +0.029 | +1.7 | +0.016 | -0.043 |  | +0.022 |
| adx_26 | +0.028 | +1.7 | +0.051 | -0.042 |  | +0.031 |
| adx_28 | +0.028 | +1.7 | +0.051 | -0.041 |  | +0.032 |
| adx_24 | +0.028 | +1.7 | +0.050 | -0.044 |  | +0.030 |
| adx_30 | +0.028 | +1.7 | +0.051 | -0.041 |  | +0.033 |
| adx_22 | +0.028 | +1.7 | +0.046 | -0.046 |  | +0.028 |
| adx_32 | +0.027 | +1.7 | +0.050 | -0.042 |  | +0.034 |
| hour_sin | -0.009 | -1.7 | -0.010 | -0.023 |  | +0.046 |
| adx_12 | +0.028 | +1.6 | +0.005 | -0.035 |  | +0.021 |

Stable in at least one horizon: adx_22, adx_24, adx_26, adx_28, adx_30, adx_32, adx_34, adx_36, candle_inside, hour, session_active_count, session_asia, session_london, session_london_ny_overlap

Full table: `feature_screen_xauusd_h4_2025_v2.csv`
