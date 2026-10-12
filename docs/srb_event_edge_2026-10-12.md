# SRB breakout — event edge test (2026-10-12)

Data: `usdjpy_tds_h1.csv` (RoboForex MT5, custom symbol USDJPY_TDS, 2014-06-02 → 2026-09-02); test window from 2015-01-01; OOS = from 2026-01-01. Twin: `tools/data/srb_features.py` (v02 rf1 defaults).

Units: forward return in **ATR(H1,14)** from the next bar's open, direction-signed, spread deducted; trade results in **R = 1.5 ATR** (the EA's stop). `z` = mean / standard error over events (events treated as independent; SRB fires ~1 per 70 bars so overlap at h ≤ 48 is small).

## 1. Forward return after the event (ATR units)

| event set | n | h=1 | h=2 | h=4 | h=8 | h=24 | h=48 | h=96 |
|---|---:|---|---|---|---|---|---|---|
| SRB signal (all filters, no state) | 1765 | +0.104 ± 0.022 (z +4.7) | +0.135 ± 0.031 (z +4.3) | +0.150 ± 0.044 (z +3.4) | +0.206 ± 0.059 (z +3.5) | +0.178 ± 0.098 (z +1.8) | +0.058 ± 0.142 (z +0.4) | +0.226 ± 0.198 (z +1.1) |
| SRB fired (after day flag + cooldown gates) | 1124 | +0.140 ± 0.029 (z +4.8) | +0.192 ± 0.041 (z +4.7) | +0.280 ± 0.055 (z +5.1) | +0.316 ± 0.076 (z +4.2) | +0.317 ± 0.125 (z +2.5) | +0.343 ± 0.179 (z +1.9) | +0.277 ± 0.247 (z +1.1) |
| null: strong candle + EMA side, NO zone break | 35284 | -0.009 ± 0.004 (z -2.0) | -0.007 ± 0.006 (z -1.2) | -0.004 ± 0.009 (z -0.4) | +0.010 ± 0.013 (z +0.8) | +0.058 ± 0.023 (z +2.5) | +0.044 ± 0.032 (z +1.4) | -0.029 ± 0.047 (z -0.6) |
| null: zone break that FAILED strong/EMA filter | 225 | -0.074 ± 0.050 (z -1.5) | -0.033 ± 0.080 (z -0.4) | -0.146 ± 0.107 (z -1.4) | -0.160 ± 0.143 (z -1.1) | -0.326 ± 0.245 (z -1.3) | -0.223 ± 0.345 (z -0.6) | -0.833 ± 0.501 (z -1.7) |
| null: random bar, direction of that bar | 20000 | -0.007 ± 0.005 (z -1.3) | -0.013 ± 0.008 (z -1.6) | -0.009 ± 0.011 (z -0.8) | -0.002 ± 0.017 (z -0.1) | -0.010 ± 0.029 (z -0.3) | -0.031 ± 0.043 (z -0.7) | -0.082 ± 0.062 (z -1.3) |

Stability of the SRB signal by period (h=4 / h=24 / h=48):

| period | n | h=4 | h=24 | h=48 |
|---|---:|---|---|---|
| 2015–2018 | 617 | +0.154 ± 0.073 (z +2.1) | +0.128 ± 0.163 (z +0.8) | +0.263 ± 0.230 (z +1.1) |
| 2019–2022 | 604 | +0.172 ± 0.076 (z +2.3) | +0.314 ± 0.167 (z +1.9) | +0.265 ± 0.247 (z +1.1) |
| 2023–2025 | 444 | +0.102 ± 0.082 (z +1.2) | +0.153 ± 0.205 (z +0.7) | -0.277 ± 0.281 (z -1.0) |
| 2026–2026 (OOS) | 100 | +0.206 ± 0.230 (z +0.9) | -0.226 ± 0.374 (z -0.6) | -0.969 ± 0.663 (z -1.5) |

## 2. Same entries, simple exits (result in R per event; SL always 1R = 1.5 ATR)

| exit rule | all n | mean R | z | win% | PF | 2015–2025 mean R | 2026 (OOS) mean R |
|---|---:|---:|---:|---:|---:|---:|---:|
| market exit after 4 bars, SL 1R | 1765 | +0.120 | +4.8 | 50% | 1.36 | +0.122 | +0.087 |
| market exit after 24 bars, SL 1R | 1765 | +0.169 | +3.7 | 35% | 1.27 | +0.182 | -0.045 |
| TP 1.25R (= o1 leg), no time stop | 1765 | +0.147 | +5.5 | 51% | 1.30 | +0.154 | +0.035 |
| TP 2R | 1765 | +0.171 | +4.9 | 39% | 1.28 | +0.186 | -0.070 |
| TP 3R | 1765 | +0.149 | +3.5 | 29% | 1.21 | +0.165 | -0.120 |
| TP 2R + breakeven at 0.5R (≈ runner) | 1765 | +0.163 | +6.4 | 22% | 1.58 | +0.171 | +0.020 |
| TP 3R + breakeven at 0.5R | 1765 | +0.155 | +5.2 | 15% | 1.55 | +0.160 | +0.060 |

Cost sensitivity (the TDS bars carry a 0.10-pip spread and the tester charged no commission; a live raw-spread account pays about 0.5–1.5 pips round-trip incl. commission):

| extra round-trip cost | 4-bar exit mean R | TP 1.25R mean R | TP 2R + BE mean R |
|---|---:|---:|---:|
| 0.0 pips | +0.120 | +0.147 | +0.163 |
| 0.5 pips | +0.095 | +0.118 | +0.140 |
| 1.0 pips | +0.069 | +0.099 | +0.108 |
| 1.5 pips | +0.049 | +0.080 | +0.088 |
| 2.0 pips | +0.027 | +0.056 | +0.054 |

## 3. What the EA actually did with these entries (tester report, per signal, in R)

Signals 1107, o1 = 20% of risk at TP 1.25R, o2 = 80% runner (TP measured move, breakeven at 0.5R). Median hold 4.1 h.

- **o1**: mean +0.193 R (z +5.6), win 53%, breakeven 0%, loss 47%, PF 1.41
- **o2**: mean +0.165 R (z +5.7), win 40%, breakeven 32%, loss 28%, PF 1.59
- **risk-weighted per signal**: mean +0.171 R (z +6.0), PF 1.55

| period | n | mean R / signal | z |
|---|---:|---:|---:|
| 2015–2018 | 398 | +0.159 | +3.4 |
| 2019–2022 | 372 | +0.196 | +4.0 |
| 2023–2025 | 273 | +0.148 | +2.6 |
| 2026–2026 (OOS) | 64 | +0.190 | +1.6 |

## 4. Inside the events: do the zone features predict the outcome? (Spearman, event bars only)

| feature at the signal bar | n | ρ vs fwd h=8 | ρ vs fwd h=24 | ρ vs R (TP2R+BE) |
|---|---:|---:|---:|---:|
| touches_broken | 1765 | -0.025 (z -1.1) | -0.032 (z -1.4) | -0.025 (z -1.1) |
| room_ahead_atr | 594 | -0.011 (z -0.3) | +0.011 (z +0.3) | +0.007 (z +0.2) |
| candle_range_atr | 1765 | +0.004 (z +0.2) | -0.024 (z -1.0) | -0.002 (z -0.1) |
| close_loc_signed | 1765 | -0.002 (z -0.1) | -0.005 (z -0.2) | -0.017 (z -0.7) |
| ema_dist_signed | 1765 | +0.068 (z +2.8) | +0.040 (z +1.7) | +0.026 (z +1.1) |
| zone_tol_atr | 1765 | -0.011 (z -0.5) | -0.018 (z -0.8) | -0.019 (z -0.8) |
| n_res | 1765 | -0.010 (z -0.4) | -0.008 (z -0.3) | -0.006 (z -0.3) |
| n_sup | 1765 | -0.020 (z -0.8) | -0.024 (z -1.0) | -0.009 (z -0.4) |

ρ is the rank correlation between the feature and the outcome across events; |z| ≥ 3 is the bar used in `feature_screen.py`.

## 5. Why the 183-feature screen could not have found this

The SRB condition is true on 1765 of 72717 H1 bars (2.43%). A flag that is 0 on 97.6% of bars has a whole-slice rank correlation with forward return of at most a few hundredths even when the conditional mean is large, so `feature_screen.py` (unconditional IC) is the wrong instrument for event features; this file is the conditional test.
