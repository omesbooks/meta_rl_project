# Gated entries + continuous stop-loss head — results (2026-10-13)

Follow-up to `docs/srb_rl_ab_2026-10-12.md` (SRB columns as plain features: the agent never used
the sparse signal, PF ~0.9–1.1). Two design changes were tested on the same H1 USDJPY data:

- **Gated entries** (`rl_train --entry_gate srb_sig`, commit a8c48ef): a buy can open only on a
  bar with `srb_sig = +1`, a sell only on `−1`. The EA supplies the timing; RL decides accept/skip
  and when to exit (close / max_hold 24).
- **Continuous stop-loss head** (`--sl_mode continuous`, commits cdc4f27 / 92c4d56 / 42e766a):
  `hybrid_policy.HybridPolicy` adds a Gaussian head that outputs the stop distance in ATR
  (0.5–3.0 × MT5-style ATR(14)); the env sets the stop at entry and fills it intrabar.

Setup as before: `usdjpy_tds_h1_srb.csv` (41 features, 2015-01 → 2026-09), split 70/15/15,
Test 2024-11-29 → 2026-09-02 (252 SRB signals), window 15, balanced reward, basic_4, 1 pip
round-trip, backtest `pure_agent` (no SL unless the model provides one), best checkpoint on
Validation, 3 seeds × 200k steps.

## Results (Test slice)

Benchmarks on the same 252 signals: every signal, exit after 4 bars, no stop → **PF 1.35,
+1,077 pips**; the same with SL 1.5 ATR → PF 1.28.

| arm | seed | trades (% of signals) | PF [95% CI] | net pips | hold | exits |
|---|---:|---:|---|---:|---:|---|
| G0 gated, no SL | 0 | 47 (19%) | 2.35 [1.03, 6.26] | +571 | 8.4 | 43 close / 4 max_hold |
| | 1 | 41 (16%) | 1.71 [0.71, 3.76] | +242 | 4.2 | 40 / 1 |
| | 2 | 101 (40%) | 2.12 [1.22, 3.92] | +1,154 | 5.0 | 100 / 1 |
| | pooled | 189 | **2.09**, CI lo > 1 on 2/3 | | | |
| G1 gated + continuous SL | 0 | 77 (31%) | 1.09 [0.63, 2.01] | +89 | 5.4 | 51 close / 23 SL / 3 |
| | 1 | 63 (25%) | 1.38 [0.74, 2.80] | +257 | 4.9 | 47 / 14 / 2 |
| | 2 | 52 (21%) | 1.97 [0.98, 4.13] | +417 | 3.8 | 44 / 7 / 1 |
| | pooled | 192 | **1.35**, CI lo > 1 on 0/3 | | | |

Stops chosen by the G1 head: mean 1.72 / 1.63 / 1.61 ATR with sd 0.13 / 0.18 / 0.24 — a
near-constant value next to the 1.75 initialisation (min 1.13, max 2.16 over all trades).

## Does the filter have skill? (accepted vs skipped signals under the same fixed 4-bar rule)

| model | accepted: PF / mean R | skipped: PF / mean R | verdict |
|---|---|---|---|
| G0 s0 | 1.78 / +0.22 | 1.28 / +0.10 | picks better signals |
| G0 s1 | 0.66 / −0.15 | 1.53 / +0.18 | picks WORSE signals; profit came from its exits |
| G0 s2 | 2.42 / +0.38 | 0.89 / −0.05 | picks better signals |
| G1 s0 | 0.81 / −0.09 | 1.74 / +0.22 | worse |
| G1 s1 | 0.85 / −0.06 | 1.55 / +0.19 | worse |
| G1 s2 | 1.35 / +0.16 | 1.35 / +0.12 | neutral |

What is consistent across all six models is the **exit timing**: nearly every trade is closed
by the agent before max_hold (hold 4–8 bars) and the realised PF is above the fixed-rule PF of
the same accepted signals in every G0 seed. Signal selection is not consistent (2 of 3 G0 seeds
have skill, 1 is anti-skilled), so the pooled PF 2.09 overstates what a single model would give.

## Reading

1. Gated entries fix the failure of the first A/B: all six gated models trade only on EA bars
   and reach PF 1.1–2.4 instead of ~1.0. The EA signal plus RL-timed exits is where the value is.
2. The continuous SL head did not add value here: pooled PF 1.35 vs 2.09 without it, the head
   learned an almost constant 1.6–1.7 ATR (expected with ~770 training events: most trades end by
   the agent's own close before the stop matters, so the head receives little gradient), and the
   stops that do fire (7–23 per model) take losses the agent would otherwise have managed.
   A fixed 1.5–1.7 ATR stop in the EA reproduces what the head learned.
3. Caveats: 41–101 Test trades per model; PF CIs are wide; the Validation slice is flat for the
   SRB event, so best-checkpoint selection is noisy; net pips beat the all-signals benchmark in
   only 1 of 3 G0 seeds because the agent takes 16–40% of the signals.

## Decision points (open)

- Phase 3/4 of the continuous SL (ONNX 2 outputs, EA, live_trader, GUI): not justified by this
  result; the code stays (tested) for a case with far more events.
- To firm up the gated design: more seeds (5–10) of G0; a variant that forces entry on every
  signal and lets RL time only the exit (tests the exit-timing hypothesis directly); a G0 arm with
  a fixed 1.5 ATR stop as an env rule.
- Deployment of a G0-style model needs the SRB zone/signal logic in MQL (indicator + EA gate);
  the discrete export path itself already works.

Artifacts: `docs/ablation_h1_gate.md`, `docs/ablation_h1_gate_sl.md`; models `h1gate_C_w15_t100_s*`,
`h1gsl_C_w15_t100_s*` (gitignored). Diagnostic script: scratchpad `gate_diag.py` (accepted vs
skipped signals under `srb_event_edge.exit_sim`).

## Addendum — H4 + D1 context block (same day)

`build_srb_h1_dataset.py --ctx` appends 22 lag-safe higher-TF columns (`tools/data/htf_context.py`:
EMA50/200 distance + slope, RSI14, Wilder ADX14, ATR pct/ratio, 20-bar range position, ret 1/5,
trend, on H4 and on D1, each H1 bar seeing only higher bars that had closed). G0 recipe otherwise unchanged.

| arm | features | obs | seed 0 | seed 1 | seed 2 | pooled PF | CI lo > 1 |
|---|---:|---:|---|---|---|---:|---:|
| G0 | 41 | 618 | 2.35 (47 tr, +571 pips) | 1.71 (41, +242) | 2.12 (101, +1,154) | 2.09 | 2/3 |
| G0-ctx | 63 | 948 | 1.36 (85, +441) | 1.00 (60, −7) | 0.53 (59, −572) | 0.96 | 0/3 |

Accepted-vs-skipped under the fixed 4-bar rule: ctx s0 1.52 / 1.26, s1 0.87 / 1.55, s2 1.22 / 1.40 —
and in s2 the agent's own exits turned a +1.22 selection into PF 0.53.

Why context hurt (conditional test on all 1,765 signal bars, outcome = 4-bar exit in R at 1 pip):
no context, zone or base feature reaches |z| ≥ 3 against the per-signal outcome; the strongest are
`rsi_14` (+0.067, z 2.8), `ema50_dist_atr` (+0.062, z 2.6), `d1_ema200_dist` (+0.060, z 2.5). The
per-signal outcome has mean +0.06 R and sd 1.23 R (signal-to-noise 0.05 per event); with 1,246 Train
signals the smallest detectable rank correlation at z = 3 is ~0.085, so there is nothing for a
selection policy to learn from these inputs, and 22 × 15 = 330 extra window inputs (carrying only
~22–44 distinct values, since H4 repeats 4× and D1 24× inside the window) only let PPO fit noise.
The EA's own use of H4 is a single object — the zone — which G0 already had through the SRB block;
the EA never uses H4/D1 trend or volatility.

Same direction as the earlier 28 → 41 step: on this problem, fewer inputs train better.

## Addendum 2 — lean feature set (base 28 + `srb_sig` only)

`usdjpy_tds_h1_lean.csv` = the 41-feature set minus the 12 zone columns (dist/touches/room/brk/n_zones);
same rows, G0 recipe, gated.

| arm | features | obs | seed 0 | seed 1 | seed 2 | pooled PF | CI lo > 1 |
|---|---:|---:|---|---|---|---:|---:|
| G0-lean | 29 | 438 | 1.77 (112 tr, +690 pips) | 1.35 (32, +145) | 0.93 (67, −71) | 1.31 | 1/3 |
| G0 | 41 | 618 | 2.35 (47, +571) | 1.71 (41, +242) | 2.12 (101, +1,154) | 2.09 | 2/3 |
| G0-ctx | 63 | 948 | 1.36 (85, +441) | 1.00 (60, −7) | 0.53 (59, −572) | 0.96 | 0/3 |

Accepted-vs-skipped (fixed 4-bar rule): lean s0 1.48 / 1.27, s1 0.80 / 1.47, s2 1.32 / 1.37.

"Fewer inputs = better" does not hold below 41: dropping the zone columns lands at the all-signals
benchmark (1.31 vs 1.35). The zone geometry (distance to the next level, room, touches) is plausibly
what the exit-timing policy uses (take profit near the next level); the ctx block, which carries no
such information, only added noise. Across the four gated arms the per-seed spread (0.5–2.4) is as
large as the between-arm differences, so G0's pooled 2.09 should be read as "best of four noisy
arms", not as an established edge; a 5–10-seed run of G0 (and a forced-entry, exit-only variant) is
the next step that would separate the two.
