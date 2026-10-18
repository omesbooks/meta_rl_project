# Monitor: Train vs Validation gap + Decision Stability (2026-10-18)

Why: the training-rollout candle replay shows SAMPLED (exploring) actions on Train bars that the
agent revisits in random order, so it cannot say whether the model is learning something real.
Two questions the user actually decides on were missing: *is it starting to overfit?* and *has the
model's reading of the same chart settled?*

## What is measured

After every evaluation (every 10,000 steps) `rl_train.py --viz` walks the current policy
**greedily** (argmax, no exploration) over two **fixed** slices of equal length
(`min(len(validation) − window − 2, 1500)` bars, costs included):

| slice | what it is | role |
|---|---|---|
| validation | the unseen validation slice, from its first bar | out-of-sample behaviour |
| train tail | the last bars of Train (closest in time to validation) | in-sample behaviour of the same policy |

`snapshot.greedy_walk` returns per bar: action probabilities, greedy action, position, equity,
closed-trade pnl, reward. `walk_stats` reduces each walk to return %, PF (from gross win/loss in bp —
never `Infinity` in the JSON), win rate, trades, reward per bar, max DD. Cost: ~1 ms per bar on CPU,
so ≤ 3 s per evaluation at the 1500-bar cap.

## Train vs Validation card

- `index.json` entry: `gap = {train, val, frames, verdict, walk_s}`.
- Small multiples with both series: slice return %, gap (train − validation, pp), PF (capped at 5
  on the chart), win rate, trades, reward per bar. Gold = train, cyan = validation.
- Verdict (`snapshot.gap_verdict`, last 4 evaluations, thresholds in the chip text):

| chip | rule |
|---|---|
| GAP WIDENING — OVERFIT? (red) | train moved ≥ +0.5 pp and validation ≤ −0.5 pp over the window |
| GENERALISING (green) | validation moved ≥ +0.5 pp |
| VALIDATION PEAKED AT EVAL k (amber) | the best validation eval is ≥ 3 evals back and ≥ 1 pp above now |
| MEMORISING THE TRAIN SLICE (amber) | train − validation ≥ 5 pp while validation ≤ 0 |
| NO CLEAR TREND (grey) | none of the above |
| COLLECTING (grey) | fewer than 3 evaluations |

`best/best_model.zip` (EvalCallback) already keeps the policy of the best validation reward, so
"peaked" is actionable without retraining.

## Decision Stability card

- `<run>/viz/stability.json`: `bars` (ts/o/h/l/c of the validation slice, stored once), `actions`,
  `evals[]` with `p` (uint8 probabilities per bar × action), `greedy`, `pos`, `changed` (share of
  greedy decisions that differ from the previous evaluation), `confidence` (mean max probability),
  `entropy`, `in_position`, `trades`, `return_pct`; top-level `settled_since` (first eval after which
  every later eval changed ≤ 10%, needs 2 later evals) and `settled_thresh`.
- Heatmap: rows = evaluations (oldest on top), columns = bars, colour = greedy action
  (hold grey, buy green, sell magenta, close gold), brightness = that action's probability.
  A run that keeps repainting the same bars has not settled.
- Shaded chart: the close line of the slice with the background shaded by P(buy) (green) and
  P(sell) (magenta) of the evaluation picked with the slider (default: latest), plus a
  long/short position strip. Dragging the slider shows how the reading of the same chart changed.
- KPIs: changed-mind %, settled since, confidence, entropy, in-a-position %.

## Caveats

- Both walks are greedy; the training rollouts (sampled actions) stay a different, noisier view.
- The train tail is in-sample by construction; the gap is a trend signal, not an edge estimate.
  Edge still comes from `backtest_live.py` (random baseline, Monte Carlo, M1 replay).
- 1500-bar cap: a longer validation slice is truncated from its start for the walk (the
  EvalCallback reward still covers the whole slice).
- Runs trained before 2026-10-18 have neither `gap` entries nor `stability.json`.

Tests: `test_viz_snapshot.py::test_greedy_walk_stability_rows_and_file`, `::test_gap_verdict_rules`.
