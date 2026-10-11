# Live visual — training rollout replay (Level 2), 2026-10-11

## What changed

The 3D viewer used to show only **eval replays**: every 10,000 steps the current
policy was replayed *deterministically* (argmax) on the validation slice. That is
real network data, but it is not what training does — training samples actions,
explores, and only sees the training slice.

Now `rl_train.py --viz` also captures the **actual PPO rollouts**:

| | Training (live) — new | Eval replay — existing |
|---|---|---|
| Data | Train slice, the bars the agent just played | Validation slice |
| Actions | **Sampled** actions from the rollout buffer (exploring) | Argmax (greedy) |
| Extra per bar | env reward, PPO advantage (GAE), greedy action | — |
| Activations | one batched forward pass over the buffer observations, same policy weights that sampled them (callback runs before `train()`) | forward pass along the replay |
| Cadence | after every rollout (`n_steps`, default 2048), at most every 4 s | every 10,000 steps |
| Files | `<run>/viz/rollout_latest.json` (overwritten) + `rollout_meta.json` (polled) | `<run>/viz/snap_NNN.json` + `index.json` |
| Log line | `VIZ_ROLLOUT <seq> <step> <path>` (first + every 10th) | `VIZ_SNAPSHOT <i> <step> <path>` |

A rollout snapshot is the most recent contiguous stretch of one episode inside the
buffer (≤ 400 bars, ≥ 150 if available) so the price chart never jumps across an
episode reset. `test_viz_snapshot.py::test_rollout_callback_writes_what_the_agent_played`
checks the bars are contiguous and that each frame's observation is the window
ending at the bar it is labelled with.

## Viewer

- **source** menu (bottom controls): *Training (live)* / *Eval replay*. A run that
  is still training opens on Training; a finished run (rollout file older than
  2 min) opens on its evals. Scrubbing the eval timeline switches to Eval;
  **● LIVE** goes back to Training.
- Training view adds:
  - coloured **dots above candles** where the sampled action ≠ the greedy argmax
    (dot colour = the action it tried),
  - an **advantage strip** on the chart floor (green up = better than the critic
    expected, pink down = worse; scaled by the 95th percentile of |adv|),
  - tiles **REWARD**, **ADVANTAGE**, **SAMPLED · GREEDY** above the chart,
  - timeline label: rollout #, step, reward Σ over the shown bars, exploring %.
- A new rollout never cuts a replay short: it waits for the current replay to end
  (or loads at once if the window is idle). Because rollouts arrive faster than a
  replay plays, a Training view keeps playing for the whole run — use Dashboard
  mode if CPU matters.
- Dashboard mode stays eval-based (it never downloads rollouts).
- The GUI opens the window on the first `VIZ_ROLLOUT` line (~2k steps) instead of
  waiting for the first evaluation.
- Narrow windows: when the timeline and the controls would collide, the timeline
  stacks above the controls and the chart moves up.

## Cost (measured)

| Item | Value |
|---|---|
| Capture per rollout (pack + JSON + atomic write, n_in 2003, 400 bars, CPU) | median 71 ms |
| Rollout duration at ~290 fps | ~7 s → overhead ≈ 1% |
| `rollout_latest.json` | ~1.4 MB, rewritten ≥ 4 s apart |
| Viewer, Lite, Training source | ~30% of one core (always playing while training) |

Windows note: `os.replace` onto a file the viewer server has open raises
`PermissionError`; `_atomic_write` retries for up to ~1 s.

## Reading it honestly

- Exploration % falls as the policy entropy drops (smoke run `vizroll`: ~60% at
  10k steps → ~35–45% at 35–60k). High exploration late in training means the
  policy is still near-uniform, not that it is "trying ideas".
- Advantage is per-step credit relative to the critic, not profit. A green strip
  on a losing trade can be correct (the outcome was less bad than V(s) predicted).
- Training-slice equity is in-sample and includes exploration noise. It is not
  edge evidence; neither is the validation replay.

Smoke run: `rl_train.py usdjpy_h4_2025_v2.csv --steps 60000 --window 10 --name vizroll --viz`
(run 2317da3febf54cccabf16c0b33b518ec): 30 rollouts written, 6 evals, no viz errors.
