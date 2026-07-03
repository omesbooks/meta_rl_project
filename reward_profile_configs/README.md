# Reward Profile JSON Recipes

This folder stores reusable reward-profile recipes for training.

Use them from the Train page:

```text
Advanced Reward Settings -> Load JSON
```

Or from CLI:

```powershell
.\.venv\Scripts\python.exe rl_train.py <csv> --name rl_custom --reward_profile_json reward_profile_configs\anti_overtrade_example.json
```

Developer-mode formulas are also supported:

```powershell
.\.venv\Scripts\python.exe rl_train.py <csv> --name rl_formula --reward_profile_json reward_profile_configs\developer_formula_example.json
```

JSON shape:

```json
{
  "schema": "metafxclub.reward_profile.v1",
  "name": "anti_overtrade_example",
  "base_profile": "anti_overtrade",
  "overrides": {
    "trade_penalty": 0.012,
    "unrealized_scale": 0.04
  },
  "developer_mode": {
    "enabled": true,
    "formula": "(unrealized_delta * unrealized_scale) - (trade_penalty if just_opened else 0)"
  },
  "notes": "Optional human note"
}
```

`base_profile` must be one of:

- `balanced`
- `anti_overtrade`
- `low_drawdown`
- `trend_follower`
- `scalper`

Only safe slider keys are accepted in `overrides`; invalid keys or out-of-range values stop training before it starts.

Formula rules:

- Expression only; no imports, assignments, attributes, loops, or file/system access.
- Safe helpers: `abs`, `min`, `max`, `round`, `sqrt`, `log`, `exp`, `tanh`, `clip`, `sign`.
- Available variables include `trade_closed_pnl`, `unrealized_delta`, `just_opened`, `is_idle`, `time_decay_active`, and all reward params.
