# EA Entry SL/TP Fix

Date: 2026-09-08

## Cause and Change

`OpenPosition()` unconditionally called `CalcSLTP()` and passed its values to both
Buy and Sell. `InpUseSLTP=false` previously affected lot sizing only.

The shared EA template now initializes SL and TP to zero and calls `CalcSLTP()`
only when `InpUseSLTP=true`. A zero SL also supplies zero stop distance to the
existing balance-based lot-sizing fallback, including when ATR data is missing.

Entry stops and action-managed stops remain separate. With a management-capable
model, `InpEnableActionManagement=true` can still add/improve SL later. It does not
create a new TP. Existing open positions are not modified by this fix.

## Corrected Package

`mt5_files/packages/metafxclub_ai_algo_uj_sltp_fix/MQL5/`

- Model: `rl_uj_extra`, final checkpoint, Basic 4, window 20, 183 features.
- EA: `Experts/metafxclub_ai_algo_uj_EA.mq5` and compiled `.ex5`.
- The original package and files installed in MT5 were not replaced.
- Select `Set SL/TP on entry = false` when using the corrected EA if entry stops
  are not wanted. The template default remains true; input settings are not
  inferred from the screenshot or changed automatically.
- Changing source alone does not update a running EA. The corrected binary must
  be installed/reloaded separately. Do not expect this fix to clear existing stops.

## Verification

- `python -m unittest test_ea_entry_stops -v`: 4/4 source-contract checks passed.
  These check the switch guard for both Buy/Sell, unchanged directional ATR
  calculation, zero-distance lot fallback, and independent managed SL behavior.
  They are not broker execution simulations.
- Actual export through `export_to_onnx.py` completed; generated EA passed the
  same entry source-contract check and contains no model-name placeholder.
- Exporter's ONNX/PyTorch comparison passed, maximum difference `5.96e-07`.
- Compiled the exported EA using RoboForex MT5 MetaEditor in a temporary build
  directory: **0 errors, 0 warnings**. Compile log is beside the packaged `.ex5`.
- Binary size: 4,035,982 bytes.
- Binary SHA256: `DF08147C6AE677FBB88D56D0D52D817F009FB1BA28F8CE15F3730FDB553D1155`.
- No Strategy Tester order-execution run or live order test was performed.
- No terminal deployment, order modification, commit, or push was performed.

Compilation used a separate include/resource root as documented by
[MetaEditor command-line integration](https://www.metatrader5.com/en/metaeditor/help/beginning/integration_ide).
Standard MQL libraries were copied only into temporary build storage, not into
the deployment package or over the user's terminal libraries.

## Reproduce

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest test_ea_entry_stops -v
.\.venv\Scripts\python.exe export_to_onnx.py rl_uj_extra --name metafxclub_ai_algo_uj --source final --output_dir mt5_files/packages/metafxclub_ai_algo_uj_sltp_fix/MQL5
```

Re-export changes source/package files, not the existing compiled `.ex5`.
Compile again after any later re-export before deploying that package.
