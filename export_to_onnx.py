"""
export_to_onnx.py — Convert PPO model to ONNX + generate MT5 deployment package
================================================================================

Pipeline:
  1. Load trained PPO model (PyTorch via stable_baselines3)
  2. Wrap policy network with softmax output
  3. Export to ONNX (single-file, MT5-compatible)
  4. Verify ONNX output matches PyTorch
  5. Generate MQL5 config header (norm stats)
  6. Generate customized EA from template (with model name baked in)

Output structure (in --output_dir):
    MQL5/
    ├── Files/
    │   └── <deploy_name>.onnx
    ├── Include/
    │   ├── <deploy_name>_config.mqh
    │   └── RL_Indicators.mqh   (copied from template)
    └── Experts/
        └── <deploy_name>_EA.mq5

Usage:
    python export_to_onnx.py <model_name> [--name <deploy_name>] [--output_dir <path>]

Examples:
    # Default: deploy_name = model_name, output to mt5_files/MQL5/
    python export_to_onnx.py rl_prod_v10_enriched

    # Custom deploy name and output dir
    python export_to_onnx.py rl_prod_v10_enriched --name rl_v10 --output_dir mt5_files/MQL5
"""
import sys, io, argparse, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class PolicyWrapper(nn.Module):
    """Wraps SB3 policy to output softmax probabilities directly."""
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.pi_features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        return torch.softmax(logits, dim=-1)


# Path to template files (relative to this script)
SCRIPT_DIR = Path(__file__).parent.resolve()
TEMPLATE_EA  = SCRIPT_DIR / "mt5_files" / "MQL5" / "Experts" / "ML_RL_Trader_template.mq5"
TEMPLATE_INC = SCRIPT_DIR / "mt5_files" / "MQL5" / "Include"  / "RL_Indicators.mqh"


def export_model(model_name: str, deploy_name: str = None, output_dir: str = None):
    print("=" * 70)
    print(f"  Exporting {model_name} → MT5 deployment package")
    print("=" * 70)

    if deploy_name is None:
        deploy_name = model_name

    if output_dir is None:
        output_dir = SCRIPT_DIR / "mt5_files" / "MQL5"
    else:
        output_dir = Path(output_dir)

    out_files   = output_dir / "Files"
    out_include = output_dir / "Include"
    out_experts = output_dir / "Experts"
    for d in [out_files, out_include, out_experts]:
        d.mkdir(parents=True, exist_ok=True)

    # === Verify templates exist ===
    if not TEMPLATE_EA.exists():
        print(f"❌ Template EA not found: {TEMPLATE_EA}")
        return 1
    if not TEMPLATE_INC.exists():
        print(f"❌ Template indicators not found: {TEMPLATE_INC}")
        return 1

    # === Load PPO model ===
    from stable_baselines3 import PPO
    candidates = [
        f"{model_name}_best/best_model.zip",
        f"{model_name}.zip",
    ]
    model_path = None
    for c in candidates:
        if Path(c).exists():
            model_path = c
            break

    if model_path is None:
        print(f"❌ Model not found. Tried: {candidates}")
        return 1

    print(f"\n[load] {model_path}")
    model = PPO.load(model_path, device='cpu')

    obs_dim = model.policy.observation_space.shape[0]
    n_actions = int(model.policy.action_space.n)
    print(f"  Input dim:  {obs_dim}")
    print(f"  Actions:    {n_actions}")

    # === Wrap policy ===
    wrapped = PolicyWrapper(model.policy)
    wrapped.eval()

    # === Export to ONNX ===
    onnx_filename = f"{deploy_name}.onnx"
    onnx_path = out_files / onnx_filename

    print(f"\n[export] -> {onnx_path}")
    dummy = torch.randn(1, obs_dim, dtype=torch.float32)
    # NOTE: not using dynamic_axes / dynamic_shapes since MT5 always
    # inferences with batch_size=1 (one bar at a time).
    # Fixed shape avoids dynamo deprecation warning + simpler ONNX graph.
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_probs'],
    )

    # === Consolidate external data into single file (MT5 requires this!) ===
    print(f"\n[consolidate] merging external data into single .onnx file...")
    import onnx as onnx_pkg
    loaded = onnx_pkg.load(str(onnx_path), load_external_data=True)
    onnx_pkg.save_model(
        loaded, str(onnx_path),
        save_as_external_data=False,
    )
    data_file = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_file.exists():
        data_file.unlink()
    onnx_size_kb = onnx_path.stat().st_size / 1024
    print(f"  ✅ Single-file ONNX: {onnx_size_kb:.1f} KB")

    # === Verify ONNX matches PyTorch ===
    print(f"\n[verify] comparing PyTorch vs ONNX outputs...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    test_inputs = [
        torch.randn(1, obs_dim, dtype=torch.float32),
        torch.zeros(1, obs_dim, dtype=torch.float32),
        torch.ones(1, obs_dim, dtype=torch.float32),
    ]
    max_diff = 0.0
    for i, t in enumerate(test_inputs):
        with torch.no_grad():
            torch_out = wrapped(t).numpy()
        onnx_out = sess.run(None, {'state': t.numpy()})[0]
        diff = np.abs(torch_out - onnx_out).max()
        max_diff = max(max_diff, diff)
    if max_diff < 1e-5:
        print(f"  ✅ ONNX matches PyTorch (max diff: {max_diff:.2e})")
    else:
        print(f"  ⚠️  Max diff = {max_diff:.2e} — investigate!")

    # === Load norm stats ===
    norm_path = f"{model_name}_norm.csv"
    if not Path(norm_path).exists():
        print(f"\n⚠️  Norm stats not found: {norm_path}")
        return 1

    print(f"\n[norm] {norm_path}")
    norm = pd.read_csv(norm_path, index_col=0)
    feat_count = len(norm)
    extras = 3
    if (obs_dim - extras) % feat_count != 0:
        print(f"⚠️  obs_dim mismatch — cannot determine window size!")
        window = 0
    else:
        window = (obs_dim - extras) // feat_count
    print(f"  Features: {feat_count}, Window: {window}")

    # === Generate config.mqh ===
    config_filename = f"{deploy_name}_config.mqh"
    config_path = out_include / config_filename

    feat_names = list(norm.index)
    means = norm['mean'].values
    stds = norm['std'].values

    mqh = []
    mqh.append("//+------------------------------------------------------------------+")
    mqh.append(f"//| {config_filename} — Auto-generated by export_to_onnx.py")
    mqh.append(f"//| Model: {model_name}")
    mqh.append(f"//| Deploy: {deploy_name}")
    mqh.append(f"//| Input dim: {obs_dim} = window({window}) × features({feat_count}) + 3")
    mqh.append(f"//| Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mqh.append("//+------------------------------------------------------------------+")
    mqh.append(f"#ifndef {deploy_name.upper()}_CONFIG_MQH")
    mqh.append(f"#define {deploy_name.upper()}_CONFIG_MQH")
    mqh.append("")
    mqh.append(f"#define RL_INPUT_DIM     {obs_dim}")
    mqh.append(f"#define RL_OUTPUT_DIM    {n_actions}")
    mqh.append(f"#define RL_WINDOW_SIZE   {window}")
    mqh.append(f"#define RL_FEATURE_COUNT {feat_count}")
    mqh.append("")
    mqh.append("// Feature names (ordered)")
    mqh.append(f"const string RL_FEATURE_NAMES[{feat_count}] = {{")
    for i, name in enumerate(feat_names):
        comma = "," if i < feat_count - 1 else ""
        mqh.append(f"   \"{name}\"{comma}")
    mqh.append("};")
    mqh.append("")
    mqh.append("// Normalization: mean")
    mqh.append(f"const double RL_FEAT_MEAN[{feat_count}] = {{")
    for i, m in enumerate(means):
        comma = "," if i < feat_count - 1 else ""
        mqh.append(f"   {m:+.10f}{comma}  // {feat_names[i]}")
    mqh.append("};")
    mqh.append("")
    mqh.append("// Normalization: std")
    mqh.append(f"const double RL_FEAT_STD[{feat_count}] = {{")
    for i, s in enumerate(stds):
        comma = "," if i < feat_count - 1 else ""
        mqh.append(f"   {s:+.10f}{comma}  // {feat_names[i]}")
    mqh.append("};")
    mqh.append("")

    # === Embed DataCollector params (if sidecar exists) ===
    import json as _json
    params_file = SCRIPT_DIR / f"{model_name}.params.json"
    mqh.append("//=== DataCollector feature-computation params (for parity) ===")
    if params_file.exists():
        try:
            params = _json.loads(params_file.read_text(encoding="utf-8"))
            mqh.append(f"// Embedded from {params_file.name} — applied via")
            mqh.append("// RL_ApplyDataCollectorConfig() (call in OnInit before RL_InitIndicators).")
            mqh.append("")
            INT_KEYS = ("PMIN", "PMAX", "PERIOD", "WINDOW", "FAST", "SLOW", "SIGNAL", "_P")
            BOOL_KEYS = ("CP_Hammer", "CP_Engulfing", "CP_Inside", "CP_Outside",
                         "CP_Star", "CP_Soldiers", "CP_Marubozu", "CP_Harami",
                         "CP_Piercing", "CP_MatHold",
                         "CP_MatholdReqBreak", "CP_InsideStrict")
            def _kind(k, v):
                if isinstance(v, bool): return "bool"
                if isinstance(v, int) or (isinstance(v, float) and v.is_integer()
                                          and any(x in k for x in INT_KEYS)): return "int"
                return "double"
            def _fmt(v, kind):
                if kind == "bool": return "true" if v else "false"
                if kind == "int":  return str(int(v))
                return f"{float(v):.6f}"

            mqh.append("void RL_ApplyDataCollectorConfig()")
            mqh.append("{")
            for k, v in params.items():
                kind = "bool" if k in BOOL_KEYS else _kind(k, v)
                mqh.append(f"   {k} = {_fmt(v, kind)};")
            mqh.append("}")
        except Exception as e:
            mqh.append(f"// WARN: could not parse params.json ({e})")
            mqh.append("void RL_ApplyDataCollectorConfig() { /* no params */ }")
    else:
        mqh.append("// No params.json sidecar — EA will use RL_Indicators defaults.")
        mqh.append("void RL_ApplyDataCollectorConfig() { /* no-op */ }")

    mqh.append("")
    mqh.append(f"#endif // {deploy_name.upper()}_CONFIG_MQH")

    config_path.write_text('\n'.join(mqh), encoding='utf-8')
    print(f"\n[config] -> {config_path}")
    if params_file.exists():
        print(f"[config] embedded params from {params_file.name}")

    # === Generate EA from template ===
    ea_filename = f"{deploy_name}_EA.mq5"
    ea_path = out_experts / ea_filename

    template_text = TEMPLATE_EA.read_text(encoding='utf-8')
    customized = (template_text
        .replace("__MODEL_NAME__",    deploy_name)
        .replace("__CONFIG_HEADER__", config_filename)
        .replace("__ONNX_FILE__",     onnx_filename))
    ea_path.write_text(customized, encoding='utf-8')
    print(f"[EA]     -> {ea_path}")

    # === Copy RL_Indicators.mqh (if not already there) ===
    indicators_dest = out_include / "RL_Indicators.mqh"
    if not indicators_dest.exists() or indicators_dest != TEMPLATE_INC:
        if indicators_dest.exists() and indicators_dest.samefile(TEMPLATE_INC):
            pass  # same file, no copy needed
        else:
            shutil.copy2(TEMPLATE_INC, indicators_dest)
    print(f"[helpers]-> {indicators_dest}")

    # === Summary ===
    print()
    print("=" * 70)
    print(f"  ✅ EXPORT COMPLETE — {deploy_name}")
    print("=" * 70)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── Files/")
    print(f"  │   └── {onnx_filename}        ({onnx_size_kb:.1f} KB)")
    print(f"  ├── Include/")
    print(f"  │   ├── {config_filename}")
    print(f"  │   └── RL_Indicators.mqh")
    print(f"  └── Experts/")
    print(f"      └── {ea_filename}")

    print(f"\n📋 To deploy in MT5:")
    print(f"  1. Open MT5 → File → Open Data Folder")
    print(f"  2. Copy {output_dir}/* contents → MQL5/")
    print(f"     (mirrors structure: Files/, Include/, Experts/)")
    print(f"  3. Open MetaEditor → Experts/{ea_filename}")
    print(f"  4. Compile (F7)")
    print(f"  5. Strategy Tester → select {deploy_name}_EA")

    print(f"\n⚙️  EA Settings to match training:")
    print(f"  • Window:     {window}")
    print(f"  • Confidence: 0.95 (recommended)")
    print(f"  • Symbol/TF:  match training data")

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", help="Model name (without .zip)")
    ap.add_argument("--name", dest="deploy_name", default=None,
                    help="Deployment name (default: same as model_name)")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (default: mt5_files/MQL5/)")
    args = ap.parse_args()
    return export_model(args.model_name, args.deploy_name, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
