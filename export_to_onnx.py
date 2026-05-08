"""
export_to_onnx.py — Convert PPO model to ONNX for MT5 deployment
==================================================================

Pipeline:
  1. Load trained PPO model (PyTorch via stable_baselines3)
  2. Wrap policy network with softmax output
  3. Export to ONNX format (compatible with MT5 5.0+ OnnxRun)
  4. Verify ONNX output matches PyTorch output
  5. Generate MQL5 header (.mqh) with normalization constants
  6. Print setup instructions

Usage:
    python export_to_onnx.py <model_name>

Example:
    python export_to_onnx.py rl_prod_v10_enriched
    # → produces: rl_prod_v10_enriched.onnx
    # →           rl_prod_v10_enriched_config.mqh
"""
import sys, io, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class PolicyWrapper(nn.Module):
    """Wraps SB3 policy to output softmax probabilities directly.
    Input:  (batch, obs_dim) - observation vector
    Output: (batch, n_actions) - action probabilities
    """
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.pi_features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        # SB3 policy forward pass for actor:
        # obs -> features_extractor -> mlp_extractor.policy_net -> action_net -> logits
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        # Apply softmax to get action probabilities
        return torch.softmax(logits, dim=-1)


def export_model(model_name: str):
    print("=" * 70)
    print(f"  Exporting {model_name} to ONNX")
    print("=" * 70)

    # === Load PPO model ===
    from stable_baselines3 import PPO

    # Try best checkpoint first, fall back to direct
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

    # === Wrap policy with softmax ===
    wrapped = PolicyWrapper(model.policy)
    wrapped.eval()

    # === Export to ONNX ===
    onnx_path = f"{model_name}.onnx"
    dummy = torch.randn(1, obs_dim, dtype=torch.float32)

    print(f"\n[export] -> {onnx_path}")
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_probs'],
        dynamic_axes={
            'state':        {0: 'batch_size'},
            'action_probs': {0: 'batch_size'},
        },
    )

    # === Consolidate external data into single file (MT5 requires this!) ===
    # PyTorch's new dynamo exporter sometimes splits weights into .onnx.data
    # MT5's OnnxCreate() does NOT support external data → must merge
    print(f"\n[consolidate] merging external data into single .onnx file...")
    import onnx as onnx_pkg
    loaded = onnx_pkg.load(onnx_path, load_external_data=True)
    onnx_pkg.save_model(
        loaded, onnx_path,
        save_as_external_data=False,
    )
    data_file = Path(f"{onnx_path}.data")
    if data_file.exists():
        data_file.unlink()
        print(f"  removed: {data_file.name}")
    print(f"  ✅ Single-file ONNX: {Path(onnx_path).stat().st_size / 1024:.1f} KB")

    # === Verify ONNX matches PyTorch ===
    print(f"\n[verify] comparing PyTorch vs ONNX outputs...")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)

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
        print(f"  Test {i+1}: max diff = {diff:.2e}")
    if max_diff < 1e-5:
        print(f"  ✅ ONNX output matches PyTorch (max diff: {max_diff:.2e})")
    else:
        print(f"  ⚠️  Max diff = {max_diff:.2e} — investigate!")

    # === Load norm stats ===
    norm_path = f"{model_name}_norm.csv"
    if not Path(norm_path).exists():
        print(f"\n⚠️  Norm stats not found: {norm_path}")
        print(f"     EA will not be able to normalize features correctly!")
        return 1

    print(f"\n[norm] {norm_path}")
    norm = pd.read_csv(norm_path, index_col=0)
    feat_count = len(norm)
    print(f"  Features: {feat_count}")

    # Calculate window size from input dim
    # input_dim = window * features + 3 (position info)
    extras = 3  # position, unrealized_pnl, bars_in_position
    if (obs_dim - extras) % feat_count != 0:
        print(f"⚠️  obs_dim ({obs_dim}) - 3 not divisible by features ({feat_count})!")
        print(f"     Cannot determine window size automatically")
        window = 0
    else:
        window = (obs_dim - extras) // feat_count
        print(f"  Window size: {window}")

    # === Generate MQL5 config header ===
    mqh_path = f"{model_name}_config.mqh"
    print(f"\n[generate] -> {mqh_path}")

    feat_names = list(norm.index)
    means = norm['mean'].values
    stds = norm['std'].values

    mqh = []
    mqh.append("//+------------------------------------------------------------------+")
    mqh.append(f"//| {model_name}_config.mqh — Auto-generated by export_to_onnx.py |")
    mqh.append(f"//| Model: {model_name}")
    mqh.append(f"//| Input dim: {obs_dim} = window({window}) x features({feat_count}) + 3")
    mqh.append(f"//| Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mqh.append("//+------------------------------------------------------------------+")
    mqh.append("#ifndef RL_CONFIG_MQH")
    mqh.append("#define RL_CONFIG_MQH")
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
    mqh.append("#endif // RL_CONFIG_MQH")

    Path(mqh_path).write_text('\n'.join(mqh), encoding='utf-8')
    print(f"  ✅ Generated {mqh_path}")

    # === Summary ===
    print()
    print("=" * 70)
    print("  EXPORT COMPLETE")
    print("=" * 70)

    onnx_size = Path(onnx_path).stat().st_size / 1024
    mqh_size = Path(mqh_path).stat().st_size / 1024

    print(f"\n  📦 Files generated:")
    print(f"    {onnx_path}        ({onnx_size:.1f} KB)")
    print(f"    {mqh_path}  ({mqh_size:.1f} KB)")

    print(f"\n  📊 Model info:")
    print(f"    Input:    {obs_dim} floats (window={window}, features={feat_count}, +3 pos)")
    print(f"    Output:   {n_actions} probabilities (Hold/Buy/Sell/Close)")
    print(f"    Features: {feat_count}")

    print(f"\n  🚀 Next steps for MT5 deployment:")
    print(f"    1. Copy {onnx_path} to: <MT5>/MQL5/Files/")
    print(f"    2. Copy {mqh_path} to: <MT5>/MQL5/Include/")
    print(f"    3. Copy ML_RL_Trader.mq5 + RL_Indicators.mqh to: <MT5>/MQL5/Experts/")
    print(f"    4. Compile in MetaEditor (F7)")
    print(f"    5. Open Strategy Tester → select EA → Run")

    print(f"\n  ⚙️  Required EA inputs (must match training):")
    print(f"    - Symbol:     EURUSD (or what you trained on)")
    print(f"    - Timeframe:  H4")
    print(f"    - Window:     {window}")
    print(f"    - Conf:       0.95")

    # Print first few features for verification
    print(f"\n  🔍 First 10 features (for verification):")
    for i in range(min(10, feat_count)):
        print(f"    [{i:2d}] {feat_names[i]:25s}  μ={means[i]:+.4f}  σ={stds[i]:+.4f}")

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", help="Model name (without .zip)")
    args = ap.parse_args()
    return export_model(args.model_name)


if __name__ == "__main__":
    sys.exit(main())
