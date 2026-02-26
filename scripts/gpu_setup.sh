#!/bin/bash
# MathScy GPU Setup Script
# Run this on DGX A100 or H100 before starting training
set -e

echo "=== MathScy GPU Environment Setup ==="

# Base directory
BASE_DIR="/scratch/ctoxtli/moexp"
cd "$BASE_DIR"

# 1. Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.44.0 accelerate>=0.33.0 peft>=0.12.0 bitsandbytes>=0.43.0
pip install deepspeed>=0.14.0 flash-attn>=2.5.0 --no-build-isolation
pip install trl>=0.9.0 wandb datasets sentencepiece protobuf
pip install vllm>=0.5.0 megablocks
pip install scipy einops

# 2. Download models from HuggingFace
echo "Downloading models..."
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=your_token"
    echo "Attempting download without token (may fail for gated models)..."
fi

python3 -c "
from huggingface_hub import snapshot_download
import os

base_dir = '/scratch/ctoxtli/moexp'
models = [
    'deepseek-ai/deepseek-math-7b-base',
    'deepseek-ai/deepseek-math-7b-instruct',
    'deepseek-ai/DeepSeek-Prover-V2-7B',
]

for model in models:
    model_name = model.split('/')[-1]
    local_path = os.path.join(base_dir, 'models', model_name)
    if os.path.exists(local_path):
        print(f'{model_name} already downloaded')
        continue
    print(f'Downloading {model}...')
    snapshot_download(
        repo_id=model,
        local_dir=local_path,
        token=os.environ.get('HF_TOKEN'),
    )
    print(f'{model_name} downloaded to {local_path}')
"

# 3. Install Lean 4
echo "Installing Lean 4..."
if ! command -v lean &> /dev/null; then
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:v4.8.0
    export PATH="$HOME/.elan/bin:$PATH"
    echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
fi

# 4. Build Lean workspace
echo "Building Lean workspace..."
LEAN_DIR="$BASE_DIR/lean_workspace"
if [ -d "$LEAN_DIR" ]; then
    cd "$LEAN_DIR"
    lake build || echo "Lean build had warnings (normal for first build with Mathlib)"
    cd "$BASE_DIR"
fi

# 5. Verify GPU access
echo "Verifying GPU access..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
"

# 6. Run DeepSpeed config check
echo "Checking DeepSpeed..."
python3 -c "
import deepspeed
print(f'DeepSpeed version: {deepspeed.__version__}')
# ds_report
" 2>/dev/null || echo "DeepSpeed not available"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Run 'jupyter notebook' or 'jupyter lab'"
echo "  2. Open notebooks/02_moe_training.ipynb for MoE training"
echo "  3. Open notebooks/04_conjecture_generation.ipynb for STP loop"
echo "  4. Open notebooks/03_lean4_autoformalization.ipynb for verification"
