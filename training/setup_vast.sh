#!/usr/bin/env bash
# DocSmile Vast.ai bootstrap.
# Run ONCE on a fresh Vast.ai instance. ~5-7 minutes total on a good image.
#
# Recommended instance image:
#   pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
#   (or any PyTorch >= 2.2 + CUDA 12.1 image — wheels are CUDA-tagged)
#
# This script INTENTIONALLY does NOT touch the system torch install.
# The Vast.ai image ships a CUDA-matched torch build; reinstalling it can
# silently switch to CPU-only and break training.

set -euo pipefail

echo "============================================================"
echo "DocSmile Vast.ai setup"
echo "============================================================"

# --- 0. Sanity: GPU + CUDA + torch in the compatibility window ---
echo "[0/6] Pre-flight: checking GPU + torch + CUDA on the image..."
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch is not installed on this image. Pick a pytorch:* image.")

if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is not available on this instance.")

ver = torch.__version__
cuda_ver = torch.version.cuda
print(f"  torch          : {ver}")
print(f"  torch.cuda     : {cuda_ver}")
print(f"  device         : {torch.cuda.get_device_name(0)}")
print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")

# Unsloth supports torch 2.4 .. 2.7 (as of 2025.6.x). Block both extremes
# *before* we waste 5 minutes on a doomed pip install.
major, minor = (int(x) for x in ver.split('+')[0].split('.')[:2])
if (major, minor) < (2, 4):
    sys.exit(
        f"ERROR: torch {ver} is older than 2.4 — Unsloth requires >=2.4.\n"
        f"       Destroy this instance and rent one with image:\n"
        f"         pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel"
    )
if (major, minor) > (2, 7):
    sys.exit(
        f"ERROR: torch {ver} is newer than 2.7 — Unsloth and flash-attn don't\n"
        f"       have wheels for it yet. Destroy this instance and rent one with:\n"
        f"         pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel\n"
        f"         (or pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel)"
    )
if cuda_ver and not cuda_ver.startswith("12."):
    sys.exit(
        f"ERROR: CUDA {cuda_ver} is not 12.x — flash-attn / bitsandbytes wheels\n"
        f"       expect CUDA 12.1. Destroy this instance and pick a CUDA 12.x image."
    )
print("  -> torch/cuda are in the supported range.")
PY

# --- 1. Persistent workspace + cache ---
mkdir -p /workspace/.hf_cache
mkdir -p /workspace/.hf_cache/datasets
mkdir -p /workspace/checkpoints
mkdir -p /workspace/runs

export HF_HOME=/workspace/.hf_cache
export TRANSFORMERS_CACHE=/workspace/.hf_cache
export HF_DATASETS_CACHE=/workspace/.hf_cache/datasets

# --- 2. System packages ---
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git tmux htop nano curl wget tree >/dev/null

# --- 3. Python deps from requirements.txt ---
# CRITICAL FLAGS:
#   --no-deps  : the requirements.txt is a hand-curated COMPATIBLE matrix.
#                If pip resolves dependencies, it pulls newer transformers,
#                which then crashes on torch.int1 (needs torch >= 2.6).
#   --force-reinstall : the Vast.ai image may already have older versions of
#                some packages. Force a clean replacement.
echo "[2/6] Installing Python deps (this is the slow part)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install --no-cache-dir --upgrade pip wheel setuptools >/dev/null

# Pre-install shared utility packages with normal dep resolution so common
# transitive deps (numpy, pyarrow, etc.) are present.
pip install --no-cache-dir \
    numpy pandas pyarrow scipy \
    sentencepiece "protobuf>=4.25,<6" \
    "huggingface-hub>=0.26.0,<1" "safetensors>=0.4.5" \
    "tensorboard>=2.18.0" "psutil>=6.0" "nvidia-ml-py>=12.560.30" \
    "pyyaml>=6.0" "tqdm>=4.66" einops packaging filelock requests \
    dill multiprocess xxhash aiohttp "fsspec[http]<=2024.9.0" regex hf_transfer

# Now install the training stack with --no-deps. This locks in the exact
# tested-compatible versions and prevents pip from "upgrading" us into broken
# combinations.
pip install --no-cache-dir --force-reinstall --no-deps \
    -r "${SCRIPT_DIR}/requirements.txt"

# --- 4. FlashAttention 2 (prebuilt wheel) ---
echo "[3/6] Installing FlashAttention 2..."
# Try the prebuilt wheel first. If it doesn't exist for this torch+CUDA combo,
# fall back to plain pip (which may compile from source — slow but works).
if ! pip install --no-cache-dir --no-build-isolation \
       "flash-attn>=2.6.3,<3" 2>/dev/null; then
    echo "  WARN: prebuilt wheel didn't match; falling back to source build."
    echo "        This may take ~20-30 min on first install."
    MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
        "flash-attn>=2.6.3,<3"
fi

# --- 5. Verify final install ---
echo "[4/6] Verifying final install..."
python - <<'PY'
import torch
print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  cuda device    : {torch.cuda.get_device_name(0)}")
print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")
import unsloth
print(f"  unsloth        : {unsloth.__version__}")
try:
    import unsloth_zoo
    print(f"  unsloth_zoo    : {unsloth_zoo.__version__}")
except (ImportError, AttributeError):
    print("  unsloth_zoo    : present (version unknown)")
try:
    import flash_attn
    print(f"  flash_attn     : {flash_attn.__version__}")
except ImportError:
    print("  flash_attn     : NOT INSTALLED  (training will fall back to xformers/sdpa)")
import transformers, trl, peft, datasets, bitsandbytes
print(f"  transformers   : {transformers.__version__}")
print(f"  trl            : {trl.__version__}")
print(f"  peft           : {peft.__version__}")
print(f"  datasets       : {datasets.__version__}")
print(f"  bitsandbytes   : {bitsandbytes.__version__}")
PY

# --- 6. Env vars persisted to bashrc ---
echo "[5/6] Writing env defaults to ~/.bashrc ..."
if ! grep -q "# DocSmile env" ~/.bashrc 2>/dev/null; then
    {
      echo ""
      echo "# DocSmile env"
      echo "export HF_HOME=/workspace/.hf_cache"
      echo "export TRANSFORMERS_CACHE=/workspace/.hf_cache"
      echo "export HF_DATASETS_CACHE=/workspace/.hf_cache/datasets"
      echo "export HF_HUB_DISABLE_TELEMETRY=1"
      echo "export TOKENIZERS_PARALLELISM=true"
    } >> ~/.bashrc
    echo "  (added to ~/.bashrc)"
else
    echo "  (already in ~/.bashrc, skipping)"
fi

# --- 7. Final hints ---
echo "[6/6] Done."
echo ""
echo "============================================================"
echo "  Setup complete."
echo "  Next:"
echo "    1. cp training/.env.example training/.env"
echo "       nano training/.env             # paste your HF_TOKEN"
echo "    2. huggingface-cli login --token \$(grep ^HF_TOKEN training/.env | cut -d= -f2)"
echo "    3. tmux new -s pall"
echo "    4. python training/SFT/sft_train.py --config training/SFT/sft_config.yaml --max-steps 100 --smoke-limit 5000"
echo "       python training/SFT/sft_train.py --config training/SFT/sft_config.yaml"
echo "       (Ctrl-b d  to detach;  tmux attach -t pall  to reconnect)"
echo "    5. In another tmux pane:"
echo "       tensorboard --logdir /workspace/runs --port 6006 --bind_all"
echo "============================================================"
