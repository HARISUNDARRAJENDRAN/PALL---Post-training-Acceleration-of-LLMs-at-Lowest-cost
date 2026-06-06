#!/usr/bin/env bash
# DocSmile VLM — environment bootstrap for AWS g6e.2xlarge (1x L40S 48GB).
# Recommended AMI: "Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)" (CUDA 12.x + drivers preinstalled).
# Run:  bash vlm/setup.sh
set -euo pipefail

echo "[0/6] GPU check"
nvidia-smi || { echo "No GPU / driver. Use a Deep Learning AMI."; exit 1; }

echo "[1/6] Python venv"
python3 -m venv ~/vlmenv
source ~/vlmenv/bin/activate
python -m pip install -U pip wheel setuptools

echo "[2/6] PyTorch (CUDA 12.1)"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "[3/6] Core deps"
pip install -r "$(dirname "$0")/requirements.txt"

echo "[4/6] FlashAttention-2 (prebuilt wheel; ~2 min)"
pip install flash-attn==2.7.0.post2 --no-build-isolation || \
  echo "  WARN: flash-attn install failed -> training will fall back to sdpa attention."

echo "[5/6] HF auth + fast downloads"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc
# put your token in ~/.hf_token (chmod 600). Used by training + to pull the backbone.
if [ -f ~/.hf_token ]; then huggingface-cli login --token "$(cat ~/.hf_token)"; fi

echo "[6/6] Verify"
python - <<'PY'
import torch, transformers, trl, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__)
try:
    import flash_attn; print("flash-attn", flash_attn.__version__)
except Exception: print("flash-attn: NOT installed (sdpa fallback)")
PY
echo "DONE. Activate with: source ~/vlmenv/bin/activate"
