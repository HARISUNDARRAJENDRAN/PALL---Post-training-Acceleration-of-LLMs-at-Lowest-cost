#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DocSmile
source /workspace/venv/bin/activate

export HF_HOME=/workspace/.hf_cache
export TRANSFORMERS_CACHE=/workspace/.hf_cache
export HF_DATASETS_CACHE=/workspace/.hf_cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

date -Is > /workspace/logs/sft_v2_a100_start.txt
python training/SFT/sft_train.py --config training/SFT/sft_v2_a100_config.yaml 2>&1 | tee /workspace/logs/sft_v2_a100_train.log
rc=${PIPESTATUS[0]}
date -Is > /workspace/logs/sft_v2_a100_end.txt
tmux kill-session -t gpu_sampler 2>/dev/null || true
exit "$rc"
