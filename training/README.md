# DocSmile CPT training pipeline

Continued Pretraining for **Llama 3.1 8B** on the dental corpus, using
**Unsloth + QLoRA (rank 64) + FlashAttention 2 + sequence packing** on a
single **A100 40GB** rented on Vast.ai.

This directory is self-contained: clone the repo, run two scripts, watch
TensorBoard. ~5-7 minute setup, ~6-8 hour training run.

## File layout

```
training/
├── .env.example         # template; copy to .env and fill in HF_TOKEN
├── requirements.txt     # pinned versions, fast install on Vast.ai
├── setup_vast.sh        # one-shot bootstrap on a Vast.ai instance
├── prepare_data.py      # ONE-TIME from laptop: upload FINAL/* to HF Hub
├── cpt_config.yaml      # all hyperparameters
├── cpt_train.py         # main training script
├── callbacks.py         # GPU mem + throughput + MCQ eval callbacks
├── resume.sh            # one-liner to resume after spot interruption
└── README.md            # this file
```

## One-time setup on your laptop

The FINAL/ JSONL files (CPT 406K + SFT 195K + DPO 10K) are pushed to a private
HuggingFace dataset repo so every future Vast.ai instance can download them
in seconds.

```powershell
# from project root
cd training
cp .env.example .env
# Edit .env and paste your real HF_TOKEN  (https://huggingface.co/settings/tokens)
#   ⚠️  rotate the token if it has ever been pasted in chat or committed
# Then:
pip install huggingface-hub
python prepare_data.py --dry-run     # sanity-check the plan
python prepare_data.py               # uploads to https://huggingface.co/datasets/Harisundar/docsmile-dental
```

This is a one-time step. After it's done, Vast.ai instances pull the data
directly from the Hub.

## Renting a Vast.ai instance

Pick an instance with:

- **GPU**: 1 × A100 80GB *or* A100 40GB (40GB is fine for this config)
- **Disk**: ≥ 50 GB (model weights + caches + checkpoints)
- **Image**: must be **torch 2.4–2.7 + CUDA 12.x**. Tested combinations:
  - `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` (recommended)
  - `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
  - **Do NOT pick** torch 2.8+/2.10 or CUDA 13.x images — Unsloth and
    flash-attn don't have matching wheels and `pip install` will fail.
- **Region**: cheapest available; we don't depend on a specific one

Bid on a spot instance only if the persistent volume is enabled — otherwise
spot interruptions wipe `/workspace` and you re-download everything.

## On the Vast.ai instance

After SSH'ing in:

```bash
# 1. Clone repo
git clone https://github.com/<your-user>/DocSmile.git
cd DocSmile/training

# 2. Bootstrap (~5-7 min: pip deps + flash-attn prebuilt wheel + verify)
bash setup_vast.sh

# 3. Configure secrets
cp .env.example .env
nano .env                       # paste your real HF_TOKEN
huggingface-cli login --token "$(grep ^HF_TOKEN .env | cut -d= -f2)"

# 4. (Optional) Tmux so the session survives if your laptop disconnects
tmux new -s docsmile

# 5. Start training
python cpt_train.py --config cpt_config.yaml

# 6. In a second tmux pane, run TensorBoard
tensorboard --logdir /workspace/runs --port 6006 --bind_all
```

To view TensorBoard from your laptop:

```bash
ssh -p <vast_ssh_port> -L 6006:localhost:6006 root@<vast_host>
# Then open http://localhost:6006 in your browser.
```

## What's being optimized

| Lever | Value | Why |
|---|---|---|
| Model | Llama 3.1 8B (NF4) | best 8B base, broad medical pretraining |
| LoRA rank | 64 | strong domain shift (dental medicine) |
| LoRA alpha | 128 | standard 2× rank |
| Target modules | all 7 linear | full attention + MLP coverage |
| Context | 2048 | matches chunk size distribution; faster than 4096 |
| Effective batch | 32 (4 × 8 accum) | ~65K tokens / step |
| Optimizer | paged AdamW 8-bit | memory-efficient |
| LR | 2e-4 | Unsloth QLoRA default |
| Schedule | cosine, 3% warmup | standard |
| Precision | bf16 | A100 native |
| Packing | on | eliminates pad waste, ~1.5-3× throughput |
| Grad checkpointing | unsloth | selective recompute |
| FlashAttention | 2 (A100) | memory-efficient attention kernel |

## Monitoring (TensorBoard tags)

| Tag | What |
|---|---|
| `train/loss` | per-logging-step training loss |
| `train/grad_norm` | gradient norm (clipping threshold = 1.0) |
| `train/learning_rate` | current LR (cosine decay) |
| `eval/loss` | validation loss every 500 steps |
| `eval/perplexity` | exp(eval_loss) at end of training |
| `gpu/mem_used_gb` | GPU memory currently allocated |
| `gpu/mem_reserved_gb` | reserved memory (allocator pool) |
| `gpu/mem_peak_gb` | peak memory since training start |
| `throughput/tokens_per_sec` | sustained training throughput |
| `throughput/samples_per_sec` | samples per second |
| `throughput/eta_min` | running ETA in minutes |
| `eval_mcq/accuracy` | (optional) dental MCQ accuracy every 1000 steps |

## Resuming after a spot interruption

The training script pushes the adapter to your HF model repo every
`save_steps` (1000). If the Vast.ai instance dies:

1. Rent a fresh instance with the same image
2. Clone the repo and `bash setup_vast.sh` again (~5 min)
3. `cp .env.example .env` and refill `HF_TOKEN`
4. Run `bash resume.sh` — pulls latest checkpoint from HF Hub, resumes training

Worst case you lose ≤ 1 hour of progress (steps since last save).

## Cost estimate

| GPU | $/hr (Vast.ai) | Wall time | Cost (single full run) |
|---|---|---|---|
| 1× A100 40GB | ~$0.70 | ~7-9 hr | **~$5-7 (≈ ₹400-580)** |
| 1× A100 80GB | ~$1.00 | ~6-8 hr | **~$6-8 (≈ ₹500-660)** |

A second run for ablation (e.g. rank 32 vs 64, or with/without packing) fits
inside your ₹4,000 budget with room to spare.

## After CPT

Once CPT is done and the adapter is on the Hub, the same pattern repeats for
SFT (then DPO):

- Reuse this script with a new config (`sft_config.yaml`) pointing at
  `data.config: "sft"` and `data.text_field: "question"` (plus a
  formatting function — to be written in `sft_train.py`).
- Initialize from the CPT adapter via `model_name = HF_MODEL_REPO`.

That's planned, not built yet — focus is CPT first.
