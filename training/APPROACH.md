# DocSmile training approach

This document explains the technology choices for DocSmile's three-stage training
pipeline (CPT → SFT → DPO) and why each was picked over the obvious alternatives.
For *how* to run the pipeline, see `README.md`.

## Pipeline overview

```
Llama 3.1 8B (base)
    │
    ▼  Stage 1: Continued Pretraining (CPT)
       ~85k dental text chunks (~175M tokens), 1 epoch, ~14 hr on A100-40GB
    │
    ▼  Stage 2: Supervised Fine-Tuning (SFT)
       Dental {question, answer, source, topic} pairs
    │
    ▼  Stage 3: Direct Preference Optimization (DPO)
       Dental {prompt, chosen, rejected} preferences
    │
    ▼
DocSmile-8B (final adapter)
```

CPT teaches terminology and stylistic priors of the domain; SFT teaches the
instruction-following format we want; DPO sharpens preferences (correctness,
clinical caution, citation style).

---

## Tech choices

### Base model: Llama 3.1 8B (via `unsloth/Meta-Llama-3.1-8B`)

| Considered | Verdict |
|---|---|
| **Llama 3.1 8B** ✅ | Strong base, open weights, 128k context, well-supported by Unsloth |
| Llama 2 7B | Older, weaker on reasoning benchmarks, no GQA |
| Mistral 7B | Comparable, but weaker on medical eval (MedQA, USMLE) |
| Phi-3-mini (3.8B) | Smaller, fits cheaper GPUs, but loses reasoning headroom |
| Qwen 2.5 7B | Strong alternative; deferred — Llama ecosystem (Unsloth, FA2, eval tools) is more mature for medical fine-tuning |

**Why Unsloth's prequantized variant**: their `unsloth/Meta-Llama-3.1-8B` ships
NF4-quantized weights, so the download is ~5 GB instead of 16 GB, and the loader
skips the on-device quantization step.

### Parameter-efficient training: QLoRA (NF4 + LoRA r=64)

| Considered | Verdict |
|---|---|
| **QLoRA (4-bit NF4 + LoRA r=64, all linear)** ✅ | Trains 8B in ~15 GB VRAM. NF4 is information-theoretically optimal for normal-distributed weights. r=64 is the standard "expressive adapter" size for 8B. |
| Full fine-tuning | Needs ~200 GB VRAM for 8B (Adam states + grads + activations). Not affordable. |
| LoRA without quantization (fp16) | Needs ~30 GB VRAM — would force A100-80GB. |
| LoRA r=8/16 | Lower-rank adapters underfit domain shifts of this size (175M CPT tokens). |
| DoRA (decomposed LoRA) | ~5% better at +30% training time. Not worth it at our scale. |
| Prefix tuning / P-tuning | Far less expressive than LoRA for CPT-scale shifts. |

LoRA is attached to **all** linear projections (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj`) — restricting to just attention
hurts CPT quality measurably.

### Training stack: Unsloth + TRL `SFTTrainer`

| Considered | Verdict |
|---|---|
| **Unsloth + TRL 0.13 `SFTTrainer`** ✅ | Unsloth provides ~2x faster CUDA kernels for attention/MLP, baked-in FA2, gradient checkpointing variant. TRL wraps HF Trainer with `dataset_text_field`, packing, chat templates. |
| Vanilla HuggingFace Trainer | Works but ~2x slower; no quantization-aware kernels. |
| Axolotl | Excellent config-driven framework, but heavier; we want Python-level control for callbacks (MCQ eval, throughput logging). |
| torchtune | Newer (Meta-official), less mature ecosystem for QLoRA + custom callbacks. |
| Raw PyTorch training loop | Too much boilerplate to recreate gradient checkpointing, scheduler, hub push, resume. |

### Attention: Flash Attention 2

| Considered | Verdict |
|---|---|
| **FlashAttention-2** ✅ | 2–3x faster than baseline attention, ~10x lower memory in attention, exact (not approximate). |
| PyTorch SDPA | Decent fallback, but ~30-40% slower at 2048 seq length. |
| xformers `memory_efficient_attention` | Older, slower than FA2 on Ampere/Hopper. |
| Vanilla attention | OOMs at seq=2048 with our config; ~5x slower. |

### Sequence packing

| Considered | Verdict |
|---|---|
| **Packing (`packing=True`)** ✅ | Stuffs multiple short chunks into a single 2048-token sequence — near-zero PAD-token waste. Improves effective tokens-per-step by ~2-3x for our corpus (CPT chunks average ~800 tokens). |
| Padding to max_seq_length | Wastes ~50% of compute on PAD tokens. |
| Bucketing by length | Better than padding, but packing dominates it for variable-length corpora. |

### Optimizer: `paged_adamw_8bit` (bitsandbytes)

| Considered | Verdict |
|---|---|
| **paged_adamw_8bit** ✅ | 8-bit optimizer states halve memory vs fp32 Adam. "Paged" version offloads optimizer states to CPU when GPU memory spikes, preventing OOM on training restarts. |
| `adamw_torch` (fp32 states) | 4x more memory; would force A100-80GB. |
| Adafactor | Less expressive on LoRA fine-tuning; convergence is worse. |
| Lion | Faster per-step but slightly worse final loss; trade-off not worth it for CPT. |

### Precision: bf16

| Considered | Verdict |
|---|---|
| **bf16** ✅ | A100/H100 native, full fp32 dynamic range, no loss scaling needed. |
| fp16 | Needs loss scaling, can underflow on small grads, deprecated for new LLM training on Ampere+. |
| fp32 | 2x memory, no real benefit on bf16-capable hardware. |
| fp8 (TransformerEngine) | Promising but immature for QLoRA; reserved for future ablations. |

### LR schedule: cosine + 3% warmup

| Considered | Verdict |
|---|---|
| **Cosine + 3% warmup** ✅ | De-facto standard for LLM fine-tuning. Warmup prevents early instability; cosine decay gives smooth approach to minimum. |
| Constant LR | Worse final loss; no exploration vs exploitation tradeoff. |
| Linear decay | Sharper end-of-training, less stable for short runs. |
| One-cycle / cyclic | Overkill for 1-epoch CPT; better suited to long pre-training. |

Peak LR = 2e-4 — the empirical sweet spot for LoRA r=64 on Llama-class models
(Unsloth and the QLoRA paper both converge on this range).

### Batch geometry: micro-batch 4 × grad-accum 8 → effective 32, seq 2048

| Choice | Why |
|---|---|
| Effective batch 32 | Standard for 8B SFT/CPT — large enough for stable gradients, small enough to fit on a single A100-40GB. |
| Micro-batch 4 | Caps activation memory. Could go to 8 if `use_gradient_checkpointing=false`, ~2x throughput (deferred for stability on first full run). |
| Seq length 2048 | Long enough to capture textbook paragraphs (avg dental chunk ~800 tokens after our cleaning pipeline). 4096 doubles memory for marginal corpus-side benefit. |

### Compute platform: Modal

| Considered | Verdict |
|---|---|
| **Modal** ✅ | Per-second billing, prebuilt CUDA images, persistent volumes, secrets, detached runs (laptop can disconnect), automatic spot recovery. Trivial Python wrapping. |
| Vast.ai | 3-5x cheaper but spot-prone; needs manual SSH, env setup, watchdog. Acceptable fallback. |
| Lambda Labs | A100s at $1-1.50/hr but reserved-only — no per-second spot. |
| AWS SageMaker | Massive overhead, expensive, IAM headaches. |
| RunPod | Comparable to Modal, but Python ergonomics weaker. |
| Google Colab Pro+ | 12-hr cap, no checkpoint persistence, no detached runs — disqualified for 14-hr CPT. |

HuggingFace Hub is used as the durable checkpoint store: every `save_steps`
(=1000) the adapter is pushed to a private repo. If the Modal volume is lost
(spot eviction, accidental wipe), training resumes from the latest Hub revision.

### Data pipeline: HF Datasets + pre-built `Harisundar/pall`

CPT corpus (`Harisundar/pall::cpt`) is built by `prepare_data.py` from a
multi-source dental literature pool (PubMed, open-access dental journals,
WHO/ADA guidelines, dental textbooks where licensing permits). Cleaning includes
language ID, dental-relevance keyword filter, dedup (MinHash), and tokenization
to 2048-token chunks with overlap-respecting boundaries.

We deliberately pull the dataset *from the Hub at training time* (not local
files): it makes the training run reproducible from any machine, and the Hub
revision tracks data versioning.

---

## What this approach trades off

**What we give up**:

- **Raw capability ceiling** — an 8B model + 175M-token CPT will not outperform GPT-4 on open-ended dental conversation. It can outperform GPT-4 on narrow benchmarks (MCQ, terminology, coding) and serves at ~100x lower cost.
- **Multi-GPU training** — single A100 means we can't scale to 70B base models without major reconfiguration (FSDP/DeepSpeed).
- **Inference speed during training** — Unsloth optimizes training, not serving. For deployment, the adapter is merged back to fp16 weights for vLLM/TGI serving.

**What we gain**:

- **Reproducibility on a $20 budget** — entire CPT runs on a single A100 in <14 hr at ~$15-20.
- **Recoverability** — Hub push every 1k steps means a worst-case loss of ~30 min of training.
- **Defensible research story** — every choice has a "compared to X" justification rather than a "we used whatever the tutorial said."

---

## Observed metrics from the 2026-05-23 → 2026-05-24 full CPT run

- **Wall time**: ~14h 58m (single A100-40GB on Modal)
- **Step time**: ~19.9 s/step
- **Throughput**: ~3,290 tokens/sec (below Unsloth's theoretical ~6.5k — bottleneck is `use_gradient_checkpointing: "unsloth"` recomputing activations; trades compute for the 25 GB VRAM headroom)
- **Peak VRAM**: 15.25 GB (during eval) / 14.5 GB (training steady-state) — 36% of A100-40GB capacity
- **No NaN / no divergence / no grad spikes** — grad norm stayed in 0.15–0.21 throughout

### Train loss progression
| step | train_loss |
|---|---|
| 10 | 1.879 |
| 60 | 1.766 |
| 500 | ~1.70 |
| 1000 | ~1.70 |
| 1500 | ~1.68 |
| 2000 | ~1.67 |
| 2500 | 1.665 |
| 2669 (final) | 1.668 |

### Eval loss + perplexity progression
| step | eval_loss | perplexity |
|---|---|---|
| 500 | 1.7344 | 5.66 |
| 1000 | 1.7171 | 5.57 |
| 1500 | 1.6999 | 5.47 |
| 2000 | 1.6892 | 5.41 |
| 2500 | 1.6845 | 5.39 |
| **final (2669)** | **1.6844** | **5.389** |

Last 169 steps moved eval by 0.0001 — the cosine tail was effectively flat by design, confirming the schedule was correctly sized for 1 epoch. Train/eval gap stayed at ~0.02–0.04 throughout → **no overfitting**, consistent generalization.

---

## Next-stage optimization opportunities (for SFT/DPO)

1. Set `use_gradient_checkpointing: false` (or `true` instead of `"unsloth"`) — expected ~2x throughput, memory likely fits in 25-30 GB.
2. Bump `per_device_train_batch_size` to 8 — fills tensor cores better.
3. Consider DoRA for SFT if the training budget allows the +30% time cost — marginal quality gain on instruction following.
4. Add `eval_perplexity` to TensorBoard during training (currently only computed at final eval) — gives a live convergence signal.
