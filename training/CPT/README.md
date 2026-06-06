# Stage 1 — Continued Pre-Training (CPT)

Injects dental-domain knowledge into the NF4-quantized Llama-3.1-8B base via causal
language modeling on a ~175M-token dental corpus.

## Files
- `cpt_train.py` — trainer (Unsloth + TRL `SFTTrainer` with packing, paged AdamW 8-bit)
- `cpt_config.yaml` — all hyperparameters

## Key config
| | |
|---|---|
| Base | `unsloth/Meta-Llama-3.1-8B` (4-bit NF4) |
| LoRA | r=64, α=128, dropout=0.05, all 7 projections |
| Data | `Harisundar/pall` subset `cpt` (train + 1% validation) |
| Schedule | 1 epoch, eff. batch 32 (4 × grad-accum 8), lr 2e-4 cosine |
| Optimizer | paged_adamw_8bit, bf16, packing ON, grad-checkpointing (unsloth) |
| Output | adapter → `Harisundar/pall-llama3.1-8b-cpt` |

## Run
```bash
python training/CPT/cpt_train.py --config training/CPT/cpt_config.yaml
# resume from last checkpoint:
python training/CPT/cpt_train.py --config training/CPT/cpt_config.yaml --resume
```

Shared helpers (`callbacks.py`, `prepare_data.py`) live one level up in `training/`.
Next stage: [`../SFT`](../SFT).
