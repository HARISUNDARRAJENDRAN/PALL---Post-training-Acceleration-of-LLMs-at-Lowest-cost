# Stage 2 — Supervised Fine-Tuning (SFT)

Teaches dental instruction-following on top of the CPT-adapted base, training a fresh
QLoRA adapter with loss masked to assistant-response tokens only.

## Files
- `sft_train.py` — trainer (TRL `SFTTrainer`, response-token masking)
- `sft_v2_a100_config.yaml` — **canonical** A100-40GB config (used by the launcher)
- `sft_config.yaml` — baseline config
- `run_sft_v2_a100.sh` — launcher (env + tmux + logging)

## Key config (`sft_v2_a100_config.yaml`)
| | |
|---|---|
| Base | CPT-adapted NF4 Llama-3.1-8B + fresh LoRA r=64/α=128 |
| Data | `Harisundar/pall` subset `sft` (391,693 train / 20,615 val) |
| Schedule | 2 epochs, eff. batch 48 (24 × grad-accum 2), lr 1e-4 cosine |
| Memory | grad-checkpointing OFF (enables batch 24), `group_by_length` ON |
| Optimizer | paged_adamw_8bit, bf16, seq 2048 |
| Output | adapter → `Harisundar/pall-llama3.1-8b-sft-v2` |

## Run
```bash
# canonical A100 run (via launcher)
bash training/SFT/run_sft_v2_a100.sh

# or directly
python training/SFT/sft_train.py --config training/SFT/sft_v2_a100_config.yaml

# smoke test
python training/SFT/sft_train.py --config training/SFT/sft_config.yaml --max-steps 100 --smoke-limit 5000
```

Prev: [`../CPT`](../CPT) · Next: [`../DPO`](../DPO).
