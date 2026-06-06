# Stage 3 — Direct Preference Optimization (DPO)

Aligns the merged CPT+SFT model toward clinically safe, well-hedged responses. The CPT
and SFT adapters are merged into a bf16 base, re-quantized to NF4, and a fresh QLoRA
adapter is trained with the DPO objective against a frozen reference policy.

## Files
- `dpo_train.py` — trainer (TRL `DPOTrainer`, sigmoid loss)
- `dpo_v3_config.yaml` — **canonical** config
- `dpo_config.yaml` — baseline config
- `dpo_pipeline.sh` — full pipeline (merge → re-quantize → smoke → full run in tmux)
- `vllm_generate_rejected.py` — generate rejected responses via vLLM for pair construction

## Key config (`dpo_v3_config.yaml`)
| | |
|---|---|
| Base | merged CPT+SFT (bf16 → NF4) + fresh LoRA r=64/α=128 |
| Data | `Harisundar/pall` subset `dpo` (10,185 train / 536 val) — prompt/chosen/rejected |
| Schedule | 1 epoch, eff. batch 16 (4 × grad-accum 4), lr 2e-6 cosine |
| DPO | β=0.1, sigmoid loss, max_length 1024, max_prompt_length 512 |
| Optimizer | paged_adamw_8bit, bf16, grad-checkpointing ON |
| Output | adapter → `Harisundar/pall-llama3.1-8b-dpo-v3` |

> Note: lr is two orders of magnitude below SFT (2e-6 vs 1e-4) to prevent drift from the
> SFT anchor — DPO sharpens behavior, it should not re-learn the task.

## Run
```bash
# full pipeline (merge + smoke + train)
bash training/DPO/dpo_pipeline.sh

# or directly
python training/DPO/dpo_train.py --config training/DPO/dpo_v3_config.yaml

# smoke test
python training/DPO/dpo_train.py --config training/DPO/dpo_config.yaml --max-steps 5 --smoke-limit 200
```

Prev: [`../SFT`](../SFT). After DPO, merge all three adapters into the final model
(see the root `README.md`).
