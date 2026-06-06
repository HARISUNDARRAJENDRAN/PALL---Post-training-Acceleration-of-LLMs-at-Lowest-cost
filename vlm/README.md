# DocSmile VLM — training runbook (AWS g6e.2xlarge / L40S 48 GB)

Turns the text-only `Harisundar/PALL-Text` into a multimodal dental VLM by
grafting a SigLIP vision tower + MLP projector (LLaVA-style), then 2-stage training on
`IMAGES/vlm_train/` (32,884 records, 52,461 images).

## Hardware
- **g6e.2xlarge** = 1× NVIDIA L40S (48 GB), 8 vCPU, 64 GB RAM, ~450 GB NVMe.
- 48 GB lets us run **bf16 LoRA (no quantization)**, bigger batches, and the 6-image
  samples — all of which were tight on the originally-planned L4 24 GB.

## What YOU set up (then send me SSH)
1. **Launch g6e.2xlarge** with AMI **"Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)"**.
   Root/EBS volume **≥ 200 GB gp3** (backbone 16 GB + dataset 11 GB + checkpoints).
2. **Open SSH** (port 22) to your IP; key-pair handy.
3. **Get the code + data onto the box:**
   - Clone the repo (or `scp` it), OR at minimum upload `vlm/` and `IMAGES/vlm_train/`.
   - Dataset is ~11 GB: `scp`/`rsync` `IMAGES/vlm_train/` or zip it first.
4. **HF token:** `echo hf_xxx > ~/.hf_token && chmod 600 ~/.hf_token`
   (needs read access to your backbone repo; + accept `google/medsiglip-448` terms IF you
   want MedSigLIP instead of the default open SigLIP).
5. **Bootstrap env:** `bash vlm/setup.sh`  (installs torch cu121, transformers, flash-attn, …)
6. **Send me the SSH string** (`ssh -i key.pem ubuntu@<public-dns>`). I take it from here.

## What I do once I have SSH (the "remaining training part")
- Write/finalize the thin training code against `vlm/configs/training.yaml`:
  - `build_model.py` — assemble `LlavaForConditionalGeneration` (vision tower + projector +
    your backbone), add `<image>` token, save base VLM.
  - `data.py` — JSONL → chat-templated samples + image loading/collation (single & multi-image).
  - `train.py` — stage-1 (projector) then stage-2 (LoRA+projector), checkpointing to EBS.
  - `eval.py` — metrics on `test.jsonl` (balanced-acc/F1, LLM-judge, image-shuffle control).
- Smoke-test on ~50 samples, confirm VRAM headroom, then launch full runs in tmux.
- Report metrics; merge adapters; (optional) push to HF; serve via vLLM.

## Plan (encoded in configs/training.yaml)
| Stage | Trainable | Data | ~Time (L40S) |
|------|-----------|------|------|
| 1 — VL alignment | projector only (vision+LLM frozen) | single-image subset (~25k) | 1–2 h |
| 2 — instruction tuning | LoRA(LLM r16) + projector | all 32,884 (incl multi-image) | 10–15 h |
| 3 — DPO (optional) | deferred — no multimodal pref data yet | — | — |

Default encoder = `google/siglip-so400m-patch14-384` (open, immediate). Swap to
`google/medsiglip-448` (better medical prior; license-gated) by editing `model.vision_encoder`
and `data.image_size: 448` in the config.

## Notes / risks
- **Modality collapse** is the top risk (templated/imbalanced classification). Stage-2 mixes
  tasks; we verify with an **image-shuffle control** (accuracy must drop when the image is random).
- Backbone is DPO-tuned → stage-1 keeps the LLM frozen and stage-2 uses modest LoRA to avoid
  erasing dental text skills (optionally mix a little text-only replay in stage 2).
- Wide panoramic radiographs are square-resized for v1 (AnyRes/tiling is a later upgrade).
