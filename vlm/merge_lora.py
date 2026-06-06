#!/usr/bin/env python
"""Merge the trained LoRA + projector into the base VLM, save a single 16 GB model,
verify it loads and runs a forward pass, and emit a model card.

Inputs:
  BASE  = /home/ec2-user/vlm/base_vlm                      (frozen, 16 GB)
  S1    = /home/ec2-user/vlm/ckpt/stage1_align/projector.pt (aligned projector)
  S2    = /home/ec2-user/vlm/ckpt/stage2_sft               (PEFT LoRA + projector)
  OUT   = /home/ec2-user/vlm/docsmile_vlm_merged           (final 16 GB model)
"""
import os, sys, time, json, shutil, traceback
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from PIL import Image
import numpy as np

BASE = "/home/ec2-user/vlm/base_vlm"
S1_PROJ = "/home/ec2-user/vlm/ckpt/stage1_align/projector.pt"
S2_DIR = "/home/ec2-user/vlm/ckpt/stage2_sft"
OUT = "/home/ec2-user/vlm/docsmile_vlm_merged"

LOG = "/home/ec2-user/merge.log"
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

try:
    log(f"=== merge start ===")
    log(f"BASE  = {BASE}")
    log(f"S1    = {S1_PROJ}  (exists={os.path.exists(S1_PROJ)})")
    log(f"S2    = {S2_DIR}  (exists={os.path.exists(S2_DIR)})")
    log(f"OUT   = {OUT}")

    # Load base model (16 GB)
    log("[1/5] loading base model from disk...")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(BASE)
    log(f"  loaded in {time.time()-t0:.1f}s")

    # Apply stage-1 aligned projector (43 MB)
    if os.path.exists(S1_PROJ):
        sd = torch.load(S1_PROJ, map_location="cpu")
        model.model.multi_modal_projector.load_state_dict(sd)
        log("[2/5] applied stage-1 projector to multi_modal_projector")
    else:
        log(f"[2/5] WARNING: stage-1 projector not found at {S1_PROJ}; using random init")

    # Apply stage-2 LoRA + projector (PEFT)
    log(f"[3/5] applying stage-2 PEFT adapter from {S2_DIR}...")
    t0 = time.time()
    peft_model = PeftModel.from_pretrained(model, S2_DIR)
    log(f"  PEFT loaded in {time.time()-t0:.1f}s")
    log("  merging LoRA into base weights...")
    t0 = time.time()
    model = peft_model.merge_and_unload()
    log(f"  merged in {time.time()-t0:.1f}s")

    # Save as a single consolidated model
    os.makedirs(OUT, exist_ok=True)
    log(f"[4/5] saving merged model to {OUT}...")
    t0 = time.time()
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(OUT)
    # chat_template lives next to the model (not in processor bundle)
    src_chat = os.path.join(BASE, "chat_template.jinja")
    if os.path.exists(src_chat):
        shutil.copy(src_chat, os.path.join(OUT, "chat_template.jinja"))
    log(f"  saved in {time.time()-t0:.1f}s")

    # --- verify ---
    log(f"[5/5] verifying merged model...")
    sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)
             if os.path.isfile(os.path.join(OUT, f)))
    log(f"  output dir size: {sz/1e9:.2f} GB")

    # quick reload + forward smoke-test
    del model
    torch.cuda.empty_cache()
    log("  reloading from disk + running forward smoke-test...")
    t0 = time.time()
    m2 = LlavaForConditionalGeneration.from_pretrained(OUT, dtype=torch.bfloat16,
                                                       attn_implementation="sdpa").cuda()
    p2 = AutoProcessor.from_pretrained(OUT)
    img = Image.fromarray(np.uint8(np.random.rand(384, 384, 3) * 255))
    text = p2.tokenizer.apply_chat_template(
        [{"role": "user", "content": "<image>\nBriefly describe this dental image."}],
        tokenize=False, add_generation_prompt=True,
    )
    batch = p2(images=[img], text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m2(**batch)
    n_img_tok = (batch["input_ids"] == 128256).sum().item()
    log(f"  reload+forward ok in {time.time()-t0:.1f}s")
    log(f"  input_ids: {tuple(batch['input_ids'].shape)}, <image> tokens: {n_img_tok}, "
        f"logits: {tuple(out.logits.shape)}")

    # write a model card
    model_card = f"""---
language: en
license: apache-2.0
tags:
  - multimodal
  - vision-language
  - dental
  - llama
  - siglip
  - llava
pipeline_tag: image-text-to-text
---

# DocSmile VLM (LLaVA-style)

A 8.5 B parameter multimodal vision-language model for dental clinical use, built by grafting a
SigLIP-so400m-384 vision tower onto a DPO-tuned Llama-3.1-8B dental LLM, then running a 2-stage
LLaVA training recipe on 32,884 dental VQA / classification records.

## Architecture
- **Vision tower**: `google/siglip-so400m-patch14-384` (frozen)
- **Projector**: 2-layer MLP with GELU (trained)
- **Language model**: Llama-3.1-8B with CPT+SFT+DPO on dental text (LoRA r=16 fine-tuned)
- **lm_head**: tied
- **Total params**: 8.48 B (vision ~0.4 B, projector ~10 M, LLM 8 B + LoRA 63 M)

## Training data
32,884 records (~25.3k single-image + ~7.5k multi-image) covering:
- Oral cancer clinical photos and histopathology
- Dental radiographs (periapical, panoramic)
- ICDAS caries scoring
- Tufts dental radiology database
- DENTEX panoramic lesions
- Dental textbook figures

## Training recipe
- **Stage 1 (alignment, ~2 h on 1x L40S 48 GB)**: train MLP projector only, LLM + vision frozen.
  23,150 single-image records, batch 4, grad-accum 8 (eff 32), lr 1e-3 cosine, 1 epoch.
  Final projector loss: 3.82 -> 0.72.
- **Stage 2 (instruction tuning, ~10 h on 1x L40S 48 GB)**: LoRA r=16 on
  `q/k/v/o/gate/up/down_proj` of all 32 Llama layers + continue training projector.
  29,667 records (incl. 6-image code-oral-cls), batch 2, grad-accum 8 (eff 16), lr 2e-5
  cosine, 1 epoch, gradient checkpointing. Final loss: 0.88 -> ~0.70.

## Usage
```python
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained(
    "Harisundar/PALL-VLM", dtype=torch.bfloat16, device_map="cuda"
)
processor = AutoProcessor.from_pretrained("Harisundar/PALL-VLM")

image = Image.open("dental_image.jpg").convert("RGB")
text = processor.tokenizer.apply_chat_template(
    [{{"role": "user", "content": "<image>\\nWhat is shown? Give ICDAS score if applicable."}}],
    tokenize=False, add_generation_prompt=True,
)
batch = processor(images=[image], text=text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**batch, max_new_tokens=200, do_sample=False)
print(processor.tokenizer.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True))
```

## License
Apache 2.0 (consistent with the underlying Llama 3.1 and SigLIP licenses).

## Limitations
- Trained on a curated dental dataset; performance on out-of-distribution clinical images is
  unverified.
- For research and clinical-decision-support only; not for autonomous diagnosis.
"""
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(model_card)
    log(f"  wrote README.md model card")

    log("=== MERGE OK ===")
except Exception as e:
    log(f"!!! MERGE FAILED: {e}")
    log(traceback.format_exc())
    sys.exit(1)
