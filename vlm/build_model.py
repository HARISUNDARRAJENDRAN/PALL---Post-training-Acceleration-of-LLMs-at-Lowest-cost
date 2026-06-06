#!/usr/bin/env python
"""Assemble a LLaVA-style VLM = SigLIP vision tower + MLP projector + DPO'd Llama-3.1-8B.
Loads real weights for vision + language; projector is randomly initialised (trained in
stage 1). Adds an <image> token, builds a LlavaProcessor, runs a forward smoke-test, saves.

transformers 5.9 layout: LlavaForConditionalGeneration.model.{vision_tower,
multi_modal_projector, language_model} + .lm_head .

Usage: python build_model.py --backbone <hf> --vision <hf> --out <dir>
"""
import argparse, os, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor,
                          SiglipVisionModel, LlavaConfig, LlavaForConditionalGeneration,
                          LlavaProcessor)

# canonical Llama-3.1 chat template (backbone tokenizer ships without one).
# uses bos_token var -> tokenize rendered text with add_special_tokens=False.
LLAMA3_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Harisundar/PALL-Text")
    ap.add_argument("--vision", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--out", default="/home/ec2-user/vlm/base_vlm")
    ap.add_argument("--image_token", default="<image>")
    a = ap.parse_args()
    dtype = torch.bfloat16

    print("[1/6] tokenizer + <image> token")
    tok = AutoTokenizer.from_pretrained(a.backbone)
    if tok.chat_template is None:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
        print("    set Llama-3.1 chat template (backbone had none)")
    if a.image_token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [a.image_token]})
    image_token_id = tok.convert_tokens_to_ids(a.image_token)
    print("    image_token_id =", image_token_id, "| vocab =", len(tok))

    print("[2/6] vision tower:", a.vision)
    vision = SiglipVisionModel.from_pretrained(a.vision, dtype=dtype)
    img_proc = AutoImageProcessor.from_pretrained(a.vision)
    vc = vision.config
    num_img_tokens = (vc.image_size // vc.patch_size) ** 2
    print("    image_size", vc.image_size, "patch", vc.patch_size, "-> img_tokens", num_img_tokens)

    print("[3/6] backbone LLM:", a.backbone)
    llm = AutoModelForCausalLM.from_pretrained(a.backbone, dtype=dtype)

    print("[4/6] assemble LlavaForConditionalGeneration")
    cfg = LlavaConfig(
        vision_config=vc.to_dict(),
        text_config=llm.config.to_dict(),
        image_token_index=image_token_id,
        image_seq_length=num_img_tokens,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",   # SigLIP has no CLS token -> keep all patches
        vision_feature_layer=-1,
    )
    model = LlavaForConditionalGeneration(cfg)
    # graft real weights (random language_model from init is replaced -> GC'd)
    model.model.vision_tower = vision
    model.model.language_model = llm.model
    model.lm_head = llm.lm_head
    model.resize_token_embeddings(len(tok))
    model.config.image_token_index = image_token_id
    model.config.text_config.vocab_size = model.get_input_embeddings().weight.shape[0]
    model = model.to(dtype)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"    total params: {nparam/1e9:.2f}B")

    print("[5/6] processor + forward smoke-test")
    processor = LlavaProcessor(
        image_processor=img_proc, tokenizer=tok,
        patch_size=vc.patch_size, vision_feature_select_strategy="full",
        image_token=a.image_token, num_additional_image_tokens=0,
    )
    processor.chat_template = tok.chat_template   # reuse Llama-3.1 chat template
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.uint8(np.random.rand(384, 384, 3) * 255))
    # raw prompt (decoupled from template) — our dataset puts <image> inline in the text
    text = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + a.image_token + "\nWhat is shown?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")
    batch = processor(images=[img], text=text, return_tensors="pt", add_special_tokens=False)
    n_img_tok = (batch["input_ids"] == image_token_id).sum().item()
    print("    input_ids len:", batch["input_ids"].shape, "| <image> tokens:", n_img_tok,
          "(expected", num_img_tokens, ")")
    assert n_img_tok == num_img_tokens, "image-token expansion mismatch!"
    model_dev = model.to("cuda")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    print("    logits:", tuple(out.logits.shape), "-> forward OK")
    del out; torch.cuda.empty_cache()
    model = model.to("cpu")

    print("[6/6] save ->", a.out)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    processor.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("DONE.")

if __name__ == "__main__":
    main()
