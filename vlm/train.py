#!/usr/bin/env python
"""DocSmile VLM training — stage 1 (projector align) and stage 2 (LoRA SFT).

stage1: freeze vision + LLM + lm_head; train multi_modal_projector only. Saves projector.pt.
stage2: freeze vision; LoRA on the language_model + full-train projector (init from stage1).

Run:
  python train.py --stage 1 --config configs/training.yaml
  python train.py --stage 2 --config configs/training.yaml
"""
import argparse, os, json, yaml, torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments
from data import VLMDataset, Collator

def load_cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--base", default="/home/ec2-user/vlm/base_vlm")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, train on only N samples (debug)")
    ap.add_argument("--data_root", default=None, help="override cfg data.root")
    ap.add_argument("--train_file", default=None, help="override cfg data.train")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    sc = cfg["stage1_align"] if a.stage == 1 else cfg["stage2_sft"]
    droot = a.data_root or cfg["data"]["root"]
    train_file = a.train_file or cfg["data"]["train"]
    img_tokens = 729

    print(f"[load] processor + model from {a.base}")
    processor = AutoProcessor.from_pretrained(a.base)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = LlavaForConditionalGeneration.from_pretrained(
        a.base, dtype=torch.bfloat16,
        attn_implementation="sdpa")          # flash-attn unavailable on cu130/py313
    model.config.pad_token_id = tok.pad_token_id

    # ---------------- freezing / PEFT ----------------
    if a.stage == 1:
        model.model.vision_tower.requires_grad_(False)
        model.model.language_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
        model.model.multi_modal_projector.requires_grad_(True)
    else:
        from peft import LoraConfig, get_peft_model
        # init projector from stage 1
        proj_pt = os.path.join(cfg["stage1_align"]["output_dir"], "projector.pt")
        if os.path.exists(proj_pt):
            model.model.multi_modal_projector.load_state_dict(torch.load(proj_pt, map_location="cpu"))
            print("[stage2] loaded stage-1 projector")
        else:
            print("[stage2] WARNING: stage-1 projector not found, using random projector")
        model.model.vision_tower.requires_grad_(False)
        lc = LoraConfig(
            r=sc["lora"]["r"], lora_alpha=sc["lora"]["alpha"], lora_dropout=sc["lora"]["dropout"],
            bias="none", task_type="CAUSAL_LM",
            # regex: only language_model projections (NOT the vision tower's q/k/v)
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            modules_to_save=["multi_modal_projector"],   # train + save projector too
        )
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    model.config.use_cache = False
    if sc.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ---------------- data ----------------
    single = (sc.get("data_filter") == "single_image")
    train_ds = VLMDataset(os.path.join(droot, train_file), droot, processor,
                          sc["max_seq_len"], single_image_only=single, img_tokens_per_image=img_tokens)
    print(f"[data] train: {len(train_ds)} (dropped over-length: {train_ds._dropped})")
    if a.smoke:
        train_ds.rows = train_ds.rows[:a.smoke]
        print(f"[smoke] using {len(train_ds)} samples")
    collator = Collator(tok.pad_token_id)

    args = TrainingArguments(
        output_dir=sc["output_dir"],
        per_device_train_batch_size=sc["per_device_train_batch_size"],
        gradient_accumulation_steps=sc["gradient_accumulation_steps"],
        learning_rate=float(sc["learning_rate"]),
        lr_scheduler_type=sc["lr_scheduler_type"],
        warmup_ratio=sc["warmup_ratio"],
        num_train_epochs=sc["num_train_epochs"],
        bf16=True,
        gradient_checkpointing=bool(sc.get("gradient_checkpointing")),
        logging_steps=sc["logging_steps"],
        # stage1: no Trainer checkpoints (16GB full-model dumps) -> we save projector.pt manually.
        # stage2: PEFT -> checkpoints are small adapters, keep them for resumability.
        save_strategy=("no" if a.stage == 1 else "steps"),
        save_steps=sc["save_steps"],
        save_total_limit=sc.get("save_total_limit", 3),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    # ---------------- save ----------------
    os.makedirs(sc["output_dir"], exist_ok=True)
    if a.stage == 1:
        torch.save(model.model.multi_modal_projector.state_dict(),
                   os.path.join(sc["output_dir"], "projector.pt"))
        print("[stage1] saved projector.pt")
    else:
        trainer.save_model(sc["output_dir"])   # LoRA adapter + projector (modules_to_save)
        print("[stage2] saved LoRA adapter + projector")
    processor.save_pretrained(sc["output_dir"])
    print("DONE.")

if __name__ == "__main__":
    main()
