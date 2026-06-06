"""DocSmile DPO training script.

Continues from the SFT LoRA adapter and trains it further with DPO on
on-policy preference pairs (chosen = gold long answer, rejected = SFT model
generation on the same prompt).

Usage:
    python training/DPO/dpo_train.py --config training/DPO/dpo_config.yaml

Smoke:
    python training/DPO/dpo_train.py --config training/DPO/dpo_config.yaml --max-steps 5 --smoke-limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

import datasets.utils._dill as _dill_mod
if not hasattr(_dill_mod, "log"):
    _dill_mod.log = lambda *a, **kw: None

from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import Dataset
from transformers import set_seed
from trl import DPOConfig, DPOTrainer


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def load_env_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ[k] = v


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pairs(path: Path, smoke_limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompt = r.get("prompt")
            chosen = r.get("chosen")
            rejected = r.get("rejected")
            if not prompt or not chosen or not rejected:
                continue
            # Strings (legacy) or conversational (list-of-messages). Pass through.
            if isinstance(chosen, str) and isinstance(rejected, str):
                if chosen.strip() == rejected.strip():
                    continue
            rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
            if smoke_limit and len(rows) >= smoke_limit:
                break
    return rows


def split_train_val(rows: list[dict], val_fraction: float, seed: int):
    import random
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(rows) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [rows[i] for i in idx if i not in val_idx]
    val = [rows[i] for i in idx if i in val_idx]
    return train, val


def ensure_trainable_adapter(model) -> None:
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train > 0:
        return
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError("No trainable LoRA parameters found after loading SFT adapter.")
    print(f"  forced LoRA params trainable: {n_train / 1e6:.1f}M")


# Custom Llama-3.1 chat template that EXACTLY reproduces the SFT format_example
# (BOS + role headers + content + <|eot_id|>, with NO "Cutting Knowledge Date"
# preamble). Using this for conversational DPO rows guarantees the prompt /
# completion tokenization matches what the model saw during SFT.
SFT_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
)


def build_model_and_tokenizer(cfg: dict):
    m = cfg["model"]
    adapter = cfg.get("resume_adapter", {})
    merged_base = m.get("merged_base", False)

    if merged_base:
        # Approach B: load the full CPT+SFT merged model as a FROZEN base and
        # attach a FRESH LoRA. The DPO reference (adapter disabled) is then the
        # merged CPT+SFT model itself -> correct KL anchor.
        banner(f"Loading MERGED CPT+SFT model as base: {m['name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=m["name"],
            max_seq_length=m["max_seq_length"],
            dtype=m["dtype"],
            load_in_4bit=m["load_in_4bit"],
            token=os.environ.get("HF_TOKEN"),
        )
        lc = cfg["lora"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=lc["r"],
            target_modules=lc["target_modules"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc.get("dropout", 0.0),
            bias=lc.get("bias", "none"),
            use_gradient_checkpointing=lc.get("use_gradient_checkpointing", "unsloth"),
            random_state=lc.get("random_state", 3407),
            use_rslora=lc.get("use_rslora", False),
        )
    else:
        if not adapter.get("enabled", False):
            raise RuntimeError("Set model.merged_base=true OR resume_adapter.enabled=true")
        repo = adapter["repo"]
        cfg_path = Path(repo) / "adapter_config.json"
        if cfg_path.exists():
            ac = json.loads(cfg_path.read_text(encoding="utf-8"))
            if ac.get("inference_mode") is True:
                raise RuntimeError(f"{cfg_path} has inference_mode=true. Flip to false first.")
        banner(f"Loading SFT adapter with Unsloth: {repo}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=repo,
            max_seq_length=m["max_seq_length"],
            dtype=m["dtype"],
            load_in_4bit=m["load_in_4bit"],
            token=os.environ.get("HF_TOKEN"),
        )
        ensure_trainable_adapter(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Force the SFT-matching template (NOT the stock Instruct template, which
    # injects a date preamble the model never saw during SFT).
    tokenizer.chat_template = SFT_CHAT_TEMPLATE
    print("  set chat_template = SFT_CHAT_TEMPLATE (exact format_example match)")

    model.train()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"  trainable params: {n_train / 1e6:.1f}M / {n_total / 1e9:.2f}B "
        f"({100 * n_train / n_total:.3f}%)"
    )
    return model, tokenizer


def _to_conversational(rows: list[dict], system_prompt: str) -> list[dict]:
    """Wrap string prompt/chosen/rejected into conversational format so TRL
    applies SFT_CHAT_TEMPLATE. User turn is the bare question (matches SFT's
    no-topic branch and the inference-time prompt). Already-conversational rows
    (list-of-messages) are passed through untouched."""
    out = []
    sys_msg = [{"role": "system", "content": system_prompt}] if system_prompt else []
    for r in rows:
        p, c, rj = r["prompt"], r["chosen"], r["rejected"]
        if isinstance(p, list):  # already conversational
            out.append(r)
            continue
        out.append({
            "prompt": sys_msg + [{"role": "user", "content": p}],
            "chosen": [{"role": "assistant", "content": c}],
            "rejected": [{"role": "assistant", "content": rj}],
        })
    return out


def build_datasets(cfg: dict, smoke_limit: int = 0):
    d = cfg["data"]
    path = Path(d["train_path"])
    banner(f"Loading DPO pairs from {path}")
    rows = load_pairs(path, smoke_limit=smoke_limit)
    print(f"  loaded {len(rows):,} valid pairs")
    system_prompt = str(d.get("system_prompt", "")).strip()
    rows = _to_conversational(rows, system_prompt)
    print(f"  converted to conversational (system_prompt={'yes' if system_prompt else 'no'})")
    train, val = split_train_val(rows, d["val_fraction"], cfg["training"]["seed"])
    print(f"  train: {len(train):,}  val: {len(val):,}")
    return Dataset.from_list(train), Dataset.from_list(val)


def build_dpo_args(cfg: dict) -> DPOConfig:
    t = cfg["training"]
    s = cfg["schedule"]
    h = cfg.get("hub", {})
    tb = cfg.get("tensorboard", {})

    bf16 = t.get("bf16", False) and is_bfloat16_supported()
    fp16 = t.get("fp16", False) and not bf16
    report_to = ["tensorboard"] if tb.get("enabled", True) else []

    kwargs = {}
    if cfg.get("_max_steps") is not None:
        kwargs["max_steps"] = cfg["_max_steps"]

    return DPOConfig(
        output_dir=t["output_dir"],
        overwrite_output_dir=False,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        bf16=bf16,
        fp16=fp16,
        beta=t["beta"],
        loss_type=t["loss_type"],
        max_length=t["max_length"],
        max_prompt_length=t["max_prompt_length"],
        precompute_ref_log_probs=t.get("precompute_ref_log_probs", False),
        logging_steps=s["logging_steps"],
        eval_strategy="steps",
        eval_steps=s["eval_steps"],
        save_strategy="steps",
        save_steps=s["save_steps"],
        save_total_limit=s["save_total_limit"],
        logging_dir=tb.get("logdir", "./runs/dpo"),
        report_to=report_to,
        push_to_hub=h.get("push_to_hub", False),
        hub_model_id=h.get("hub_model_id"),
        hub_strategy=h.get("hub_strategy", "every_save"),
        hub_private_repo=h.get("hub_private_repo", True),
        hub_token=os.environ.get("HF_TOKEN"),
        dataloader_num_workers=t["dataloader_num_workers"],
        seed=t["seed"],
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataset_num_proc=t["dataset_num_proc"],
        **kwargs,
    )


def find_resume_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    candidates = []
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
            except ValueError:
                continue
            if (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists():
                candidates.append((step, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/DPO/dpo_config.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--smoke-limit", type=int, default=0)
    p.add_argument("--skip-final-eval", action="store_true")
    args = p.parse_args(argv)

    load_env_dotenv()
    cfg = load_config(Path(args.config))
    set_seed(cfg["training"]["seed"])

    if args.max_steps is not None:
        cfg["_max_steps"] = args.max_steps
    if args.smoke_limit or args.max_steps is not None:
        cfg["_skip_final_eval"] = True
        cfg.setdefault("hub", {})["push_to_hub"] = False

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    banner("Hardware")
    print(f"  device : {torch.cuda.get_device_name(0)}")
    print(f"  vram   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    print(f"  bf16   : {is_bfloat16_supported()}")

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds, val_ds = build_datasets(cfg, smoke_limit=args.smoke_limit)
    dpo_args = build_dpo_args(cfg)

    banner("Constructing DPOTrainer (ref_model=None: SFT adapter disabled for ref pass)")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    resume_from = None
    if args.resume:
        resume_from = find_resume_checkpoint(Path(dpo_args.output_dir))
        if resume_from is None:
            print(f"[resume] no checkpoint under {dpo_args.output_dir}; starting fresh")
        else:
            print(f"[resume] resuming from {resume_from}")

    effective_batch = dpo_args.per_device_train_batch_size * dpo_args.gradient_accumulation_steps
    banner("Effective batch + sequence math")
    print(f"  per_device_batch : {dpo_args.per_device_train_batch_size}")
    print(f"  grad_accum_steps : {dpo_args.gradient_accumulation_steps}")
    print(f"  effective batch  : {effective_batch}")
    print(f"  max_length       : {dpo_args.max_length}")
    print(f"  max_prompt_length: {dpo_args.max_prompt_length}")
    print(f"  beta             : {dpo_args.beta}")
    print(f"  max_steps override : {cfg.get('_max_steps', 'none')}")

    banner("Starting DPO")
    result = trainer.train(resume_from_checkpoint=resume_from)
    print(result)

    final_dir = Path(dpo_args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[done] final adapter saved to {final_dir}")

    if cfg.get("_skip_final_eval", False):
        print("[smoke] skipping final eval and hub push")
    else:
        banner("Final eval")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if dpo_args.push_to_hub and not cfg.get("_skip_final_eval", False):
        print("[hub] final push to HF Hub")
        trainer.push_to_hub(commit_message="final DPO adapter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
