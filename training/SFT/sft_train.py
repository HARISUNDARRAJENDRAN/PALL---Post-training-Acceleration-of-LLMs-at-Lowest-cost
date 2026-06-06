"""DocSmile supervised fine-tuning (SFT) script.

Usage on a Vast.ai instance:
    python training/SFT/sft_train.py --config training/SFT/sft_config.yaml

Smoke benchmark:
    python training/SFT/sft_train.py --config training/SFT/sft_config.yaml --max-steps 100 --smoke-limit 5000

TensorBoard:
    tensorboard --logdir /workspace/runs --port 6006 --bind_all
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
from datasets import load_dataset
from peft import PeftModel
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

# callbacks.py lives one level up in training/ (shared across CPT/SFT/DPO)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from callbacks import (
    GpuMemoryCallback,
    GradientNormCallback,
    MCQEvalCallback,
    ThroughputCallback,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")


def validate_resume_adapter(repo: str) -> None:
    """Fail fast if a local CPT adapter is still saved as inference-only."""
    path = Path(repo)
    if not path.exists():
        return
    cfg_path = path / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Local adapter is missing adapter_config.json: {path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("inference_mode") is True:
        raise RuntimeError(
            f"{cfg_path} has inference_mode=true. Run "
            "`python training/make_trainable_adapter.py` on the server first, "
            "or point resume_adapter.repo at /workspace/cpt_adapter_trainable."
        )


def print_trainable_parameters(model) -> None:
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        return
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"  trainable params: {n_train / 1e6:.1f}M / {n_total / 1e9:.2f}B "
        f"({100 * n_train / max(n_total, 1):.3f}%)"
    )


def ensure_trainable_adapter(model) -> None:
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train > 0:
        return
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError("No trainable LoRA parameters found after loading CPT adapter.")
    print(f"  forced LoRA params trainable: {n_train / 1e6:.1f}M")


def apply_memory_speed_policy(model, cfg: dict) -> None:
    use_gc = cfg["lora"].get("use_gradient_checkpointing", False)
    if use_gc:
        return

    try:
        import torch.utils.checkpoint
        import transformers.modeling_utils
        from unsloth_zoo import gradient_checkpointing as gc

        if hasattr(gc, "unpatch_unsloth_gradient_checkpointing"):
            gc.unpatch_unsloth_gradient_checkpointing()
        if hasattr(gc, "unpatch_gradient_checkpointing"):
            gc.unpatch_gradient_checkpointing()
        if hasattr(gc, "unpatch_unsloth_smart_gradient_checkpointing"):
            gc.unpatch_unsloth_smart_gradient_checkpointing()
        if hasattr(torch.utils.checkpoint, "_old_checkpoint"):
            torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint
            del torch.utils.checkpoint._old_checkpoint
        transformers.modeling_utils.checkpoint = torch.utils.checkpoint.checkpoint
        os.environ.pop("UNSLOTH_PATCHED", None)
    except Exception as exc:
        print(f"  WARN: could not fully unpatch Unsloth gradient checkpointing: {exc}")

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    for obj in (model, base, getattr(base, "model", None) if base is not None else None):
        if obj is not None and hasattr(obj, "config"):
            obj.config.use_cache = False
        if obj is not None and hasattr(obj, "gradient_checkpointing"):
            obj.gradient_checkpointing = False
    print("  gradient checkpointing: disabled (favor speed over VRAM)")


def build_model_and_tokenizer(cfg: dict):
    m = cfg["model"]
    adapter = cfg.get("resume_adapter", {})
    if adapter.get("enabled", False):
        repo = adapter["repo"]
        validate_resume_adapter(repo)
        method = adapter.get("load_method", "unsloth")
        if method == "unsloth":
            banner(f"Loading CPT adapter with Unsloth {repo}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=repo,
                max_seq_length=m["max_seq_length"],
                dtype=m["dtype"],
                load_in_4bit=m["load_in_4bit"],
                token=os.environ.get("HF_TOKEN"),
            )
        elif method == "peft":
            banner(f"Loading {m['name']} (4-bit={m['load_in_4bit']}, ctx={m['max_seq_length']})")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=m["name"],
                max_seq_length=m["max_seq_length"],
                dtype=m["dtype"],
                load_in_4bit=m["load_in_4bit"],
            )
            banner(f"Loading CPT adapter with PEFT {repo}")
            model = PeftModel.from_pretrained(
                model,
                repo,
                token=os.environ.get("HF_TOKEN"),
                is_trainable=True,
            )
            if hasattr(FastLanguageModel, "patch_peft_model"):
                model = FastLanguageModel.patch_peft_model(
                    model,
                    use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
                )
        else:
            raise ValueError(f"Unsupported resume_adapter.load_method: {method}")
        apply_memory_speed_policy(model, cfg)
        ensure_trainable_adapter(model)
        print_trainable_parameters(model)
    else:
        banner(f"Loading {m['name']} (4-bit={m['load_in_4bit']}, ctx={m['max_seq_length']})")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=m["name"],
            max_seq_length=m["max_seq_length"],
            dtype=m["dtype"],
            load_in_4bit=m["load_in_4bit"],
        )
        l = cfg["lora"]
        banner(
            f"Attaching LoRA r={l['r']} alpha={l['alpha']} "
            f"drop={l['dropout']} targets={len(l['target_modules'])}"
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=l["r"],
            target_modules=l["target_modules"],
            lora_alpha=l["alpha"],
            lora_dropout=l["dropout"],
            bias=l["bias"],
            use_gradient_checkpointing=l["use_gradient_checkpointing"],
            use_rslora=l["use_rslora"],
            random_state=l["random_state"],
        )
        apply_memory_speed_policy(model, cfg)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"  trainable params: {n_train / 1e6:.1f}M / {n_total / 1e9:.2f}B "
            f"({100 * n_train / n_total:.3f}%)"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.train()
    return model, tokenizer


def format_example(row: dict, cfg: dict) -> str:
    d = cfg["data"]
    question = str(row.get(d["question_field"], "")).strip()
    answer = str(row.get(d["answer_field"], "")).strip()
    topic = str(row.get(d.get("topic_field", "topic"), "")).strip()
    system = str(d.get("system_prompt", "")).strip()

    user = question
    if topic:
        user = f"Topic: {topic}\n\n{question}"

    if system:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{answer}<|eot_id|>"
        )
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{answer}<|eot_id|>"
    )


def build_datasets(cfg: dict):
    d = cfg["data"]
    if d.get("train_file"):
        train_file = str(Path(d["train_file"]).expanduser())
        val_file = str(Path(d["val_file"]).expanduser())
        banner(f"Loading local JSONL dataset {train_file}")
        train_ds = load_dataset("json", data_files=train_file, split="train")
        val_ds = load_dataset("json", data_files=val_file, split="train")
    else:
        banner(f"Loading dataset {d['repo']} (config={d['config']})")
        train_ds = load_dataset(
            d["repo"], d["config"], split=d["train_split"], token=os.environ.get("HF_TOKEN")
        )
        val_ds = load_dataset(
            d["repo"], d["config"], split=d["val_split"], token=os.environ.get("HF_TOKEN")
        )

    smoke_n = cfg.get("_smoke_limit", 0)
    if smoke_n and smoke_n > 0 and smoke_n < len(train_ds):
        print(f"  [smoke] truncating train: {len(train_ds):,} -> {smoke_n:,} rows")
        train_ds = train_ds.select(range(smoke_n))

    text_field = cfg["data"]["text_field"]
    num_proc = cfg["training"]["dataset_num_proc"]

    def _format_batch(batch: dict) -> dict:
        n = len(next(iter(batch.values())))
        return {
            text_field: [
                format_example({k: v[i] for k, v in batch.items()}, cfg)
                for i in range(n)
            ]
        }

    train_ds = train_ds.map(
        _format_batch,
        batched=True,
        num_proc=num_proc if len(train_ds) > 1 else None,
        remove_columns=[c for c in train_ds.column_names if c != text_field],
    )
    val_ds = val_ds.map(
        _format_batch,
        batched=True,
        num_proc=num_proc if len(val_ds) > 1 else None,
        remove_columns=[c for c in val_ds.column_names if c != text_field],
    )

    print(f"  train rows: {len(train_ds):,}")
    print(f"  val   rows: {len(val_ds):,}")
    print(f"  columns   : {train_ds.column_names}")
    return train_ds, val_ds


def build_training_args(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    s = cfg["schedule"]
    d = cfg["data"]
    h = cfg.get("hub", {})
    tb = cfg.get("tensorboard", {})

    bf16 = t.get("bf16", False) and is_bfloat16_supported()
    fp16 = t.get("fp16", False) and not bf16
    report_to = ["tensorboard"] if tb.get("enabled", True) else []

    kwargs = {}
    if cfg.get("_max_steps") is not None:
        kwargs["max_steps"] = cfg["_max_steps"]

    return SFTConfig(
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
        logging_steps=s["logging_steps"],
        eval_strategy="steps",
        eval_steps=s["eval_steps"],
        save_strategy="steps",
        save_steps=s["save_steps"],
        save_total_limit=s["save_total_limit"],
        logging_dir=tb.get("logdir", "./runs/sft"),
        report_to=report_to,
        push_to_hub=h.get("push_to_hub", False),
        hub_model_id=h.get("hub_model_id"),
        hub_strategy=h.get("hub_strategy", "every_save"),
        hub_private_repo=h.get("hub_private_repo", True),
        hub_token=os.environ.get("HF_TOKEN"),
        dataloader_num_workers=t["dataloader_num_workers"],
        seed=t["seed"],
        load_best_model_at_end=False,
        remove_unused_columns=True,
        group_by_length=t["group_by_length"],
        dataset_text_field=d["text_field"],
        max_seq_length=cfg["model"]["max_seq_length"],
        packing=t["packing"],
        dataset_num_proc=t["dataset_num_proc"],
        **kwargs,
    )


def build_trainer(cfg: dict, model, tokenizer, train_ds, val_ds, args):
    callbacks = [
        GpuMemoryCallback(),
        ThroughputCallback(max_seq_length=cfg["model"]["max_seq_length"]),
        GradientNormCallback(),
    ]
    mcq = cfg.get("mcq_eval", {})
    if mcq.get("enabled", False):
        eval_path = Path(__file__).parent / mcq["path"]
        callbacks.append(MCQEvalCallback(
            eval_path=eval_path.resolve(),
            tokenizer=tokenizer,
            every_n_steps=mcq.get("every_n_steps", 1000),
            max_samples=mcq.get("max_samples", 100),
        ))

    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        callbacks=callbacks,
    )


def find_resume_checkpoint(output_dir: Path) -> Path | None:
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
    p.add_argument("--config", default="training/SFT/sft_config.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--logdir", default=None)
    p.add_argument("--smoke-limit", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip expensive final validation eval; useful for smoke and throughput tests.",
    )
    args = p.parse_args(argv)

    load_env_dotenv()
    cfg = load_config(Path(args.config))
    set_seed(cfg["training"]["seed"])
    if cfg["training"].get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.logdir:
        cfg.setdefault("tensorboard", {})["logdir"] = args.logdir
    if args.smoke_limit and args.smoke_limit > 0:
        cfg["_smoke_limit"] = args.smoke_limit
    if args.max_steps is not None:
        cfg["_max_steps"] = args.max_steps
    if args.skip_final_eval or args.max_steps is not None:
        cfg["_skip_final_eval"] = True
        cfg.setdefault("hub", {})["push_to_hub"] = False

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    banner("Hardware")
    print(f"  device : {torch.cuda.get_device_name(0)}")
    print(f"  vram   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    print(f"  bf16   : {is_bfloat16_supported()}")
    print("  flash_attn : ", end="")
    try:
        import flash_attn  # noqa: F401
        print("yes")
    except ImportError:
        print("no")

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds, val_ds = build_datasets(cfg)
    targs = build_training_args(cfg)
    trainer = build_trainer(cfg, model, tokenizer, train_ds, val_ds, targs)

    resume_from = None
    if args.resume:
        resume_from = find_resume_checkpoint(Path(targs.output_dir))
        if resume_from is None:
            print(f"[resume] no checkpoint under {targs.output_dir}; starting from scratch")
        else:
            print(f"[resume] resuming from {resume_from}")

    effective_batch = targs.per_device_train_batch_size * targs.gradient_accumulation_steps
    tokens_per_step = effective_batch * cfg["model"]["max_seq_length"]
    banner("Effective batch + token math")
    print(f"  per_device_batch    : {targs.per_device_train_batch_size}")
    print(f"  grad_accum_steps    : {targs.gradient_accumulation_steps}")
    print(f"  effective batch     : {effective_batch}")
    print(f"  max_seq_length      : {cfg['model']['max_seq_length']}")
    print(f"  tokens per opt step : {tokens_per_step:,}")
    print(f"  max_steps override  : {cfg.get('_max_steps', 'none')}")

    banner("Starting SFT")
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    print(train_result)

    final_dir = Path(targs.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[done] final adapter saved to {final_dir}")

    if cfg.get("_skip_final_eval", False):
        print("\n[smoke] skipping final eval and hub push")
    else:
        banner("Final eval")
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss")
        if loss is not None:
            import math as _math
            metrics["eval_perplexity"] = _math.exp(min(loss, 20.0))
            print(f"  eval_loss       = {loss:.4f}")
            print(f"  eval_perplexity = {metrics['eval_perplexity']:.2f}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if targs.push_to_hub and not cfg.get("_skip_final_eval", False):
        print("[hub] final push to HF Hub")
        trainer.push_to_hub(commit_message="final SFT adapter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
