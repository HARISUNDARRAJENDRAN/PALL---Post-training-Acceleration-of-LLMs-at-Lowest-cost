"""Training callbacks for Pall CPT.

Plugged into HuggingFace Trainer:
  - GpuMemoryCallback     : VRAM used / reserved, every logging_steps
  - ThroughputCallback    : tokens/sec, samples/sec, ETA
  - GradientNormCallback  : gradient norm at each logging step
  - MCQEvalCallback       : runs MCQ accuracy on evals/text_only/medmcqa_dental_mcq.jsonl
                            every N steps (used after baseline measurement)

All metrics are forwarded into the Trainer's `log` mechanism, so they show up
in TensorBoard automatically.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class GpuMemoryCallback(TrainerCallback):
    """Logs GPU memory (used + reserved) at every logging step."""

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
        logs: dict | None = None, **kwargs,
    ):
        if logs is None or not torch.cuda.is_available():
            return
        used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logs["gpu/mem_used_gb"] = round(used_gb, 2)
        logs["gpu/mem_reserved_gb"] = round(reserved_gb, 2)
        logs["gpu/mem_peak_gb"] = round(peak_gb, 2)


class ThroughputCallback(TrainerCallback):
    """Logs tokens/sec and samples/sec, plus a running ETA."""

    def __init__(self, max_seq_length: int):
        self.max_seq_length = max_seq_length
        self.start_time: float | None = None
        self.last_step = 0
        self.last_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.last_time is None or self.start_time is None:
            return
        now = time.time()
        dt = now - self.last_time
        d_steps = state.global_step - self.last_step
        if dt <= 0 or d_steps <= 0:
            return

        effective_batch = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * max(1, args.world_size if hasattr(args, "world_size") else 1)
        )
        tokens_per_step = effective_batch * self.max_seq_length
        steps_per_sec = d_steps / dt
        samples_per_sec = effective_batch * steps_per_sec
        tokens_per_sec = tokens_per_step * steps_per_sec

        logs["throughput/tokens_per_sec"] = round(tokens_per_sec, 1)
        logs["throughput/samples_per_sec"] = round(samples_per_sec, 2)
        logs["throughput/steps_per_sec"] = round(steps_per_sec, 3)

        # ETA
        if state.max_steps and state.max_steps > 0:
            remaining = state.max_steps - state.global_step
            eta_sec = remaining / max(steps_per_sec, 1e-9)
            logs["throughput/eta_min"] = round(eta_sec / 60, 1)

        self.last_time = now
        self.last_step = state.global_step


class GradientNormCallback(TrainerCallback):
    """No-op placeholder.

    HuggingFace Trainer already logs `grad_norm` to the `logs` dict when
    `max_grad_norm > 0` and `optim != adafactor`. This class exists so the
    intent is explicit in cpt_train.py and we can extend it later (e.g.
    per-layer norms) without changing the trainer wiring.
    """

    pass


class MCQEvalCallback(TrainerCallback):
    """Cheap MCQ accuracy check on a small held-out dental set.

    Computes log-probability of each option (A/B/C/D) given the question and
    picks the argmax. Avoids generation (faster). Logs as `eval_mcq/accuracy`.

    Disabled by default (mcq_eval.enabled=false). Turn on once you have a
    baseline number for the untouched model.
    """

    def __init__(
        self,
        eval_path: str | Path,
        tokenizer,
        every_n_steps: int = 1000,
        max_samples: int = 100,
    ):
        self.eval_path = Path(eval_path)
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.max_samples = max_samples
        self._items: list[dict] | None = None

    def _load_items(self) -> list[dict]:
        if self._items is not None:
            return self._items
        items: list[dict] = []
        if not self.eval_path.exists():
            print(f"[mcq_eval] WARN: {self.eval_path} not found, callback disabled")
            self._items = []
            return self._items
        with self.eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "question" in r and "options" in r and "answer_index" in r:
                    items.append(r)
                if len(items) >= self.max_samples:
                    break
        print(f"[mcq_eval] loaded {len(items)} MCQ items from {self.eval_path}")
        self._items = items
        return items

    @torch.no_grad()
    def _score(self, model, items: list[dict]) -> float:
        if not items:
            return float("nan")
        model.eval()
        n_correct = 0
        for it in items:
            q = it["question"]
            options = it["options"]
            gold = it["answer_index"]
            losses = []
            for opt in options:
                prompt = f"Question: {q}\nAnswer: {opt}"
                enc = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512,
                ).to(model.device)
                out = model(**enc, labels=enc["input_ids"])
                losses.append(out.loss.item())
            pred = int(min(range(len(losses)), key=lambda i: losses[i]))
            if pred == gold:
                n_correct += 1
        model.train()
        return n_correct / len(items)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        items = self._load_items()
        if not items:
            return
        model = kwargs.get("model")
        if model is None:
            return
        acc = self._score(model, items)
        if state.log_history is not None:
            state.log_history.append({"step": step, "eval_mcq/accuracy": acc})
        print(f"[mcq_eval] step {step}: accuracy = {acc:.3f} on {len(items)} items")
