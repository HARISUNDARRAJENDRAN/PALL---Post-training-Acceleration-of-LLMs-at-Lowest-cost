"""Generate rejected responses for DPO using vLLM batch inference.

vLLM with continuous batching gets ~2-4k tok/sec on A100 vs Unsloth's ~150.
We use vLLM's native LoRA support to avoid merging the adapter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def already_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["global_idx"])
            except Exception:
                continue
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--staging", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base", default="unsloth/Meta-Llama-3.1-8B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-frac", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=512, help="Submit rows to vLLM in chunks for periodic writes")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()
    load_env()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    done = already_done(output)
    print(f"[resume] {len(done):,} rows already in output, skipping those", flush=True)

    rows = []
    with open(args.staging, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["global_idx"] not in done:
                rows.append(r)
    if args.limit > 0:
        rows = rows[:args.limit]
    total = len(rows)
    print(f"[plan] generating {total:,} rows", flush=True)
    if total == 0:
        print("nothing to do.")
        return 0

    print(f"[vllm] loading base={args.base} with LoRA adapter={args.adapter}", flush=True)
    llm = LLM(
        model=args.base,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_frac,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_num_seqs=256,
        download_dir="/workspace/.hf_cache",
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    lora_req = LoRARequest("sft-adapter", 1, args.adapter)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new,
        stop_token_ids=[128009, 128001],
        skip_special_tokens=True,
    )

    f_out = output.open("a", encoding="utf-8")
    start = time.time()
    completed = 0

    for chunk_start in range(0, total, args.chunk):
        chunk = rows[chunk_start:chunk_start + args.chunk]
        prompts = [r["prompt"] for r in chunk]
        outs = llm.generate(prompts, sampling, lora_request=lora_req, use_tqdm=False)
        for r, o in zip(chunk, outs):
            text = o.outputs[0].text.strip()
            if not text:
                continue
            dpo = {
                "global_idx": r["global_idx"],
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": text,
                "source": r["source"],
                "topic": r.get("topic", ""),
                "n_tokens_chosen": r.get("n_tokens", 0),
            }
            f_out.write(json.dumps(dpo, ensure_ascii=False) + "\n")
            completed += 1
        f_out.flush()

        elapsed = time.time() - start
        rate = completed / max(elapsed, 1e-3)
        remaining = (total - completed) / max(rate, 1e-6)
        print(
            f"[{completed:>6,}/{total:,}] {rate:.1f} samples/sec | "
            f"elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m",
            flush=True,
        )

    f_out.close()
    print(f"[done] processed {completed:,} in {(time.time()-start)/60:.1f} min")
    print(f"[done] output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
