"""One-time: upload FINAL/* JSONL files to a private HuggingFace dataset repo.

Run from your laptop:
    cd training
    cp .env.example .env       # fill in HF_TOKEN, HF_USERNAME, HF_DATA_REPO
    python prepare_data.py

This creates a single private dataset repo with three "configs":
    cpt     -> dental_cpt_final.jsonl                  (1 split: train)
    sft     -> dental_sft_final_{train,val}.jsonl       (2 splits)
    dpo     -> dental_dpo_final_{train,val}.jsonl       (2 splits)
    cptonly -> dental_cpt_final_{train,val}.jsonl      (2 splits)

On the Vast.ai instance, you then read with:
    from datasets import load_dataset
    ds = load_dataset("Harisundar/pall-dental", "cpt", split="train")

Why this layout: one dataset repo, three configs, keeps the HF Hub view clean
and the training script can pick the subset it needs.

Also carves a small validation slice (1%) out of dental_cpt_final.jsonl
locally so CPT training can compute eval perplexity without a separate file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "FINAL"


def _read_env(path: Path) -> dict:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _load_env() -> dict:
    """Load env from .env then OS environ.

    `.env` wins over shell env vars to avoid stale-shell-token foot-guns
    (e.g. an old HF_TOKEN exported in PowerShell shadowing a freshly-rotated
    token in .env). If .env is missing a key, fall back to the shell env.
    """
    env = _read_env(Path(__file__).parent / ".env")
    for k in ("HF_TOKEN", "HF_USERNAME", "HF_DATA_REPO"):
        if not env.get(k) and os.environ.get(k):
            env[k] = os.environ[k]
    return env


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _carve_cpt_val(cpt_path: Path, val_frac: float, seed: int) -> tuple[Path, Path]:
    """Split CPT into train/val on disk under FINAL/."""
    train_out = FINAL / "dental_cpt_final_train.jsonl"
    val_out = FINAL / "dental_cpt_final_val.jsonl"
    if train_out.exists() and val_out.exists():
        return train_out, val_out

    print(f"[carve] splitting CPT: val_frac={val_frac} seed={seed}")
    rng = random.Random(seed)
    n_total = _count_lines(cpt_path)
    n_val = max(100, int(n_total * val_frac))
    val_indices = set(rng.sample(range(n_total), n_val))
    print(f"[carve] total={n_total} val={n_val} train={n_total - n_val}")

    with cpt_path.open("r", encoding="utf-8") as fin, \
         train_out.open("w", encoding="utf-8") as ftrain, \
         val_out.open("w", encoding="utf-8") as fval:
        for i, line in enumerate(fin):
            if i in val_indices:
                fval.write(line)
            else:
                ftrain.write(line)
    return train_out, val_out


def upload(
    env: dict,
    dry_run: bool = False,
    val_frac: float = 0.01,
    seed: int = 3407,
    subset: str = "all",
) -> int:
    token = env.get("HF_TOKEN", "").strip()
    user = env.get("HF_USERNAME", "").strip()
    repo = env.get("HF_DATA_REPO", "").strip()

    if not token:
        print("ERROR: HF_TOKEN not set in .env or environment", file=sys.stderr)
        return 1
    if not repo:
        if not user:
            print("ERROR: HF_DATA_REPO or HF_USERNAME must be set", file=sys.stderr)
            return 1
        repo = f"{user}/pall-dental"
        print(f"[prepare] using default repo: {repo}")

    subset = subset.lower().strip()
    if subset not in {"all", "sft", "cpt", "dpo", "cptonly"}:
        print("ERROR: subset must be one of all, sft, cpt, dpo, cptonly", file=sys.stderr)
        return 1

    files_to_upload: list[tuple[Path, str]] = [(FINAL / "report.json", "report.json")]
    if subset in {"all", "cpt", "cptonly"}:
        cpt_path = FINAL / "dental_cpt_final.jsonl"
        if not cpt_path.exists():
            print(f"ERROR: {cpt_path} missing - run final_merge.py first", file=sys.stderr)
            return 1
        cpt_train, cpt_val = _carve_cpt_val(cpt_path, val_frac=val_frac, seed=seed)
        files_to_upload.extend([
            (cpt_train, "cpt/train.jsonl"),
            (cpt_val, "cpt/validation.jsonl"),
        ])
    if subset in {"all", "sft"}:
        files_to_upload.extend([
            (FINAL / "dental_sft_final_train.jsonl", "sft/train.jsonl"),
            (FINAL / "dental_sft_final_val.jsonl", "sft/validation.jsonl"),
        ])
    if subset in {"all", "dpo"}:
        files_to_upload.extend([
            (FINAL / "dental_dpo_final_train.jsonl", "dpo/train.jsonl"),
            (FINAL / "dental_dpo_final_val.jsonl", "dpo/validation.jsonl"),
        ])

    print("\n[prepare] files to upload:")
    total_bytes = 0
    for local, remote in files_to_upload:
        if not local.exists():
            print(f"  [MISSING]  {local} -> {remote}")
            continue
        size_mb = local.stat().st_size / (1024 * 1024)
        total_bytes += local.stat().st_size
        print(f"  {size_mb:>8.1f} MB  {local.name:<35} -> {remote}")
    print(f"\n  total: {total_bytes / (1024 * 1024):.1f} MB")

    if dry_run:
        print("\n[prepare] --dry-run: not uploading. Run without --dry-run to push.")
        return 0

    # Lazy import: only need huggingface_hub when actually uploading.
    from huggingface_hub import HfApi, create_repo, hf_hub_url

    api = HfApi(token=token)

    # Create or update repo
    print(f"\n[prepare] ensuring repo exists: {repo} (private)")
    create_repo(
        repo_id=repo, repo_type="dataset",
        private=True, exist_ok=True, token=token,
    )

    # README written alongside the data so the HF dataset page is informative
    readme = _build_readme(env)
    readme_local = FINAL / "_dataset_README.md"
    readme_local.write_text(readme, encoding="utf-8")

    print(f"[prepare] pushing files (this can take 5-20 min on slow uplink)")
    t0 = time.time()
    for local, remote in files_to_upload:
        if not local.exists():
            continue
        print(f"  -> {remote}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=repo,
            repo_type="dataset",
            token=token,
        )
    api.upload_file(
        path_or_fileobj=str(readme_local),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        token=token,
    )
    print(f"[prepare] DONE in {time.time() - t0:.0f}s  https://huggingface.co/datasets/{repo}")
    return 0


def _build_readme(env: dict) -> str:
    return f"""---
license: cc-by-nc-4.0
language: [en]
task_categories:
  - text-generation
  - question-answering
tags:
  - medical
  - dental
  - llm
configs:
  - config_name: cpt
    data_files:
      - split: train
        path: cpt/train.jsonl
      - split: validation
        path: cpt/validation.jsonl
  - config_name: sft
    data_files:
      - split: train
        path: sft/train.jsonl
      - split: validation
        path: sft/validation.jsonl
  - config_name: dpo
    data_files:
      - split: train
        path: dpo/train.jsonl
      - split: validation
        path: dpo/validation.jsonl
---

# DocSmile dental training corpus

Private working corpus for DocSmile dental LLM training.

  - `cpt`: continued-pretraining text chunks. Schema: `{{text, source}}`.
  - `sft`: supervised fine-tuning Q&A. Schema: `{{question, answer, source, topic}}`.
  - `dpo`: direct preference optimization pairs. Schema: `{{prompt, chosen, rejected, source, topic}}`.

Sources include PubMed abstracts, PMC open-access full text, OpenAlex dental
works, Wikipedia dental articles, 113 dental textbooks (cleaned), MedMCQA dental,
ChatDoctor dental, PubMedQA dental, ClinicalTrials.gov dental studies, StatPearls
dental chapters, MeSH-synthetic Q&A, and Gemini-generated textbook Q&A.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Upload FINAL/* to HuggingFace Hub")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without uploading")
    p.add_argument("--val-frac", type=float, default=0.01,
                   help="Fraction of CPT to carve for validation perplexity")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--subset", default="all",
                   help="Upload subset: all, sft, cpt, dpo, or cptonly")
    args = p.parse_args(argv)

    env = _load_env()
    return upload(
        env,
        dry_run=args.dry_run,
        val_frac=args.val_frac,
        seed=args.seed,
        subset=args.subset,
    )


if __name__ == "__main__":
    sys.exit(main())
