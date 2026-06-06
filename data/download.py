#!/usr/bin/env python
"""Download the PALL dental datasets from the Hugging Face Hub into ./data.

The three training stages (CPT, SFT, DPO) all live as subsets of a single dataset
repo, `Harisundar/pall`:

    Harisundar/pall
    ├── cpt/{train,validation}.jsonl
    ├── sft/{train,validation}.jsonl
    └── dpo/{train,validation}.jsonl

The trainers in ../training load these directly via `load_dataset` (see the
`dataset.repo` / `dataset.config` keys in each YAML config), so you usually do NOT
need to run this script to train. It exists for offline use, inspection, and to
make the data layout reproducible.

Auth: the repo is currently PRIVATE. Set HF_TOKEN in your environment (or .env) with
read access. Once the repo is made public the token becomes optional and this script
works unchanged.

Usage:
    python data/download.py                  # all stages -> ./data/<stage>/
    python data/download.py --stage cpt sft  # subset
    python data/download.py --out /some/dir
"""
import argparse
import os
from pathlib import Path

REPO_ID = "Harisundar/pall"
STAGES = {
    "cpt": ["cpt/train.jsonl", "cpt/validation.jsonl"],
    "sft": ["sft/train.jsonl", "sft/validation.jsonl"],
    "dpo": ["dpo/train.jsonl", "dpo/validation.jsonl"],
}


def _load_dotenv_token() -> str | None:
    """Fall back to HF_TOKEN from a sibling .env if not already in the environment."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for env_path in (Path(__file__).resolve().parent.parent / ".env",
                     Path(__file__).resolve().parent / ".env"):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip().lstrip("export ").strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=list(STAGES), default=list(STAGES),
                    help="which stage subsets to fetch (default: all)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent),
                    help="output directory (default: ./data)")
    ap.add_argument("--repo", default=REPO_ID)
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download

    token = _load_dotenv_token()
    if token is None:
        print("[warn] no HF_TOKEN found; this will only work once the repo is public.")

    out = Path(a.out)
    for stage in a.stage:
        for rel in STAGES[stage]:
            print(f"[get] {a.repo}:{rel}")
            local = hf_hub_download(
                repo_id=a.repo, filename=rel, repo_type="dataset",
                token=token, local_dir=str(out),
            )
            sz = Path(local).stat().st_size / 1e6
            print(f"      -> {local}  ({sz:.1f} MB)")
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
