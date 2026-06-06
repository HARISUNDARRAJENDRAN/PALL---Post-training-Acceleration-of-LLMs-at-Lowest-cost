# PALL datasets

All three training stages draw from a single Hugging Face dataset repo,
[`Harisundar/pall`](https://huggingface.co/datasets/Harisundar/pall), split into
per-stage subsets. The raw data is **not** stored in git — it lives on the Hub and is
pulled on demand.

> **Access:** the repo is currently **private**. Set `HF_TOKEN` (read access) in your
> environment or project `.env` before downloading. It will be made public after
> deployment, at which point the token is no longer required.

## Layout on the Hub

```
Harisundar/pall  (dataset, private)
├── cpt/train.jsonl          # 401,900 rows   continued-pretraining corpus (~175M tokens)
├── cpt/validation.jsonl     #   4,059 rows   1% held-out for perplexity
├── sft/train.jsonl          # 391,693 rows   instruction/response pairs
├── sft/validation.jsonl     #  20,615 rows
├── dpo/train.jsonl          #  10,185 rows   prompt / chosen / rejected triplets
├── dpo/validation.jsonl     #     536 rows
└── report.json              # prepared-data provenance/statistics
```

Row counts above are the actual trained-on splits (verified against the prepared-data
manifest in `manifests/dataset_report.json`).

## Downloading

The trainers in [`../training`](../training) load these subsets **directly** from the
Hub (see the `data.repo` / `data.config` keys in each YAML config), so you generally do
not need to download anything to train.

To fetch the files locally for inspection or offline use:

```bash
python data/download.py                  # all stages -> data/{cpt,sft,dpo}/
python data/download.py --stage cpt      # one stage
python data/download.py --out /scratch   # custom destination
```

## Stage → subset mapping

| Stage | Subset | Schema | Consumed by |
|-------|--------|--------|-------------|
| CPT | `cpt` | `{ "text": ... }` | `training/CPT/cpt_train.py` (`CPT/cpt_config.yaml`) |
| SFT | `sft` | `{ "messages": [user, assistant] }` | `training/SFT/sft_train.py` (`SFT/sft_v2_a100_config.yaml`) |
| DPO | `dpo` | `{ "prompt", "chosen", "rejected" }` | `training/DPO/dpo_train.py` (`DPO/dpo_v3_config.yaml`) |

## VLM data

The multimodal stage (`../vlm`) trains on a separate image+text corpus (32,884 records /
52,461 images). Only its manifest is checked in here
(`manifests/vlm_manifest.json`); the images themselves are too large for git and are
hosted separately. See [`../vlm/README.md`](../vlm/README.md) for the expected
`vlm_train/` layout.

## `manifests/`

Small JSON provenance files kept in git for reference (no bulk data):

- `dataset_report.json` — per-source / per-file counts for the CPT/SFT/DPO corpora.
- `vlm_manifest.json` — source distribution and split sizes for the VLM dataset.
