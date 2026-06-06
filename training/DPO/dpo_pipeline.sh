#!/bin/bash
# DocSmile DPO pipeline: wait for rejgen completion, then export + train.
# Run inside a long-lived tmux session: tmux new -s dpopipe 'bash training/dpo_pipeline.sh 2>&1 | tee /workspace/dpopipe.log'

set -euo pipefail
cd /workspace/DocSmile

REJ_LOG=/workspace/rejgen.log
DPO_DATA=/workspace/cpt_prepared/long_dpo_generated.jsonl
LOG=/workspace/dpopipe.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Activate venv
source /venv/main/bin/activate

# ---------- Stage 1: wait for rejgen ----------
log "stage 1: waiting for rejgen completion"
while true; do
  if grep -q "^\[done\] processed" "$REJ_LOG" 2>/dev/null; then
    log "rejgen finished. Last lines:"
    tail -5 "$REJ_LOG" | sed 's/^/   /'
    break
  fi
  if ! tmux has-session -t rejgen 2>/dev/null; then
    # tmux died — check if file is final-sized
    LINES=$(wc -l < "$DPO_DATA" 2>/dev/null || echo 0)
    log "rejgen tmux gone; file has $LINES lines"
    if [ "$LINES" -gt 50000 ]; then
      log "treating rejgen as done"
      break
    fi
    log "ERROR: rejgen tmux gone but file incomplete ($LINES lines). Aborting."
    exit 1
  fi
  LINES=$(wc -l < "$DPO_DATA" 2>/dev/null || echo 0)
  log "  ... $LINES lines in $DPO_DATA"
  sleep 120
done

# ---------- Stage 2: free GPU ----------
log "stage 2: ensuring rejgen tmux is closed and GPU freed"
tmux kill-session -t rejgen 2>/dev/null || true
sleep 10
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# ---------- Stage 3: dataset stats ----------
log "stage 3: dataset sanity"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/workspace/cpt_prepared/long_dpo_generated.jsonl")
n_total=n_same=n_short=n_empty=0
src_counts={}
with p.open("r",encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r=json.loads(line)
        n_total+=1
        c=(r.get("chosen") or "").strip()
        rj=(r.get("rejected") or "").strip()
        if not c or not rj: n_empty+=1; continue
        if c==rj: n_same+=1
        if len(rj)<50: n_short+=1
        src=r.get("source","unknown")
        src_counts[src]=src_counts.get(src,0)+1
print(f"  total={n_total:,}")
print(f"  same-chosen-rejected={n_same:,}")
print(f"  short-rejected(<50ch)={n_short:,}")
print(f"  empty={n_empty:,}")
print("  top sources:")
for k,v in sorted(src_counts.items(),key=lambda x:-x[1])[:10]:
    print(f"    {k}: {v:,}")
PY

# ---------- Stage 4: export to HF (non-fatal) ----------
log "stage 4: exporting DPO dataset to HF Hub"
set +e
python3 - <<'PY'
import os, json
from pathlib import Path
from huggingface_hub import HfApi
envp = Path("/workspace/DocSmile/training/.env")
for line in envp.read_text(encoding="utf-8").splitlines():
    line=line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k,_,v=line.partition("=")
    os.environ[k.strip()]=v.strip().strip('"').strip("'")

api=HfApi(token=os.environ["HF_TOKEN"])
repo=os.environ.get("HF_DATA_REPO","Harisundar/pall")
src="/workspace/cpt_prepared/long_dpo_generated.jsonl"
dest="dpo/long_dpo_generated.jsonl"
print(f"  uploading {src} -> {repo}/{dest}")
api.upload_file(
    path_or_fileobj=src,
    path_in_repo=dest,
    repo_id=repo,
    repo_type="dataset",
    commit_message="DPO: SFT-generated rejected responses on 50k long prompts",
)
print("  upload done")
PY
UPLOAD_RC=$?
set -e
if [ "$UPLOAD_RC" -ne 0 ]; then
  log "WARN: HF upload failed (rc=$UPLOAD_RC). Local file kept. Continuing to training."
fi

# ---------- Stage 5: smoke test ----------
log "stage 5: preparing trainable SFT adapter copy"
python3 - <<'PY'
import json, shutil
from pathlib import Path
src = Path("/workspace/checkpoints/sft/final")
dst = Path("/workspace/sft_adapter_trainable")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
cfg_path = dst / "adapter_config.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
cfg["inference_mode"] = False
cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"  prepared {dst} (inference_mode=False)")
PY

log "stage 5b: DPO smoke (5 steps, 200 rows)"
python3 training/DPO/dpo_train.py \
  --config training/DPO/dpo_config.yaml \
  --max-steps 5 \
  --smoke-limit 200 \
  --skip-final-eval 2>&1 | tail -50
SMOKE_RC=${PIPESTATUS[0]}
if [ "$SMOKE_RC" -ne 0 ]; then
  log "ERROR: smoke test failed (rc=$SMOKE_RC). Aborting full training."
  exit "$SMOKE_RC"
fi
log "smoke OK"

# ---------- Stage 6: full DPO ----------
log "stage 6: launching full DPO training in tmux 'dpo'"
tmux kill-session -t dpo 2>/dev/null || true
tmux new-session -d -s dpo "bash -c 'cd /workspace/DocSmile && source /venv/main/bin/activate && python3 training/DPO/dpo_train.py --config training/DPO/dpo_config.yaml 2>&1 | tee /workspace/dpo.log; sleep infinity'"
log "DPO launched. Watch: tail -f /workspace/dpo.log"
log "pipeline done."
