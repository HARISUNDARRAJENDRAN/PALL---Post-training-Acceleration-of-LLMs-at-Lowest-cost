#!/usr/bin/env python
"""Push the merged VLM to HuggingFace Hub.

Reads HF token from ~/.hf_token on the box.
"""
import os, sys, time, traceback
from huggingface_hub import HfApi, whoami

OUT = "/home/ec2-user/vlm/docsmile_vlm_merged"
REPO = "Harisundar/PALL-VLM"
TOKEN_FILE = os.path.expanduser("~/.hf_token")

LOG = "/home/ec2-user/push.log"
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

try:
    log(f"=== push start ===")
    log(f"OUT  = {OUT}")
    log(f"REPO = {REPO}")
    if not os.path.isdir(OUT):
        log(f"!! merged model dir not found at {OUT}; aborting")
        sys.exit(1)
    if not os.path.exists(TOKEN_FILE):
        log(f"!! no HF token at {TOKEN_FILE}; aborting")
        sys.exit(1)
    token = open(TOKEN_FILE).read().strip()
    log(f"  token length: {len(token)} chars")

    api = HfApi(token=token)
    me = whoami(token=token)
    log(f"  authenticated as: {me.get('name')} (id={me.get('id')})")

    log(f"  creating repo {REPO} (if missing)...")
    api.create_repo(repo_id=REPO, repo_type="model", private=False, exist_ok=True,
                    token=token)
    log(f"  repo ok")

    log(f"  uploading folder...")
    t0 = time.time()
    api.upload_folder(
        folder_path=OUT, repo_id=REPO, repo_type="model",
        commit_message="Upload DocSmile VLM v1 (LLaVA-style graft on DPO'd Llama-3.1-8B)",
        token=token,
        # ignore runs/ — those are stage training artifacts, not needed in the hub
        ignore_patterns=["runs/**", "training_args.bin", "*.tar", "*.zip"],
    )
    log(f"  upload done in {time.time()-t0:.1f}s")

    log("=== PUSH OK ===")
    log(f"View at: https://huggingface.co/{REPO}")
except Exception as e:
    log(f"!! PUSH FAILED: {e}")
    log(traceback.format_exc())
    sys.exit(1)
