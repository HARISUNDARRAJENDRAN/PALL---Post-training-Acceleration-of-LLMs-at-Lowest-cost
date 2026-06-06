#!/bin/bash
# finish_pipeline.sh — runs after training completes.
# 1. waits for training to finish
# 2. runs merge_lora.py
# 3. runs push_to_hf.py
# all logged to ~/finish.log
exec >> /home/ec2-user/finish.log 2>&1
set -e

echo "=== $(date -u +%FT%TZ) finish_pipeline START ==="

# 1. wait for training
while tmux has-session -t train_s2 2>/dev/null; do
    sleep 30
done
echo "$(date -u +%FT%TZ) training tmux exited"

# check tail of log for DONE marker
if ! grep -q "DONE" /home/ec2-user/train_s2.log 2>/dev/null; then
    echo "!! training log missing DONE marker; aborting finish pipeline"
    tail -30 /home/ec2-user/train_s2.log
    exit 1
fi
echo "training DONE marker found"

# 2. merge
echo "$(date -u +%FT%TZ) starting merge"
cd /home/ec2-user/vlm
source /opt/pytorch/bin/activate
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -W ignore /home/ec2-user/vlm/merge_lora.py
echo "merge exit: $?"

# verify size
du -sh /home/ec2-user/vlm/docsmile_vlm_merged
ls -la /home/ec2-user/vlm/docsmile_vlm_merged/

# 3. push
echo "$(date -u +%FT%TZ) starting push"
python -W ignore /home/ec2-user/vlm/push_to_hf.py
echo "push exit: $?"

echo "$(date -u +%FT%TZ) === finish_pipeline DONE ==="
echo "READY_TO_SHUTDOWN" > /home/ec2-user/finish.flag
