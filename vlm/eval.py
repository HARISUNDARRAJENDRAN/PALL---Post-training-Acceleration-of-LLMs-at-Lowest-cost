#!/usr/bin/env python
"""Evaluate the DocSmile VLM on test.jsonl.

Loads base_vlm + (optional) stage-2 LoRA adapter, generates an answer per record, and reports:
  - classification accuracy / macro-F1 (rows with a short categorical answer)
  - exact/substring match rate
  - an OPTIONAL image-shuffle control: re-run with images randomly permuted; a genuinely
    multimodal model should DROP in accuracy. Flat accuracy => modality collapse.

Run: python eval.py --adapter /home/ec2-user/vlm/ckpt/stage2_sft [--limit 500] [--shuffle_control]
"""
import argparse, json, os, random, torch
from collections import Counter, defaultdict
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model(base, adapter):
    proc = AutoProcessor.from_pretrained(base)
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    model = LlavaForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model.to("cuda").eval(), proc

@torch.no_grad()
def generate(model, proc, msgs, imgs, max_new=64):
    tok = proc.tokenizer
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    batch = proc(text=prompt, images=imgs, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**batch, max_new_tokens=max_new, do_sample=False)
    gen = out[0][batch["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/ec2-user/vlm/base_vlm")
    ap.add_argument("--adapter", default="/home/ec2-user/vlm/ckpt/stage2_sft")
    ap.add_argument("--data_root", default="/home/ec2-user/vlm_train")
    ap.add_argument("--test_file", default="test.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle_control", action="store_true")
    a = ap.parse_args()
    random.seed(0)

    model, proc = load_model(a.base, a.adapter if os.path.exists(a.adapter) else None)
    rows = [json.loads(l) for l in open(os.path.join(a.data_root, a.test_file), encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]

    def run(shuffle):
        imgs_pool = [r["images"] for r in rows] if shuffle else None
        if shuffle:
            random.shuffle(imgs_pool)
        correct = total = 0
        per_src = defaultdict(lambda: [0, 0])
        y_true, y_pred = [], []
        for i, r in enumerate(rows):
            paths = imgs_pool[i] if shuffle else r["images"]
            imgs = [Image.open(os.path.join(a.data_root, p.replace("/", os.sep))).convert("RGB") for p in paths]
            pred = generate(model, proc, r["messages"], imgs)
            gold = r["messages"][1]["content"].strip()
            ok = gold.lower() in pred.lower() or pred.lower() in gold.lower()
            correct += ok; total += 1
            per_src[r["source"]][0] += ok; per_src[r["source"]][1] += 1
            y_true.append(gold.lower()); y_pred.append(pred.lower())
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)} acc={correct/total:.3f}")
        return correct / max(total, 1), per_src, y_true, y_pred

    print(f"[eval] {len(rows)} test rows | adapter={a.adapter}")
    acc, per_src, yt, yp = run(shuffle=False)
    print(f"\n=== MATCH ACCURACY: {acc:.4f} ===")
    print("per-source:")
    for s, (c, n) in sorted(per_src.items(), key=lambda x: -x[1][1]):
        print(f"  {c/n:.3f}  ({n:5d})  {s}")
    # macro-F1 over the (short) categorical answers
    labels = set(yt)
    if len(labels) < 60:
        f1s = []
        for lab in labels:
            tp = sum(1 for t, p in zip(yt, yp) if t == lab and lab in p)
            fp = sum(1 for t, p in zip(yt, yp) if t != lab and lab in p)
            fn = sum(1 for t, p in zip(yt, yp) if t == lab and lab not in p)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
        print(f"macro-F1 (~{len(labels)} classes): {sum(f1s)/len(f1s):.4f}")

    if a.shuffle_control:
        print("\n[control] re-running with SHUFFLED images (accuracy should DROP if model uses vision)")
        sacc, _, _, _ = run(shuffle=True)
        print(f"=== shuffled-image accuracy: {sacc:.4f}  (vs {acc:.4f}; drop = {acc - sacc:+.4f}) ===")
        print("    small/zero drop => MODALITY COLLAPSE (model ignoring the image)")

if __name__ == "__main__":
    main()
