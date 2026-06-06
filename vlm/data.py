"""Dataset + collator for the DocSmile VLM.

Each JSONL row: {messages:[user(<image>...),assistant], images:[relpaths], ...}.
We render with the Llama-3.1 chat template, let the LlavaProcessor expand each <image>
into image_seq_length patch tokens + produce pixel_values, and mask the prompt so loss is
only on the assistant turn. Single- and multi-image rows are handled uniformly (the number
of <image> tokens always equals len(images)).
"""
import json, os, torch
from torch.utils.data import Dataset
from PIL import Image

class VLMDataset(Dataset):
    def __init__(self, jsonl, root, processor, max_seq_len, single_image_only=False,
                 img_tokens_per_image=729):
        self.root = root
        self.proc = processor
        self.max = max_seq_len
        rows = []
        dropped = 0
        for line in open(jsonl, encoding="utf-8"):
            r = json.loads(line)
            n = len(r["images"])
            if single_image_only and n != 1:
                continue
            # cheap length guard: image tokens + rough text estimate
            est = n * img_tokens_per_image + len(r["messages"][0]["content"]) // 3 \
                  + len(r["messages"][1]["content"]) // 3 + 16
            if est > max_seq_len:
                dropped += 1
                continue
            rows.append(r)
        self.rows = rows
        self._dropped = dropped

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = r["messages"]
        imgs = [Image.open(os.path.join(self.root, p.replace("/", os.sep))).convert("RGB")
                for p in r["images"]]
        tok = self.proc.tokenizer
        prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        full = self.proc(text=full_text, images=imgs, return_tensors="pt", add_special_tokens=False)
        prompt = self.proc(text=prompt_text, images=imgs, return_tensors="pt", add_special_tokens=False)
        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        plen = prompt["input_ids"].shape[1]
        labels[:plen] = -100                      # supervise assistant turn only
        return {"input_ids": input_ids,
                "labels": labels,
                "attention_mask": full["attention_mask"][0],
                "pixel_values": full["pixel_values"]}   # (n_images, 3, H, W)


class Collator:
    """Right-pad text; concat pixel_values across the batch (LLaVA matches images to
    <image> positions in order)."""
    def __init__(self, pad_id):
        self.pad = pad_id

    def __call__(self, feats):
        maxlen = max(f["input_ids"].size(0) for f in feats)
        ids, labels, attn, pix = [], [], [], []
        for f in feats:
            n = f["input_ids"].size(0); p = maxlen - n
            ids.append(torch.cat([f["input_ids"], torch.full((p,), self.pad, dtype=torch.long)]))
            labels.append(torch.cat([f["labels"], torch.full((p,), -100, dtype=torch.long)]))
            attn.append(torch.cat([f["attention_mask"], torch.zeros(p, dtype=torch.long)]))
            pix.append(f["pixel_values"])
        return {"input_ids": torch.stack(ids),
                "labels": torch.stack(labels),
                "attention_mask": torch.stack(attn),
                "pixel_values": torch.cat(pix, 0)}
