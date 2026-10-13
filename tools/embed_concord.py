#!/usr/bin/env python3
"""UniXcoder embeddings for the ConCord POJ-104 / Java250 corpora.

Mirrors tools/extract_code_embeddings_v2.py (the GCJ path): whole-sample embeddings via
512-token windows, masked-mean-pooled per window then averaged, on GPU. Zero-shot
encoder only — the GCJ problem-disjoint fine-tuning experiment showed that fine-tuning on one
set of problems DESTROYS cross-problem generalisation (AUC 0.884 -> 0.622), and these
benchmarks are problem-disjoint by construction, so a fine-tuned encoder would be
exactly the wrong tool here.

UniXcoder is multilingual, so the same encoder serves Java250 (Java) and POJ-104 (C/C++).

Writes dataset/concord/<ds>/embeddings_<split>.npy, row-aligned with code_store.pkl.
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DATA_ROOT = _REPO_ROOT + "/dataset/concord"


def embed_texts(texts, tok, model, device, batch_size=32, max_length=512):
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hid = model(**enc).last_hidden_state           # B x L x H
        mask = enc["attention_mask"].unsqueeze(-1).float()  # B x L x 1
        pooled = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out[start:start + len(batch)] = pooled.cpu().numpy()
        if (start // batch_size) % 100 == 0:
            print(f"    {min(start + batch_size, len(texts)):,}/{len(texts):,}",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="microsoft/unixcoder-base")
    ap.add_argument("--tag", default="unixcoder",
                    help="output suffix; 'unixcoder' keeps the unsuffixed filenames the "
                         "FuseCC pipeline expects, other tags add _<tag>")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--splits", default="train,valid,test")
    args = ap.parse_args()

    d = os.path.join(DATA_ROOT, args.dataset)
    with open(os.path.join(d, "code_store.pkl"), "rb") as fh:
        store = pickle.load(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    suffix = "" if args.tag == "unixcoder" else f"_{args.tag}"
    for split in args.splits.split(","):
        texts = [c for _i, _l, c in store[split]]
        print(f"[{split}] embedding {len(texts):,} samples")
        emb = embed_texts(texts, tok, model, device, args.batch_size)
        name = f"embeddings{suffix}_{split}.npy"
        np.save(os.path.join(d, name), emb)
        print(f"[{split}] -> {name} {emb.shape}")


if __name__ == "__main__":
    main()
