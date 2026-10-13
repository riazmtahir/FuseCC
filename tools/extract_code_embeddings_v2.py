"""Extract whole-file code embeddings on GPU for the GCJ corpus.

Improvements over extract_codebert_embeddings.py:
- GPU inference, batched
- No 256-token truncation: files are split into 512-token windows and the
  masked-mean-pooled window embeddings are averaged (weighted by real tokens)
- Supports multiple encoder models for comparison

Writes dataset/GCJ/embeddings_<tag>.pkl : dict file_path -> float32[hidden]
"""
import os
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODELS = {
    "codebert": "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
    "unixcoder": "microsoft/unixcoder-base",
    "unixcoder_ft": "models/unixcoder_gcj_ft",
    "unixcoder_ft_pd": "models/unixcoder_gcj_ft_pd",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=sorted(MODELS), required=True)
parser.add_argument("--pairs", default="dataset/GCJ/pairs_mapped.csv")
parser.add_argument("--out-dir", default="dataset/GCJ")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-length", type=int, default=512)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODELS[args.model]} on {device} ...")
tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
model = AutoModel.from_pretrained(MODELS[args.model]).to(device).eval()

df = pd.read_csv(args.pairs)
all_files = sorted(set(df["path_a"]).union(df["path_b"]))
print(f"{len(all_files)} unique files")


@torch.no_grad()
def embed_file(code: str) -> np.ndarray:
    enc = tokenizer(
        code,
        truncation=True,
        max_length=args.max_length,
        stride=0,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    chunk_vecs, chunk_weights = [], []
    for i in range(0, len(ids), args.batch_size):
        b_ids = ids[i : i + args.batch_size].to(device)
        b_mask = mask[i : i + args.batch_size].to(device)
        out = model(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
        m = b_mask.unsqueeze(-1).float()
        summed = (out * m).sum(dim=1)          # [B, H]
        counts = m.sum(dim=1)                  # [B, 1]
        chunk_vecs.append((summed / counts.clamp(min=1)).cpu())
        chunk_weights.append(counts.squeeze(-1).cpu())
    vecs = torch.cat(chunk_vecs)               # [n_chunks, H]
    w = torch.cat(chunk_weights)               # [n_chunks]
    return ((vecs * w.unsqueeze(-1)).sum(dim=0) / w.sum()).numpy().astype(np.float32)


embeddings = {}
failed = []
for fpath in tqdm(all_files, desc=f"embedding ({args.model})"):
    try:
        with open(fpath, "r", errors="ignore") as f:
            code = f.read()
        embeddings[fpath] = embed_file(code)
    except Exception as e:  # noqa: BLE001
        failed.append((fpath, repr(e)))

out_path = os.path.join(args.out_dir, f"embeddings_{args.model}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(embeddings, f)
print(f"Saved {len(embeddings)} embeddings to {out_path}; {len(failed)} failures")
for fp, err in failed[:10]:
    print("FAILED:", fp, err)
