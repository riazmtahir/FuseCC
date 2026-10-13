"""Contrastive (siamese) fine-tune of UniXcoder on GCJ clone pairs.

Split modes:
- file (default): 20% of files held out; training only sees pairs whose
  both files are in the train file set, validation pairs are both-in-heldout.
- problem: PROBLEM-DISJOINT split by contest problem (googlejamN). Training
  sees only pairs fully inside the train problems; validation pairs are fully
  inside --val-problems; --test-problems are excluded entirely (final eval
  only). Their file lists are saved as val_files.txt / test_files.txt.
- Objective: CosineEmbeddingLoss on mean-pooled 512-token embeddings
  (clone -> cos ~ 1, non-clone -> cos < margin).
- Saves the encoder to --out-dir and prints val pair-AUC per epoch.

Run:  nohup python tools/finetune_unixcoder_gcj.py > logs/finetune_unixcoder.log 2>&1 &
      nohup python tools/finetune_unixcoder_gcj.py --split problem \
        --out-dir models/unixcoder_gcj_ft_pd > logs/finetune_unixcoder_pd.log 2>&1 &
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["file", "problem"], default="file")
parser.add_argument("--out-dir", default="models/unixcoder_gcj_ft")
parser.add_argument("--val-problems", default="4,7,11,12")
parser.add_argument("--test-problems", default="2,3,9,10")
args = parser.parse_args()

SEED = 12345
MODEL = "microsoft/unixcoder-base"
OUT_DIR = args.out_dir
PAIRS = "dataset/GCJ/pairs_mapped.csv"
MAX_LEN = 512
BATCH_PAIRS = 12
EPOCH_PAIRS = 60000          # sampled pairs per epoch (balanced)
EPOCHS = 2
LR = 2e-5
MARGIN = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda"

df = pd.read_csv(PAIRS)
files = np.array(sorted(set(df["path_a"]) | set(df["path_b"])))
test_files = set()
if args.split == "file":
    rng = np.random.RandomState(SEED)
    val_files = set(rng.choice(files, size=int(0.2 * len(files)), replace=False))
    a_val = df["path_a"].isin(val_files).values
    b_val = df["path_b"].isin(val_files).values
    train_df = df[~a_val & ~b_val].reset_index(drop=True)
    val_df = df[a_val & b_val].reset_index(drop=True)
else:
    prob = lambda p: re.search(r"googlejam(\d+)\.", p).group(1)  # noqa: E731
    val_probs = set(args.val_problems.split(","))
    test_probs = set(args.test_problems.split(","))
    val_files = set(f for f in files if prob(f) in val_probs)
    test_files = set(f for f in files if prob(f) in test_probs)
    pa, pb = df["path_a"].map(prob), df["path_b"].map(prob)
    train_df = df[~pa.isin(val_probs | test_probs)
                  & ~pb.isin(val_probs | test_probs)].reset_index(drop=True)
    val_df = df[pa.isin(val_probs) & pb.isin(val_probs)].reset_index(drop=True)
    print(f"problem split: train problems exclude {sorted(val_probs | test_probs)}")
print(f"files: {len(files)} (val {len(val_files)}); pairs: train {len(train_df)}, val {len(val_df)}")

code_cache = {}
def read_code(path):
    if path not in code_cache:
        with open(path, "r", errors="ignore") as f:
            code_cache[path] = f.read()
    return code_cache[path]

tokenizer = AutoTokenizer.from_pretrained(MODEL)

class PairDS(Dataset):
    def __init__(self, frame, n=None, seed=0):
        if n is not None:
            pos = frame[frame["label"] == 1]
            neg = frame[frame["label"] == 0]
            k = n // 2
            frame = pd.concat([
                pos.sample(min(k, len(pos)), random_state=seed),
                neg.sample(min(k, len(neg)), random_state=seed),
            ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        r = self.frame.iloc[i]
        return read_code(r["path_a"]), read_code(r["path_b"]), float(r["label"])

def collate(batch):
    codes_a, codes_b, labels = zip(*batch)
    ta = tokenizer(list(codes_a), truncation=True, max_length=MAX_LEN,
                   padding=True, return_tensors="pt")
    tb = tokenizer(list(codes_b), truncation=True, max_length=MAX_LEN,
                   padding=True, return_tensors="pt")
    return ta, tb, torch.tensor(labels)

model = AutoModel.from_pretrained(MODEL).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler()

def encode(tok_batch):
    out = model(input_ids=tok_batch["input_ids"].to(device),
                attention_mask=tok_batch["attention_mask"].to(device)).last_hidden_state
    m = tok_batch["attention_mask"].to(device).unsqueeze(-1).float()
    return (out * m).sum(1) / m.sum(1).clamp(min=1)

@torch.no_grad()
def val_auc():
    model.eval()
    ds = PairDS(val_df, n=min(20000, len(val_df)), seed=SEED)
    dl = DataLoader(ds, batch_size=BATCH_PAIRS * 2, collate_fn=collate, num_workers=2)
    scores, labels = [], []
    for ta, tb, yb in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            ea, eb = encode(ta), encode(tb)
        scores.append(F.cosine_similarity(ea, eb).float().cpu().numpy())
        labels.append(yb.numpy())
    model.train()
    return roc_auc_score(np.concatenate(labels), np.concatenate(scores))

print(f"zero-shot val AUC: {val_auc():.4f}", flush=True)
loss_fn = torch.nn.CosineEmbeddingLoss(margin=MARGIN)

for epoch in range(EPOCHS):
    ds = PairDS(train_df, n=EPOCH_PAIRS, seed=SEED + epoch)
    dl = DataLoader(ds, batch_size=BATCH_PAIRS, collate_fn=collate,
                    num_workers=2, shuffle=True)
    running = 0.0
    for step, (ta, tb, yb) in enumerate(dl):
        target = yb.to(device) * 2 - 1  # 1 clone, -1 non-clone
        with torch.autocast("cuda", dtype=torch.float16):
            ea, eb = encode(ta), encode(tb)
            loss = loss_fn(ea, eb, target)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item()
        if (step + 1) % 200 == 0:
            print(f"epoch {epoch+1} step {step+1}/{len(dl)} loss {running/200:.4f}", flush=True)
            running = 0.0
    print(f"epoch {epoch+1} done, val AUC: {val_auc():.4f}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
with open(os.path.join(OUT_DIR, "val_files.txt"), "w") as f:
    f.write("\n".join(sorted(val_files)))
if test_files:
    with open(os.path.join(OUT_DIR, "test_files.txt"), "w") as f:
        f.write("\n".join(sorted(test_files)))
print("saved to", OUT_DIR)
