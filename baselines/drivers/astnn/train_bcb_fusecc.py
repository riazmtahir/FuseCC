# Derived from ASTNN (https://github.com/zhangj111/astnn), MIT License, Copyright (c) 2019 zhangj1994; see LICENSE-ASTNN.
"""Re-run: ASTNN on our canonical BCB pairs, binary protocol.

Differences vs the stock train.py:
- our pipeline produced binary labels (0/1), so the per-clone-type training
  loop (which relies on labels 1..5) is replaced by a single binary pass;
- test predictions are saved with (id1, id2, prob) so per-type recall can be
  computed against dataset/BCB/pairs_with_types.csv;
- deterministic seed 12345; progress logging with batches/sec.

Run from clone/: python train_bcb_fusecc.py
"""
import os as _os
_REPO_ROOT = _os.environ["FUSECC_ROOT"]  # path to the FuseCC checkout
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import warnings
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import precision_recall_fscore_support
from model import BatchProgramCC
from config import HIDDEN_DIM, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU, EPOCHS

warnings.filterwarnings("ignore")

SEED = 12345
OUT_DIR = _REPO_ROOT + "/results/baselines_rerun/astnn_bcb"
TYPES_CSV = _REPO_ROOT + "/dataset/BCB/pairs_with_types.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item["code_x"])
        x2.append(item["code_y"])
        labels.append([item["label"]])
    return x1, x2, torch.FloatTensor(labels)


def predict(model, data):
    probs = []
    i = 0
    model.eval()
    with torch.no_grad():
        while i < len(data):
            x1, x2, labels = get_batch(data, i, BATCH_SIZE)
            i += BATCH_SIZE
            model.batch_size = len(labels)
            model.hidden = model.init_hidden()
            out = model(x1, x2)
            probs.extend(out.data.cpu().numpy().ravel().tolist())
            if (i // BATCH_SIZE) % 500 == 0:
                print(f"  eval {i}/{len(data)}", flush=True)
    model.train()
    return np.array(probs)


root = "data/java/"
train_data = pd.read_pickle(root + "train/blocks.pkl").sample(frac=1, random_state=SEED)
dev_data = pd.read_pickle(root + "dev/blocks.pkl")
test_data = pd.read_pickle(root + "test/blocks.pkl")
# 20k stratified dev subsample: per-epoch convergence signal at ~1/5 the eval cost
dev_small = dev_data.groupby("label", group_keys=False).apply(
    lambda g: g.sample(n=10000, random_state=SEED))
print(f"train {len(train_data)} dev {len(dev_data)} (probe {len(dev_small)}) "
      f"test {len(test_data)}", flush=True)

EVAL_ONLY = sys.argv[1] if len(sys.argv) > 1 else None  # path to a checkpoint .pt

word2vec = Word2Vec.load(root + "train/embedding/node_w2v_128").wv
MAX_TOKENS = word2vec.vectors.shape[0]
EMBEDDING_DIM = word2vec.vectors.shape[1]
embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors

model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS,
                       BATCH_SIZE, USE_GPU, embeddings)
if USE_GPU:
    model.cuda()

optimizer = torch.optim.Adamax(model.parameters())
loss_function = torch.nn.BCELoss()

print("Start training...", flush=True)
if EVAL_ONLY:
    model.load_state_dict(torch.load(EVAL_ONLY))
    print(f"eval-only mode: loaded {EVAL_ONLY}", flush=True)
for epoch in range(0 if EVAL_ONLY else EPOCHS):
    start = time.time()
    i = 0
    total_loss, total = 0.0, 0
    while i < len(train_data):
        x1, x2, labels = get_batch(train_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        if USE_GPU:
            labels = labels.cuda()
        model.zero_grad()
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        output = model(x1, x2)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
        if (i // BATCH_SIZE) % 500 == 0:
            rate = i / (time.time() - start)
            print(f"epoch {epoch+1} batch {i//BATCH_SIZE}/{len(train_data)//BATCH_SIZE} "
                  f"loss {total_loss/total:.4f} {rate:.0f} pairs/s", flush=True)
    print(f"epoch {epoch+1} done in {(time.time()-start)/60:.1f} min "
          f"loss {total_loss/total:.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_epoch{epoch+1}.pt"))

    # dev probe after every epoch so we can early-stop by hand if converged
    dev_probs = predict(model, dev_small)
    dp, dr, df1, _ = precision_recall_fscore_support(
        dev_small["label"].values, (dev_probs > 0.5).astype(int), average="binary")
    print(f"epoch {epoch+1} dev-probe P {dp:.4f} R {dr:.4f} F1 {df1:.4f}", flush=True)

print("Final test...", flush=True)
test_probs = predict(model, test_data)
pred = (test_probs > 0.5).astype(int)
p, r, f1, _ = precision_recall_fscore_support(test_data["label"].values, pred,
                                              average="binary")
print(f"TEST P {p:.4f} R {r:.4f} F1 {f1:.4f}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
out = test_data[["id1", "id2", "label"]].copy()
out["prob"] = test_probs
out.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

# per-type recall on the test split
types = pd.read_csv(TYPES_CSV)
types["lo"] = types[["id_a", "id_b"]].min(axis=1)
types["hi"] = types[["id_a", "id_b"]].max(axis=1)
out["lo"] = out[["id1", "id2"]].min(axis=1)
out["hi"] = out[["id1", "id2"]].max(axis=1)
merged = out.merge(types[["lo", "hi", "clone_type"]], on=["lo", "hi"], how="left")
per_type = {}
for t in ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]:
    sub = merged[(merged.label == 1) & (merged.clone_type == t)]
    if len(sub):
        per_type[t] = {"recall": float((sub.prob > 0.5).mean()), "n": int(len(sub))}

summary = {
    "system": "ASTNN (Zhang et al., ICSE 2019) re-run",
    "dataset": "BCB canonical pairs, binary 3:1:1 split (pipeline_bcb)",
    "protocol": "supervised; trained on our canonical pairs; threshold 0.5",
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED,
    "test": {"precision": float(p), "recall": float(r), "f1": float(f1),
             "n": int(len(test_data))},
    "per_type_recall_test": per_type,
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2), flush=True)

sys.path.insert(0, _REPO_ROOT + "/tools")
try:
    from run_meta import save_run_meta
    save_run_meta(OUT_DIR)
except Exception as e:
    print(f"run_meta skipped: {e!r}")
