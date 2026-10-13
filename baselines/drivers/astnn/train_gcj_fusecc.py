# Derived from ASTNN (https://github.com/zhangj111/astnn), MIT License, Copyright (c) 2019 zhangj1994; see LICENSE-ASTNN.
"""Re-run: ASTNN on our GCJ pairs (stratified 150k subsample, random split).

Binary protocol matching pipeline_gcj.py output; saves test predictions and
summary.json to results/baselines_rerun/astnn_gcj/. Seed 12345.

Run from clone/: python train_gcj_fusecc.py [checkpoint.pt for eval-only]
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from model import BatchProgramCC
from config import HIDDEN_DIM, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU, EPOCHS

warnings.filterwarnings("ignore")

SEED = 12345
OUT_DIR = _REPO_ROOT + "/results/baselines_rerun/astnn_gcj"
ROOT = "data/gcj/"

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
    model.train()
    return np.array(probs)


train_data = pd.read_pickle(ROOT + "train/blocks.pkl").sample(frac=1, random_state=SEED)
dev_data = pd.read_pickle(ROOT + "dev/blocks.pkl")
test_data = pd.read_pickle(ROOT + "test/blocks.pkl")
print(f"train {len(train_data)} dev {len(dev_data)} test {len(test_data)}", flush=True)

word2vec = Word2Vec.load(ROOT + "train/embedding/node_w2v_128").wv
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

EVAL_ONLY = sys.argv[1] if len(sys.argv) > 1 else None
if EVAL_ONLY:
    model.load_state_dict(torch.load(EVAL_ONLY))
    print(f"eval-only mode: loaded {EVAL_ONLY}", flush=True)

print("Start training...", flush=True)
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

    dev_probs = predict(model, dev_data)
    dp, dr, df1, _ = precision_recall_fscore_support(
        dev_data["label"].values, (dev_probs > 0.5).astype(int), average="binary")
    print(f"epoch {epoch+1} dev P {dp:.4f} R {dr:.4f} F1 {df1:.4f}", flush=True)

print("Final test...", flush=True)
test_probs = predict(model, test_data)
pred = (test_probs > 0.5).astype(int)
p, r, f1, _ = precision_recall_fscore_support(test_data["label"].values, pred,
                                              average="binary")
auc = roc_auc_score(test_data["label"].values, test_probs)
print(f"TEST P {p:.4f} R {r:.4f} F1 {f1:.4f} AUC {auc:.4f}", flush=True)

out = test_data[["id1", "id2", "label"]].copy()
out["prob"] = test_probs
out.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

summary = {
    "system": "ASTNN (Zhang et al., ICSE 2019) re-run",
    "dataset": "GCJ googlejam4, stratified 150k-pair subsample of pairs_mapped.csv, "
               "3:1:1 random split (protocol comparable to our random-split rows)",
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED,
    "test": {"precision": float(p), "recall": float(r), "f1": float(f1),
             "auc": float(auc), "n": int(len(test_data))},
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
