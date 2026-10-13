"""Build the GCJ 148-feature matrix from the FINE-TUNED UniXcoder encoder
(models/unixcoder_gcj_ft).

Layout matches results/gcj_148/features.npz:
  cols 0-18  : score_t1..t19 (hand-crafted similarity metrics)
  col  19    : ft-UniXcoder cosine of centered, L2-normalized embeddings
               (replaces the zero-shot score_t20)
  cols 20-83 : |a-b| in 64-dim PCA space
  cols 84-147: a*b   in 64-dim PCA space

Honesty guarantee: the centering mean and PCA are fitted ONLY on the encoder's
training files (everything not listed in the --exclude-files lists), so pairs
among held-out files never influence any fitted component.

Saves <out-dir>/features.npz (+ pca_unixcoder_ft.pkl).
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings", default="dataset/GCJ/embeddings_unixcoder_ft.pkl")
parser.add_argument("--exclude-files", nargs="+",
                    default=["models/unixcoder_gcj_ft/val_files.txt"],
                    help="file lists excluded from the PCA/mean fit")
parser.add_argument("--out-dir", default="results/gcj_148_ft")
args = parser.parse_args()

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

with open(args.embeddings, "rb") as f:
    emb = pickle.load(f)
paths = sorted(emb.keys())
val_files = set()
for lst in args.exclude_files:
    with open(lst) as f:
        val_files |= set(line.strip() for line in f if line.strip())
train_paths = [p for p in paths if p not in val_files]
print(f"{len(paths)} files total, PCA fitted on {len(train_paths)} encoder-train files")

mat = np.vstack([emb[p] for p in paths]).astype(np.float64)
train_mat = np.vstack([emb[p] for p in train_paths]).astype(np.float64)

mean_vec = train_mat.mean(axis=0, keepdims=True)
c = mat - mean_vec
c = c / np.linalg.norm(c, axis=1, keepdims=True)
pca = PCA(n_components=64, random_state=12345)
pca.fit((train_mat - mean_vec) / np.linalg.norm(train_mat - mean_vec, axis=1, keepdims=True))
pc = pca.transform(c)
idx = {p: i for i, p in enumerate(paths)}

df = pd.read_csv("results/gcj_20/results.csv").dropna().reset_index(drop=True)
y = df["label"].astype(np.int8).values
ia = df["path_a"].map(idx).values.astype(int)
ib = df["path_b"].map(idx).values.astype(int)

ft_cos = (c[ia] * c[ib]).sum(axis=1).astype(np.float32)  # rows of c are unit-norm

A, B = pc[ia].astype(np.float32), pc[ib].astype(np.float32)
score_cols = [f"score_t{i}" for i in range(1, 20)]
X = np.hstack([df[score_cols].values.astype(np.float32), ft_cos[:, None],
               np.abs(A - B), A * B])
feature_names = (score_cols + ["uxc_ft_cos"]
                 + [f"uxc_ft_diff_{i}" for i in range(64)]
                 + [f"uxc_ft_prod_{i}" for i in range(64)])

np.savez_compressed(
    os.path.join(OUT_DIR, "features.npz"),
    X=X, y=y, ids=df["id"].values,
    path_a=df["path_a"].values.astype(str), path_b=df["path_b"].values.astype(str),
    feature_names=np.array(feature_names),
)
with open(os.path.join(OUT_DIR, "pca_unixcoder_ft.pkl"), "wb") as f:
    pickle.dump({"mean": mean_vec, "pca": pca}, f)
print(f"Saved X{X.shape} to {OUT_DIR}/features.npz")
