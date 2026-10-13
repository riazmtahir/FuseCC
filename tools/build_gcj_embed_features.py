"""Build the GCJ 148-feature matrix: 20 similarity scores + 128 UniXcoder
PCA interaction features (|a-b| and a*b in 64-dim PCA space of centered,
L2-normalized whole-file embeddings).

Saves results/gcj_148/features.npz (X float32, y int8, ids) and the fitted
PCA/mean for reuse on other datasets.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

OUT_DIR = "results/gcj_148"
os.makedirs(OUT_DIR, exist_ok=True)

with open("dataset/GCJ/embeddings_unixcoder.pkl", "rb") as f:
    emb = pickle.load(f)
paths = sorted(emb.keys())
mat = np.vstack([emb[p] for p in paths]).astype(np.float64)
mean_vec = mat.mean(axis=0, keepdims=True)
c = mat - mean_vec
c = c / np.linalg.norm(c, axis=1, keepdims=True)
pca = PCA(n_components=64, random_state=12345)
pc = pca.fit_transform(c)
idx = {p: i for i, p in enumerate(paths)}

df = pd.read_csv("results/gcj_20/results.csv").dropna().reset_index(drop=True)
y = df["label"].astype(np.int8).values
ia = df["path_a"].map(idx).values.astype(int)
ib = df["path_b"].map(idx).values.astype(int)

A, B = pc[ia].astype(np.float32), pc[ib].astype(np.float32)
score_cols = [f"score_t{i}" for i in range(1, 21)]
X = np.hstack([df[score_cols].values.astype(np.float32), np.abs(A - B), A * B])
feature_names = score_cols + [f"uxc_diff_{i}" for i in range(64)] + [f"uxc_prod_{i}" for i in range(64)]

np.savez_compressed(
    os.path.join(OUT_DIR, "features.npz"),
    X=X, y=y, ids=df["id"].values,
    path_a=df["path_a"].values.astype(str), path_b=df["path_b"].values.astype(str),
    feature_names=np.array(feature_names),
)
with open(os.path.join(OUT_DIR, "pca_unixcoder.pkl"), "wb") as f:
    pickle.dump({"mean": mean_vec, "pca": pca}, f)
print(f"Saved X{X.shape} to {OUT_DIR}/features.npz")
