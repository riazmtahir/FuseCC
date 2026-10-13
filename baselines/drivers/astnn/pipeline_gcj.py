# Derived from ASTNN (https://github.com/zhangj111/astnn), MIT License, Copyright (c) 2019 zhangj1994; see LICENSE-ASTNN.
"""Re-run: ASTNN preprocessing for our GCJ corpus (googlejam4).

Mirrors pipeline.py but: full-file javalang parse (compilation units, not member
declarations), integer ids assigned to the 1,664 files, stratified 150k-pair
subsample of dataset/GCJ/pairs_mapped.csv (full 544k pairs is intractable for
ASTNN's batch-32 tree encoder), 3:1:1 split, seed 12345.

Run from clone/: python pipeline_gcj.py
"""
import os as _os
_REPO_ROOT = _os.environ["FUSECC_ROOT"]  # path to the FuseCC checkout
import os
import warnings

import javalang
import pandas as pd
from gensim.models.word2vec import Word2Vec

from config import EMBEDDING_SIZE, MIN_COUNT, VOCAB_SIZE
from utils import get_sequence, get_blocks_v1

warnings.filterwarnings("ignore")

SEED = 12345
N_PAIRS = 150_000  # stratified subsample; caption note in the paper
GCJ_DIR = _REPO_ROOT + "/dataset/GCJ/googlejam4"
PAIRS_CSV = _REPO_ROOT + "/dataset/GCJ/pairs_mapped.csv"
ROOT = "data/gcj/"

os.makedirs(ROOT, exist_ok=True)

# --- parse sources -----------------------------------------------------------
ast_path = ROOT + "ast.pkl"
if os.path.exists(ast_path):
    sources = pd.read_pickle(ast_path)
    print(f"loaded {len(sources)} parsed sources")
else:
    rows = []
    files = sorted(f for f in os.listdir(GCJ_DIR)
                   if f.endswith(".java") and not f.startswith("~"))
    for fid, fname in enumerate(files):
        try:
            code = open(os.path.join(GCJ_DIR, fname), errors="ignore").read()
            tree = javalang.parse.parse(code)
            rows.append({"id": fid, "path": f"dataset/GCJ/googlejam4/{fname}",
                         "code": tree})
        except Exception as e:  # noqa: BLE001
            print(f"unparseable, dropped: {fname} ({type(e).__name__})")
    sources = pd.DataFrame(rows)
    sources.to_pickle(ast_path)
    print(f"parsed {len(sources)} files")

path2id = dict(zip(sources["path"], sources["id"]))

# --- pairs: map, filter, subsample, split ------------------------------------
pairs = pd.read_csv(PAIRS_CSV)
pairs["id1"] = pairs["path_a"].map(path2id)
pairs["id2"] = pairs["path_b"].map(path2id)
before = len(pairs)
pairs = pairs.dropna(subset=["id1", "id2"]).astype({"id1": int, "id2": int})
print(f"pairs with both files parseable: {before} -> {len(pairs)}")

sample = pairs.groupby("label", group_keys=False).apply(
    lambda g: g.sample(n=N_PAIRS // 2, random_state=SEED))
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
n = len(sample)
train = sample.iloc[: int(n * 0.6)][["id1", "id2", "label"]]
dev = sample.iloc[int(n * 0.6): int(n * 0.8)][["id1", "id2", "label"]]
test = sample.iloc[int(n * 0.8):][["id1", "id2", "label"]]
for part, df in [("train", train), ("dev", dev), ("test", test)]:
    os.makedirs(ROOT + part, exist_ok=True)
    df.to_pickle(f"{ROOT}{part}/{part}_.pkl")
    print(part, len(df), df.label.mean().round(4), "clone rate")

# --- word2vec on train-file corpus only --------------------------------------
train_ids = pd.concat([train["id1"], train["id2"]]).unique()
trees = sources.set_index("id", drop=False).loc[train_ids]


def to_seq(ast):
    seq = []
    get_sequence(ast, seq)
    return seq


corpus = trees["code"].apply(to_seq)
os.makedirs(ROOT + "train/embedding", exist_ok=True)
w2v = Word2Vec(list(corpus), vector_size=EMBEDDING_SIZE, workers=16, sg=1,
               min_count=MIN_COUNT, max_final_vocab=VOCAB_SIZE)
w2v.save(ROOT + "train/embedding/node_w2v_" + str(EMBEDDING_SIZE))
print("w2v vocab", len(w2v.wv.key_to_index))

# --- block sequences ----------------------------------------------------------
vocab = w2v.wv.key_to_index
max_token = w2v.wv.vectors.shape[0]


def tree_to_index(node):
    result = [vocab.get(node.token, max_token)]
    for child in node.children:
        result.append(tree_to_index(child))
    return result


def trans2seq(r):
    blocks = []
    get_blocks_v1(r, blocks)
    return [tree_to_index(b) for b in blocks]


blocks = sources[["id"]].copy()
blocks["code"] = sources["code"].apply(trans2seq)

for part, df in [("train", train), ("dev", dev), ("test", test)]:
    m = df.merge(blocks, how="left", left_on="id1", right_on="id")
    m = m.merge(blocks, how="left", left_on="id2", right_on="id",
                suffixes=("_x", "_y"))
    m = m.drop(columns=["id_x", "id_y"]).dropna()
    m.columns = ["id1", "id2", "label", "code_x", "code_y"]
    m.to_pickle(f"{ROOT}{part}/blocks.pkl")
    print(part, "blocks", len(m))
print("done")
