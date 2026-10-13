"""Prepare MAGNET (Multigraph_match) data_source dirs from the FA-AST benchmark
files — the same corpus/split lineage MAGNET's paper uses (its BCB forbidden-file
ids come from this set), and the same splits as our FA-AST+GMN re-run.

GCJ: javadata/{trainall,valid,test}.txt over googlejam4_src (1:1-ish labels 1/-1)
BCB: BCB/{traindata,devdata,testdata}.txt over bigclonebenchdata/<id>.txt

MAGNET expects data/data_source/<name>/dataset_files/<id>.java and a 7-column
clone_labels.txt: id1,id2,clone(0/1),split(0=train,1=test,2=val),
dataset(0=BCB,1=GCJ),clone_type,similarity.

Because full-scale training (726k-1.6M pairs x 20 epochs) exceeds our GPU budget,
train pairs are stratified-subsampled (--train-cap, seed 12345); official val and
test splits are kept (BCB val/test capped too - they are 416k each).
BCB clone_type joined from our E1 mapping (dataset/BCB/pairs_with_types.csv).

Usage: prep_magnet_data.py --dataset gcj|bcb [--train-cap N] [--eval-cap N]
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_EXT_ROOT = _os.environ.get("FUSECC_EXTERNAL", _os.path.dirname(_REPO_ROOT))
import argparse
import os
import random
import shutil

FAAST = _EXT_ROOT + "/baselines/graphmatch_clone"
MAGNET = _EXT_ROOT + "/baselines/Multigraph_match/data/data_source"
REPO = _REPO_ROOT
SEED = 12345


def read_pairs(path, sep=None):
    rows = []
    with open(path) as f:
        for ln in f:
            parts = ln.strip().replace("\\", "/").split(sep)
            if len(parts) != 3:
                continue
            a, b, lab = parts
            rows.append((os.path.basename(a), os.path.basename(b),
                         1 if lab.strip() == "1" else 0))
    return rows


def stratified_cap(rows, cap, rng):
    if cap <= 0 or len(rows) <= cap:
        return rows
    pos = [r for r in rows if r[2] == 1]
    neg = [r for r in rows if r[2] == 0]
    f = cap / len(rows)
    pos = rng.sample(pos, int(len(pos) * f))
    neg = rng.sample(neg, cap - len(pos))
    out = pos + neg
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gcj", "bcb"], required=True)
    ap.add_argument("--train-cap", type=int, default=100_000)
    ap.add_argument("--eval-cap", type=int, default=0,
                    help="cap val/test (BCB needs it; GCJ official splits are small)")
    args = ap.parse_args()
    rng = random.Random(SEED)

    if args.dataset == "gcj":
        out = os.path.join(MAGNET, "dataset_java")
        src_dir = os.path.join(FAAST, "googlejam4_src")
        splits = {0: read_pairs(f"{FAAST}/javadata/trainall.txt"),
                  2: read_pairs(f"{FAAST}/javadata/valid.txt"),
                  1: read_pairs(f"{FAAST}/javadata/test.txt")}
        ds_label, ext = 1, ".java"
        def find_src(base):
            for prob in os.listdir(src_dir):
                p = os.path.join(src_dir, prob, base)
                if os.path.exists(p):
                    return p
            return None
    else:
        out = os.path.join(MAGNET, "dataset_bigclonebench")
        src_dir = os.path.join(FAAST, "BCB", "bigclonebenchdata")
        splits = {0: read_pairs(f"{FAAST}/BCB/traindata.txt", "\t"),
                  2: read_pairs(f"{FAAST}/BCB/devdata.txt", "\t"),
                  1: read_pairs(f"{FAAST}/BCB/testdata.txt", "\t")}
        ds_label, ext = 0, ".txt"
        def find_src(base):
            p = os.path.join(src_dir, base)
            return p if os.path.exists(p) else None

    splits[0] = stratified_cap(splits[0], args.train_cap, rng)
    if args.eval_cap:
        splits[2] = stratified_cap(splits[2], args.eval_cap // 5, rng)
        splits[1] = stratified_cap(splits[1], args.eval_cap, rng)

    # clone-type map for BCB per-type reporting
    types = {}
    if args.dataset == "bcb":
        import pandas as pd
        t = pd.read_csv(f"{REPO}/dataset/BCB/pairs_with_types.csv")
        for r in t.itertuples():
            types[frozenset((str(r.id_a), str(r.id_b)))] = r.clone_type

    files_dir = os.path.join(out, "dataset_files")
    os.makedirs(files_dir, exist_ok=True)
    needed, lines, missing = set(), [], 0
    for split, rows in splits.items():
        for a, b, lab in rows:
            ia, ib = a[: -len(ext)], b[: -len(ext)]
            ct = types.get(frozenset((ia, ib)), "NA")
            ct = "NA" if (isinstance(ct, float) or ct is None) else str(ct).replace(",", "_")
            lines.append(f"{ia},{ib},{lab},{split},{ds_label},{ct},0.0")
            needed.update([a, b])
    import re
    for base in sorted(needed):
        dst = os.path.join(files_dir, base[: -len(ext)] + ".java")
        src = find_src(base)
        if src is None:
            missing += 1
            continue
        # MAGNET parses with parse_member_declaration(), which cannot handle
        # package/import headers -> strip them (their dataset_java was
        # preprocessed the same way; BCB snippets have no headers anyway)
        with open(src, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = "\n".join(l for l in text.splitlines()
                         if not re.match(r"\s*(package|import)\s", l))
        with open(dst, "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(out, "clone_labels.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    n = {s: (len(r), sum(x[2] for x in r)) for s, r in splits.items()}
    print(f"{args.dataset}: files={len(needed)} missing={missing} "
          f"train(n,pos)={n[0]} val={n[2]} test={n[1]} -> {out}")


if __name__ == "__main__":
    main()
