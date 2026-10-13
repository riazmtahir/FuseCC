#!/usr/bin/env python3
"""Build clone-detection pair sets from the ConCord POJ-104 / CodeNet-Java250 releases.

SOURCE FORMAT
`concord_assets/data/{poj104_cc,codenet_cc}/*.jsonl`, one JSON object per line:
    {"label": <problem id>, "index": <sample id>, "code": "<SentencePiece tokens>"}
The released code is ALREADY SentencePiece-tokenised (tokens joined by spaces, word
starts marked with U+2581). We detokenise with the standard inverse — drop the spaces,
turn U+2581 back into a space — which recovers syntactically valid source. Verified:
javalang parses 200/200 detokenised Java250 samples. The one thing NOT recoverable is
the original line layout (everything becomes one line); our metrics are token-based so
this does not affect them, but it is recorded here because it is a real deviation from
the raw corpora and belongs in the paper's data section.

CLONE DEFINITION (identical in spirit to our GCJ set)
Both corpora are *classification* datasets: `label` = the programming problem. Two
different submissions to the SAME problem are semantically equivalent but usually
share no syntax -> a Type-4 / semantic clone pair. Different problems -> non-clone.

PROTOCOL — this is the important part
The upstream splits are PROBLEM-DISJOINT: POJ-104 uses 64/16/24 disjoint problems for
train/valid/test (=104) and Java250 uses 125/62/63 (=250). Verified here: zero label
overlap between any two splits. So a model is always tested on problems it has never
seen, which is the strictest protocol in this project — the analogue of our GCJ
problem-disjoint run (AUC 0.884), NOT of the easier file-disjoint one. Pairs are drawn
WITHIN a split, so the disjointness is preserved end to end.

Non-clone pairs are sampled uniformly over distinct-label pairs; clone pairs uniformly
over same-label pairs. Both classes are balanced 50/50 by construction, so accuracy is
meaningful and the all-clone baseline is F1 0.667.

Usage:
    python tools/prep_concord_pairs.py --dataset java250
    python tools/prep_concord_pairs.py --dataset poj104 --train 1000 --valid 500 --test 500
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_EXT_ROOT = _os.environ.get("FUSECC_EXTERNAL", _os.path.dirname(_REPO_ROOT))
import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict

sys.path.insert(0, _REPO_ROOT)
RANDOM_STATE = 12345  # project-wide constant (see seeds.py)

ROOT = _EXT_ROOT + "/concord_assets/data"
OUT_ROOT = _REPO_ROOT + "/dataset/concord"

SPECS = {
    "java250": {
        "dir": "codenet_cc",
        "files": {"train": "Java250_train.jsonl",
                  "valid": "Java250_valid.jsonl",
                  "test": "Java250_test.jsonl"},
        "language": "java",
    },
    "poj104": {
        "dir": "poj104_cc",
        "files": {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"},
        "language": "cpp",
    },
}


def detokenize(sp_text):
    """SentencePiece-joined text -> source. '▁int ▁a' -> ' int a'."""
    return sp_text.replace(" ", "").replace("▁", " ").strip()


def load_split(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            rows.append((int(d["index"]), int(d["label"]), detokenize(d["code"])))
    return rows


def sample_pairs(rows, n_pairs, rng):
    """Balanced same-label (clone) / different-label (non-clone) pairs.

    Rejection sampling over index pairs: cheap here because every problem has
    hundreds of submissions, so a same-label hit is common and a different-label
    hit is nearly certain.
    """
    by_label = defaultdict(list)
    for i, (_idx, lab, _code) in enumerate(rows):
        by_label[lab].append(i)
    labels = sorted(by_label)
    n_each = n_pairs // 2

    seen = set()
    clones = []
    while len(clones) < n_each:
        lab = labels[rng.randrange(len(labels))]
        pool = by_label[lab]
        if len(pool) < 2:
            continue
        a, b = rng.sample(pool, 2)
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        clones.append(key)

    nonclones = []
    while len(nonclones) < n_each:
        la, lb = rng.sample(labels, 2)
        a = by_label[la][rng.randrange(len(by_label[la]))]
        b = by_label[lb][rng.randrange(len(by_label[lb]))]
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        nonclones.append(key)

    pairs = [(a, b, 1) for a, b in clones] + [(a, b, 0) for a, b in nonclones]
    rng.shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("--train", type=int, default=150_000)
    ap.add_argument("--valid", type=int, default=40_000)
    ap.add_argument("--test", type=int, default=80_000)
    args = ap.parse_args()

    spec = SPECS[args.dataset]
    out_dir = os.path.join(OUT_ROOT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    want = {"train": args.train, "valid": args.valid, "test": args.test}

    # sanity: the splits must really be problem-disjoint
    label_sets = {}
    store, manifest = {}, {}
    for split, fname in spec["files"].items():
        rows = load_split(os.path.join(ROOT, spec["dir"], fname))
        label_sets[split] = {lab for _i, lab, _c in rows}
        store[split] = rows
        manifest[split] = {"n_samples": len(rows), "n_labels": len(label_sets[split])}
        print(f"{split:5s}: {len(rows):,} samples, {len(label_sets[split])} problems")

    overlaps = {f"{a}∩{b}": len(label_sets[a] & label_sets[b])
                for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    print("label overlaps:", overlaps)
    if any(overlaps.values()):
        sys.exit("FATAL: splits are not problem-disjoint; protocol assumption broken")

    all_pairs = {}
    for split, rows in store.items():
        rng = random.Random(RANDOM_STATE + hash(split) % 1000)
        pairs = sample_pairs(rows, want[split], rng)
        all_pairs[split] = pairs
        n_pos = sum(p[2] for p in pairs)
        manifest[split].update({"n_pairs": len(pairs), "n_clone": n_pos,
                                "n_nonclone": len(pairs) - n_pos})
        print(f"{split:5s}: {len(pairs):,} pairs ({n_pos:,} clone / "
              f"{len(pairs) - n_pos:,} non-clone)")

    with open(os.path.join(out_dir, "code_store.pkl"), "wb") as fh:
        pickle.dump({s: [(i, l, c) for i, l, c in r] for s, r in store.items()}, fh,
                    protocol=4)
    with open(os.path.join(out_dir, "pairs.pkl"), "wb") as fh:
        pickle.dump(all_pairs, fh, protocol=4)

    meta = {
        "dataset": args.dataset,
        "language": spec["language"],
        "source": os.path.join(ROOT, spec["dir"]),
        "protocol": ("PROBLEM-DISJOINT (upstream ConCord splits; zero label overlap, "
                     "asserted at build time). Pairs drawn within a split. Clone = same "
                     "problem (semantic/Type-4), non-clone = different problem. Balanced "
                     "50/50, so the all-clone baseline is F1 0.667."),
        "detokenization": ("released code is SentencePiece-tokenised; inverse applied "
                           "(drop spaces, U+2581 -> space). Original line layout is NOT "
                           "recoverable; all metrics here are token-based so this does "
                           "not affect them."),
        "seed": RANDOM_STATE,
        "splits": manifest,
        "label_overlaps": overlaps,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {out_dir}/{{code_store.pkl,pairs.pkl,meta.json}}")


if __name__ == "__main__":
    main()
