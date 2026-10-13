#!/usr/bin/env python
"""FA-AST+GMN BCB re-run: generate BCB/{traindata,devdata,testdata}.txt from
MAGNET's prepped 100k/20k/100k pair list so BOTH re-runs use IDENTICAL pairs.

MAGNET's clone_labels.txt was itself built (tools/prep_magnet_data.py) as a
stratified cap of FA-AST's stock BCB splits, so this is a lossless round-trip
back into FA-AST's input format:
    ./bigclonebenchdata/<id1>.txt\t./bigclonebenchdata/<id2>.txt\t{1|-1}
Split codes in clone_labels.txt: 0=train, 1=test, 2=val. Labels: 1=clone->1,
0=non-clone->-1. Stock FA-AST files are backed up as *_upstream_stock.txt
(zero code changes in their repo: run_bcb.py --data_setting 0 reads these names).
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_EXT_ROOT = _os.environ.get("FUSECC_EXTERNAL", _os.path.dirname(_REPO_ROOT))
import os
import shutil
from collections import Counter

MAGNET_LABELS = _EXT_ROOT + "/baselines/Multigraph_match/data/data_source/dataset_bigclonebench/clone_labels.txt"
FAAST_BCB = _EXT_ROOT + "/baselines/graphmatch_clone/BCB"

SPLIT_FILE = {0: "traindata.txt", 2: "devdata.txt", 1: "testdata.txt"}


def main():
    available = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(FAAST_BCB, "bigclonebenchdata"))}
    rows = {0: [], 1: [], 2: []}
    missing = 0
    with open(MAGNET_LABELS) as f:
        for ln in f:
            parts = ln.strip().split(",")
            if len(parts) < 4:
                continue
            id1, id2, clone, split = parts[0], parts[1], int(parts[2]), int(parts[3])
            if id1 not in available or id2 not in available:
                missing += 1
                continue
            lab = 1 if clone == 1 else -1
            rows[split].append(f"./bigclonebenchdata/{id1}.txt\t./bigclonebenchdata/{id2}.txt\t{lab}\n")

    assert missing == 0, f"{missing} pairs reference ids absent from bigclonebenchdata"

    for split, fname in SPLIT_FILE.items():
        path = os.path.join(FAAST_BCB, fname)
        backup = path.replace(".txt", "_upstream_stock.txt")
        if os.path.exists(path) and not os.path.exists(backup):
            shutil.move(path, backup)
        with open(path, "w") as f:
            f.writelines(rows[split])
        pos = sum(1 for r in rows[split] if r.rstrip().endswith("\t1"))
        print(f"{fname}: {len(rows[split])} pairs ({pos} clone / {len(rows[split]) - pos} non-clone)")

    print("split codes: 0=train 1=test 2=val | label counts sanity:",
          {s: Counter(r.rstrip().split("\t")[-1] for r in rows[s]) for s in rows})


if __name__ == "__main__":
    main()
