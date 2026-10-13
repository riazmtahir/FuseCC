#!/usr/bin/env python3
"""Score FA-AST+GMN BCB re-run dumps -> results/baselines_rerun/faast_bcb/summary.json.

Works on however many epochs have finished, so a partial run stays publishable
(same salvage pattern as MAGNET BCB / ASTNN). Epoch is selected on dev F1,
never on test.

Scoring uses the tool's OWN native decision rule, unchanged, from run_bcb.py::test:
    output = cosine_similarity(emb_a, emb_b)
    prediction = torch.sign(output).item()
    clone iff prediction > args.threshold   (their default threshold = 0)
so a pair is predicted clone iff cosine > 0. Labels in their pair files are +1
(clone) / -1 (non-clone).

Dump alignment: gmnbcbresult/<mode>_dev_epoch_N and <mode>_epoch_N hold one score
per line in the exact order of BCB/devdata.txt and BCB/testdata.txt. Verified:
20,000 and 100,000 lines = the full prepped splits, i.e. FA-AST drops
no dev/test pairs (unlike MAGNET, whose native skips cost it a few hundred) --
the script asserts this rather than trusting it.

Usage:
    python tools/archive_faast_bcb.py [--graphmode astandnext] [--min-epochs 1]
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_EXT_ROOT = _os.environ.get("FUSECC_EXTERNAL", _os.path.dirname(_REPO_ROOT))
import argparse
import json
import os
import sys
from collections import defaultdict

REPO = _EXT_ROOT + "/baselines/graphmatch_clone"
OUT_DIR = _REPO_ROOT + "/results/baselines_rerun/faast_bcb"
TYPES_CSV = _REPO_ROOT + "/dataset/BCB/pairs_with_types.csv"

sys.path.insert(0, _REPO_ROOT)


def read_pairs(path):
    """-> [(id_a, id_b, label)] with label in {+1,-1}, file order preserved."""
    pairs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a, b, lab = line.split("\t")
            # './bigclonebenchdata/3046105.txt' -> '3046105'
            ida = os.path.basename(a).rsplit(".", 1)[0]
            idb = os.path.basename(b).rsplit(".", 1)[0]
            pairs.append((ida, idb, int(lab)))
    return pairs


def read_scores(path):
    with open(path) as fh:
        return [float(x) for x in fh if x.strip()]


def prf(pairs, scores, threshold=0.0):
    """Their native rule: clone iff sign(cos) > threshold, i.e. cos > 0."""
    tp = tn = fp = fn = 0
    for (_, _, label), score in zip(pairs, scores):
        pred_clone = (1.0 if score > 0 else (-1.0 if score < 0 else 0.0)) > threshold
        if pred_clone and label == 1:
            tp += 1
        elif not pred_clone and label == -1:
            tn += 1
        elif pred_clone and label == -1:
            fp += 1
        else:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(pairs)}


def load_types():
    """(id_a,id_b) unordered -> clone_type, for per-type recall."""
    types = {}
    with open(TYPES_CSV) as fh:
        header = fh.readline().rstrip("\n").split(",")
        ia, ib, ic = header.index("id_a"), header.index("id_b"), header.index("clone_type")
        for line in fh:
            f = line.rstrip("\n").split(",")
            types[frozenset((f[ia], f[ib]))] = f[ic]
    return types


def per_type_recall(pairs, scores, types, threshold=0.0):
    hit = defaultdict(int)
    tot = defaultdict(int)
    unmapped = 0
    for (a, b, label), score in zip(pairs, scores):
        if label != 1:
            continue
        ct = types.get(frozenset((a, b)))
        if ct is None:
            unmapped += 1
            continue
        tot[ct] += 1
        if (1.0 if score > 0 else (-1.0 if score < 0 else 0.0)) > threshold:
            hit[ct] += 1
    return ({ct: hit[ct] / tot[ct] for ct in sorted(tot)},
            {ct: tot[ct] for ct in sorted(tot)},
            unmapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphmode", default="astandnext")
    ap.add_argument("--min-epochs", type=int, default=1)
    args = ap.parse_args()

    dev_pairs = read_pairs(os.path.join(REPO, "BCB/devdata.txt"))
    test_pairs = read_pairs(os.path.join(REPO, "BCB/testdata.txt"))
    resdir = os.path.join(REPO, "gmnbcbresult")

    epochs = []
    n = 1
    while True:
        devf = os.path.join(resdir, f"{args.graphmode}_dev_epoch_{n}")
        tstf = os.path.join(resdir, f"{args.graphmode}_epoch_{n}")
        if not (os.path.exists(devf) and os.path.exists(tstf)):
            break
        dev_scores, test_scores = read_scores(devf), read_scores(tstf)
        # A short file means the epoch was still being written -- stop here.
        if len(dev_scores) != len(dev_pairs) or len(test_scores) != len(test_pairs):
            print(f"epoch {n}: incomplete dump "
                  f"(dev {len(dev_scores)}/{len(dev_pairs)}, "
                  f"test {len(test_scores)}/{len(test_pairs)}) -- stopping")
            break
        epochs.append({"epoch": n,
                       "dev": prf(dev_pairs, dev_scores),
                       "test": prf(test_pairs, test_scores),
                       "_test_scores": test_scores})
        print(f"epoch {n}: dev F1 {epochs[-1]['dev']['f1']:.4f}  "
              f"test F1 {epochs[-1]['test']['f1']:.4f}")
        n += 1

    if len(epochs) < args.min_epochs:
        sys.exit(f"only {len(epochs)} complete epoch(s) found; need {args.min_epochs}")

    best = max(epochs, key=lambda e: e["dev"]["f1"])          # dev-selected
    types = load_types()
    recall, counts, unmapped = per_type_recall(test_pairs, best["_test_scores"], types)
    for e in epochs:
        e.pop("_test_scores")

    complete = len(epochs) == 10
    summary = {
        "system": "FA-AST+GMN (Wang et al., SANER 2020) re-run -- BCB",
        "dataset": (f"BCB, pair lists IDENTICAL to the MAGNET BCB re-run "
                    f"(dev {len(dev_pairs):,} / test {len(test_pairs):,}); "
                    f"built by tools/prep_faast_bcb_from_magnet.py, which is a "
                    f"lossless round-trip of FA-AST's own stock splits"),
        "protocol": (f"run_bcb.py --graphmode {args.graphmode} --data_setting 0, "
                     f"upstream defaults otherwise (10-epoch budget, threshold 0, "
                     f"batch 32, 4 layers, lr 1e-3); their native decision rule "
                     f"clone iff sign(cosine) > 0; epoch selected on DEV F1 (B2), "
                     f"never on test"),
        "epochs_completed": len(epochs),
        "status": "COMPLETE (10/10 epochs)" if complete
                  else f"PARTIAL ({len(epochs)}/10 epochs) -- dev-selected among "
                       f"completed epochs, ASTNN/MAGNET salvage precedent",
        "result": {
            "selected_epoch": best["epoch"],
            "dev_f1_of_selected": best["dev"]["f1"],
            "test_precision": best["test"]["precision"],
            "test_recall": best["test"]["recall"],
            "test_f1": best["test"]["f1"],
            "test_counts": {k: best["test"][k] for k in ("tp", "tn", "fp", "fn", "n")},
        },
        "per_type_recall_at_selected_epoch": recall,
        "per_type_clone_counts": counts,
        "per_type_unmapped_clone_pairs": unmapped,
        "per_epoch": epochs,
        "notes": [
            "Published FA-AST BCB numbers used the authors' full training set; ours "
            "uses the 100k-pair stratified cap shared with MAGNET -- caption note.",
            "Trained on ALL clone types incl. WT3/T4 (regime-all analogue), so any "
            "high WT3/T4 recall here is learned from labels -- the Krinke point.",
            "Effective dev/test sizes equal the full prepped splits (no native drops), "
            "whereas MAGNET loaded 19,857/99,791 after its own skips; input pair lists "
            "are identical, per-tool effective counts differ by native constraints only.",
            "run_bcb.py declares --num_epochs without type=int; the flag is deliberately "
            "NOT passed (its default is already the int 10).",
        ],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "summary.json" if complete
                       else f"summary_partial_{len(epochs)}epochs.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    try:
        from tools.run_meta import save_run_meta
        save_run_meta(OUT_DIR, extra={"epochs_completed": len(epochs),
                                      "selected_epoch": best["epoch"],
                                      "graphmode": args.graphmode})
    except Exception as exc:                      # provenance is best-effort
        print(f"(run_meta not written: {exc})")

    print(f"\nwrote {out}")
    print(f"selected epoch {best['epoch']} (dev F1 {best['dev']['f1']:.4f}) -> "
          f"test P {best['test']['precision']:.4f} "
          f"R {best['test']['recall']:.4f} F1 {best['test']['f1']:.4f}")
    print("per-type recall:", {k: round(v, 3) for k, v in recall.items()})


if __name__ == "__main__":
    main()
