#!/usr/bin/env python3
"""Diagnose WHY FuseCC's MAP@R trails CONCORD, with numbers not guesses.

FuseCC's MAP@R is two-stage: UniXcoder cosine retrieves the top-K candidates, then the
148-feature classifier re-ranks THEM. So its MAP@R is hard-capped by stage-1 recall@K:
a true clone UniXcoder ranks below position K is never seen by the re-ranker and can
never be recovered. CONCORD is single-stage — its fine-tuned embedding ranks all samples
directly — so it has no such cap.

This computes, per encoder, recall@K for the K values that matter, and the ORACLE MAP@R
of a two-stage design (= what a perfect re-ranker on the top-K pool could achieve). That
separates the two possible causes:
  (a) low stage-1 recall  -> the cap is retrieval; a better/bigger stage 1 is the fix
  (b) high recall, low realised MAP@R -> the re-ranker is leaving score on the table
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import sys

import numpy as np

DATA_ROOT = _REPO_ROOT + "/dataset/concord"
OUT = _REPO_ROOT + "/results/concord"
import pickle


def load(ds, tag):
    d = os.path.join(DATA_ROOT, ds)
    with open(os.path.join(d, "code_store.pkl"), "rb") as fh:
        store = pickle.load(fh)
    labels = np.array([l for _i, l, _c in store["test"]])
    fn = "embeddings_test.npy" if tag == "unixcoder" else f"embeddings_{tag}_test.npy"
    E = np.load(os.path.join(d, fn)).astype(np.float32)
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    return labels, En


def recall_and_oracle(labels, En, Ks):
    """For each query: recall@K and oracle AP@R (perfect re-rank of the top-K)."""
    n = len(labels)
    from collections import Counter
    counts = Counter(labels.tolist())
    R_of = np.array([counts[l] - 1 for l in labels])
    rec = {K: [] for K in Ks}
    oracle = {K: [] for K in Ks}
    for qi in range(n):
        sims = En @ En[qi]
        sims[qi] = -np.inf
        order = np.argsort(-sims)                    # full ranking
        is_clone = labels[order] == labels[qi]
        R = R_of[qi]
        for K in Ks:
            in_topK = int(is_clone[:K].sum())
            rec[K].append(in_topK / R if R else 0.0)
            # oracle: put all true clones found in top-K first -> they fill positions
            # 1..min(in_topK,R), each precision 1 -> AP@R = min(in_topK,R)/R
            oracle[K].append(min(in_topK, R) / R if R else 0.0)
    return ({K: float(np.mean(v)) for K, v in rec.items()},
            {K: float(np.mean(v)) for K, v in oracle.items()})


def main():
    mapr_ref = {}
    for ds in ("java250", "poj104"):
        mp = json.load(open(os.path.join(OUT, ds, "mapr.json")))
        mapr_ref[ds] = {k: mp[k]["map_at_r"] for k in mp}

    report = {}
    for ds in ("java250", "poj104"):
        labels, En = load(ds, "unixcoder")
        from collections import Counter
        R = int(np.median([Counter(labels.tolist())[l] - 1 for l in labels]))
        Ks = [R, R + 100, 2 * R, 5 * R, 10 * R, len(labels) - 1]
        Ks = sorted(set(min(K, len(labels) - 1) for K in Ks))
        rec, oracle = recall_and_oracle(labels, En, Ks)
        report[ds] = {"R": R, "n_test": len(labels),
                      "unixcoder_recall_at_K": {str(K): round(rec[K], 4) for K in Ks},
                      "twostage_oracle_mapr_at_K": {str(K): round(oracle[K], 4) for K in Ks},
                      "realised": {"unixcoder_cosine": mapr_ref[ds].get("unixcoder"),
                                   "fusecc_rerank": mapr_ref[ds].get("fusecc_rerank"),
                                   "concord_ours": mapr_ref[ds].get("concord_ours")}}
        print(f"\n===== {ds}  (R={R}, n_test={len(labels)}) =====")
        print(f"{'K':>7} {'uniXcoder recall@K':>20} {'2-stage oracle MAP@R':>22}")
        for K in Ks:
            tag = "  <- our K (R+100)" if K == R + 100 else (
                  "  <- full pool" if K == len(labels) - 1 else "")
            print(f"{K:>7} {rec[K]:>20.4f} {oracle[K]:>22.4f}{tag}")
        r = report[ds]["realised"]
        print(f"realised: UniXcoder {r['unixcoder_cosine']:.4f} | "
              f"FuseCC re-rank {r['fusecc_rerank']:.4f} | "
              f"CONCORD {r['concord_ours']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "mapr_ceiling_diagnosis.json"), "w"), indent=2)
    print("\nwrote", os.path.join(OUT, "mapr_ceiling_diagnosis.json"))


if __name__ == "__main__":
    main()
