#!/usr/bin/env python3
"""MAP@R on the ConCord POJ-104 / Java250 test splits — the OFFICIAL benchmark metric.

Why this exists: these benchmarks are retrieval tasks. CodeXGLUE/CONCORD score them with
MAP@R (R = number of true clones per query = solutions-per-problem − 1, so 499 for
POJ-104 and 299 for Java250), and CONCORD reports 91.5 / 86.5. Binary pair F1 is not
comparable to any published number on these splits, so this script produces the row that
can actually sit in a baseline table.

MAP@R follows the CodeXGLUE definition: for each query, rank every other test sample,
take the top R, and average precision@k over the positions where a true clone was
retrieved, divided by R.

Two scorers, so our system and every encoder baseline go through the SAME evaluator:
  --scorer embedding : exact, all queries. Cosine over mean-pooled embeddings. Used for
                       UniXcoder / CodeBERT / GraphCodeBERT / CONCORD.
  --scorer fusecc  : two-stage. Embeddings retrieve the top-K candidates, then the
                       trained FuseCC classifier re-ranks them. Necessary because the
                       19-metric scorer costs ~125 ms/pair and POJ-104 test alone is
                       ~72M pairs; --queries samples the query set and the caption must
                       say so. Candidates outside the top-K keep their stage-1 order,
                       so K must be >= R or recall is capped by construction.

Usage:
    python tools/eval_concord_mapr.py --dataset java250 --scorer embedding --tag unixcoder
    python tools/eval_concord_mapr.py --dataset poj104 --scorer fusecc --queries 1000
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, _REPO_ROOT)
RANDOM_STATE = 12345
DATA_ROOT = _REPO_ROOT + "/dataset/concord"
OUT_ROOT = _REPO_ROOT + "/results/concord"


def load_split(dataset, split="test"):
    d = os.path.join(DATA_ROOT, dataset)
    with open(os.path.join(d, "code_store.pkl"), "rb") as fh:
        store = pickle.load(fh)
    rows = store[split]
    labels = np.array([lab for _i, lab, _c in rows])
    codes = [c for _i, _l, c in rows]
    return d, labels, codes


def emb_path(d, tag, split):
    return os.path.join(d, f"embeddings_{split}.npy" if tag in ("", "unixcoder")
                        else f"embeddings_{tag}_{split}.npy")


def average_precision_at_r(ranked_is_clone, R):
    """CodeXGLUE MAP@R: sum_k P(k)*rel(k) over the top R, divided by R."""
    hits = 0
    ap = 0.0
    for k in range(min(R, len(ranked_is_clone))):
        if ranked_is_clone[k]:
            hits += 1
            ap += hits / (k + 1)
    return ap / R if R > 0 else 0.0


def mapr_from_scores(score_fn, labels, query_idx, R_per_query, topk_only=None):
    aps = []
    n = len(labels)
    for qi in query_idx:
        scores = score_fn(qi)               # length n, higher = more similar
        scores[qi] = -np.inf                # never retrieve the query itself
        R = R_per_query[qi]
        k = n - 1 if topk_only is None else min(topk_only, n - 1)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        aps.append(average_precision_at_r(labels[top] == labels[qi], R))
    return float(np.mean(aps)), aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scorer", choices=["embedding", "fusecc"], default="embedding")
    ap.add_argument("--tag", default="unixcoder", help="embedding tag (encoder name)")
    ap.add_argument("--doc-tag", default=None,
                    help="candidate/document embedding tag for asymmetric retrieval "
                         "(instruction models with distinct query/doc prompts)")
    ap.add_argument("--queries", type=int, default=0,
                    help="0 = all queries (exact); >0 = random sample (required for "
                         "--scorer fusecc)")
    ap.add_argument("--topk", type=int, default=0,
                    help="stage-1 candidate pool for --scorer fusecc (default R+100)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--model-suffix", default="",
                    help="re-ranker artifact suffix (e.g. _jina -> xgb_148_jina.joblib); "
                         "pair with --tag jinacode_sym so stage-1 + features use jina")
    ap.add_argument("--label", default=None, help="name for this row in the artifact")
    args = ap.parse_args()

    d, labels, codes = load_split(args.dataset, "test")
    n = len(labels)
    counts = Counter(labels.tolist())
    R_per_query = np.array([counts[l] - 1 for l in labels])
    R_uniform = int(np.median(R_per_query))
    print(f"{args.dataset} test: {n:,} samples, {len(counts)} problems, "
          f"R={R_uniform} (median true clones per query)")

    E = np.load(emb_path(d, args.tag, "test")).astype(np.float32)
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    # Asymmetric retrieval: query rep from --tag, candidate rep from --doc-tag (the
    # standard setup for instruction embedders whose query/doc prompts differ). The query
    # side ranks against the DOCUMENT matrix. Symmetric when --doc-tag is omitted.
    if getattr(args, "doc_tag", None):
        D = np.load(emb_path(d, args.doc_tag, "test")).astype(np.float32)
        Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-9, None)
    else:
        Dn = En

    rng = np.random.default_rng(RANDOM_STATE)
    if args.queries and args.queries < n:
        query_idx = np.sort(rng.choice(n, size=args.queries, replace=False))
        sampled = True
    else:
        query_idx = np.arange(n)
        sampled = False

    t0 = time.time()
    if args.scorer == "embedding":
        def score_fn(qi):
            return Dn @ En[qi]
        mapr, aps = mapr_from_scores(score_fn, labels, query_idx, R_per_query)
        row = {"scorer": f"embedding-cosine ({args.tag})", "sampled_queries": sampled}
    else:
        if not args.queries:
            sys.exit("--scorer fusecc requires --queries (all-pairs is infeasible)")
        import joblib
        from tools.extract_concord_features import tokenize
        from tools.mapr_rerank import score_pairs_parallel  # batched, parallel

        msfx = getattr(args, "model_suffix", "") or ""
        model_path = os.path.join(OUT_ROOT, args.dataset, f"xgb_148{msfx}.joblib")
        pca_path = os.path.join(OUT_ROOT, args.dataset, f"pca{msfx}.joblib")
        if not (os.path.exists(model_path) and os.path.exists(pca_path)):
            sys.exit(f"need {model_path} and {pca_path} — run tools/train_concord.py first")
        clf = joblib.load(model_path)
        pca, mean_vec = joblib.load(pca_path)

        toks = [tokenize(c) for c in codes]
        # K must be >= R or the candidate pool caps recall; default just above R
        K = args.topk or (R_uniform + 100)
        Ec = E - mean_vec
        Pc = pca.transform(Ec).astype(np.float32)
        Ecn = Ec / np.clip(np.linalg.norm(Ec, axis=1, keepdims=True), 1e-9, None)

        # Stage 1: cosine retrieval of top-K per query (vectorised, cheap)
        cand_per_q = {}
        all_pairs = []
        for qi in query_idx:
            base = Ecn @ Ecn[qi]
            base[qi] = -np.inf
            cand = np.argpartition(-base, K - 1)[:K]
            cand_per_q[qi] = cand
            all_pairs.extend((qi, int(cj)) for cj in cand)
        print(f"stage-1 retrieved {len(all_pairs):,} (query,candidate) pairs "
              f"(K={K}); extracting 148 features in parallel...", flush=True)

        # Stage 2: extract 148 features for ALL pairs at once, in parallel, then predict
        t_feat = time.time()
        feats = score_pairs_parallel(all_pairs, toks, Ecn, Pc, workers=args.workers)
        prob = clf.predict_proba(feats)[:, 1]
        print(f"  scored {len(all_pairs):,} pairs in {time.time()-t_feat:.0f}s "
              f"({len(all_pairs)/max(time.time()-t_feat,1e-9):.0f} pairs/s)", flush=True)

        # map (qi,cj) -> prob, then re-rank each query's candidates
        pos = 0
        prob_of = {}
        for qi in query_idx:
            for cj in cand_per_q[qi]:
                prob_of[(qi, int(cj))] = prob[pos]
                pos += 1
        aps = []
        for qi in query_idx:
            base = Ecn @ Ecn[qi]
            base[qi] = -np.inf
            cand = cand_per_q[qi]
            base[cand] = 1.0 + np.array([prob_of[(qi, int(cj))] for cj in cand])
            R = R_per_query[qi]
            top = np.argpartition(-base, R - 1)[:R]
            top = top[np.argsort(-base[top])]
            aps.append(average_precision_at_r(labels[top] == labels[qi], R))
        mapr = float(np.mean(aps))
        row = {"scorer": f"FuseCC 148-feature re-rank over top-{K} "
                         f"({args.tag} stage-1 retrieval)", "sampled_queries": True,
               "topk": K}

    el = time.time() - t0
    row.update({
        "dataset": args.dataset,
        "metric": "MAP@R (CodeXGLUE definition)",
        "R_median": R_uniform,
        "n_test_samples": n,
        "n_queries_evaluated": len(query_idx),
        "map_at_r": round(mapr, 4),
        "map_at_r_std_over_queries": round(float(np.std(aps)), 4),
        "seconds": round(el, 1),
        "reference_published": {"CONCORD_poj104": 0.915, "CONCORD_java250": 0.865},
        "note": ("R is the median number of true clones per query (POJ-104 499 / "
                 "Java250 299). Sampled-query runs are marked; all-query runs are exact."),
    })
    label = args.label or (f"{args.scorer}_{args.tag}" if args.scorer == "embedding"
                           else "fusecc")
    out_dir = os.path.join(OUT_ROOT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "mapr.json")
    allrows = json.load(open(out_file)) if os.path.exists(out_file) else {}
    allrows[label] = row
    json.dump(allrows, open(out_file, "w"), indent=2)
    print(f"\nMAP@R = {mapr:.4f}  ({len(query_idx):,} queries, {el:.0f}s)"
          f"{'  [SAMPLED]' if sampled or args.scorer == 'fusecc' else '  [exact]'}")
    print("wrote", out_file, f"[{label}]")


if __name__ == "__main__":
    main()
