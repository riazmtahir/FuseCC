"""IV-F / RQ5: stage-wise efficiency measurement of the revised FuseCC pipeline.

Measures, genuinely, on this machine (single process unless noted):
  A. tokenization           files/s (javalang whole-file, tokenizer_core)
  B. 19-metric extraction   pairs/s (tokens in memory -> all t1..t19 kernels)
  C. UniXcoder embedding    files/s on GPU (512-token window, mean-pooled)
  D. PCA interactions       pairs/s vectorized (cos + |a-b| + a*b in 64-d PCA)
  E. XGBoost train/predict  on 19 and 148 features, GPU and CPU
Extrapolates corpus-level wall-clock (notes single-core vs the 28-core parallel
extraction used in production). Classifier-comparison per-family times already
measured in results/rq1_classifiers/*/summary.json (stage2 fit_s/pred_s).

Usage: rq5_efficiency.py [--pairs N] [--files N]
Output: results/rq5_efficiency/summary.json
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = _REPO_ROOT
SEED = 12345
sys.path.insert(0, REPO)


def stage_a_tokenize(paths):
    from tokenizer_core import extract_lexical_stream
    t0 = time.time()
    toks = {}
    for p in paths:
        toks[p] = extract_lexical_stream(p) or []
    dt = time.time() - t0
    ntok = np.mean([len(v) for v in toks.values()])
    return toks, {"files": len(paths), "seconds": round(dt, 2),
                  "files_per_s": round(len(paths) / dt, 1),
                  "mean_tokens_per_file": round(float(ntok), 1)}


def stage_b_metrics(toks, pair_paths):
    from similarity_dispatcher import SIM_MAP, INPUT_TYPE, normalize_inputs
    per_metric = {}
    t0 = time.time()
    for a, b in pair_paths:
        norm = normalize_inputs(toks[a], toks[b])
        for mid in [f"t{i}" for i in range(1, 20)]:
            ta = time.time()
            SIM_MAP[mid](*norm[INPUT_TYPE.get(mid, "tokens")])
            per_metric[mid] = per_metric.get(mid, 0.0) + time.time() - ta
    dt = time.time() - t0
    return {"pairs": len(pair_paths), "seconds": round(dt, 2),
            "pairs_per_s_1core": round(len(pair_paths) / dt, 2),
            "ms_per_pair_all19": round(1000 * dt / len(pair_paths), 2),
            "slowest_metrics_ms_per_pair": {
                k: round(1000 * v / len(pair_paths), 2)
                for k, v in sorted(per_metric.items(), key=lambda x: -x[1])[:5]}}


def stage_c_embed(paths):
    import torch
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base").cuda().eval()
    load_s = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            texts = [open(p, encoding="utf-8", errors="ignore").read()
                     for p in paths[i:i + 16]]
            enc = tok(texts, truncation=True, max_length=512, padding=True,
                      return_tensors="pt").to("cuda")
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            ((out * mask).sum(1) / mask.sum(1)).cpu().numpy()
    dt = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return {"files": len(paths), "model_load_s": round(load_s, 1),
            "seconds": round(dt, 2), "files_per_s_gpu": round(len(paths) / dt, 1)}


def stage_d_interactions(n_pairs):
    import pickle
    with open(f"{REPO}/results/gcj_148/pca_unixcoder.pkl", "rb") as f:
        pca_obj = pickle.load(f)
    with open(f"{REPO}/dataset/GCJ/embeddings_unixcoder.pkl", "rb") as f:
        emb = pickle.load(f)
    E = np.stack(list(emb.values())).astype(np.float32)
    rng = np.random.RandomState(SEED)
    ia = rng.randint(0, len(E), n_pairs)
    ib = rng.randint(0, len(E), n_pairs)
    pca = pca_obj["pca"] if isinstance(pca_obj, dict) and "pca" in pca_obj else pca_obj
    mean = pca_obj.get("mean") if isinstance(pca_obj, dict) else None
    t0 = time.time()
    A, B = E[ia], E[ib]
    if mean is not None:
        A, B = A - mean, B - mean
    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
    Pa, Pb = pca.transform(A), pca.transform(B)
    np.hstack([cos[:, None], np.abs(Pa - Pb), Pa * Pb])
    dt = time.time() - t0
    return {"pairs": int(n_pairs), "seconds": round(dt, 3),
            "pairs_per_s": round(n_pairs / dt, 0)}


def stage_e_train():
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    d = np.load(f"{REPO}/results/gcj_148/features.npz", allow_pickle=True)
    X = np.nan_to_num(d["X"].astype(np.float32), nan=0.0)
    y = d["y"].astype(int)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                              random_state=SEED)
    out = {}
    for label, cols, dev in [("xgb19_gpu", slice(0, 19), "cuda"),
                             ("xgb148_gpu", slice(None), "cuda"),
                             ("xgb19_cpu", slice(0, 19), "cpu"),
                             ("xgb148_cpu", slice(None), "cpu")]:
        m = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                          tree_method="hist", device=dev, n_jobs=-1,
                          random_state=SEED, eval_metric="logloss")
        t0 = time.time()
        m.fit(X[tr][:, cols], y[tr])
        fit_s = time.time() - t0
        t0 = time.time()
        m.predict_proba(X[te][:, cols])
        pred_s = time.time() - t0
        out[label] = {"train_pairs": len(tr), "fit_s": round(fit_s, 1),
                      "test_pairs": len(te), "pred_s": round(pred_s, 2)}
        print(f"  {label}: fit {fit_s:.1f}s pred {pred_s:.2f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--files", type=int, default=200)
    args = ap.parse_args()
    out_dir = f"{REPO}/results/rq5_efficiency"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(SEED)

    gcj_dir = f"{REPO}/dataset/GCJ/googlejam4"
    all_files = sorted(os.path.join(gcj_dir, f) for f in os.listdir(gcj_dir)
                       if f.endswith(".java"))
    print(f"GCJ corpus: {len(all_files)} files", flush=True)

    print("[A] tokenization (all GCJ files)", flush=True)
    toks, a = stage_a_tokenize(all_files)

    print("[B] 19-metric extraction", flush=True)
    pairs = [(all_files[i], all_files[j]) for i, j in zip(
        rng.randint(0, len(all_files), args.pairs),
        rng.randint(0, len(all_files), args.pairs))]
    b = stage_b_metrics(toks, pairs)
    print(f"  {b['pairs_per_s_1core']} pairs/s 1-core", flush=True)

    print("[C] UniXcoder embedding (GPU)", flush=True)
    c = stage_c_embed(all_files[:args.files])

    print("[D] PCA interaction features", flush=True)
    dd = stage_d_interactions(544_206)

    print("[E] XGBoost train/predict (GCJ 544k pairs)", flush=True)
    e = stage_e_train()

    n_pairs_gcj = 544_206
    summary = {
        "hardware": "Dell Precision 7920T, Xeon Gold 6132 (28C/56T), Quadro GPU",
        "corpus": {"gcj_files": len(all_files), "gcj_pairs": n_pairs_gcj},
        "A_tokenization": a,
        "B_metric_extraction": b,
        "C_unixcoder_embedding": c,
        "D_pca_interactions": dd,
        "E_xgboost_gcj": e,
        "extrapolated_gcj_corpus_wallclock_s": {
            "tokenize_all_files": round(len(all_files) / a["files_per_s"], 1),
            "metrics_544k_pairs_28core": round(
                n_pairs_gcj / (b["pairs_per_s_1core"] * 28), 1),
            "embed_all_files_gpu": round(len(all_files) / c["files_per_s_gpu"], 1),
            "pca_interactions_544k": dd["seconds"],
            "train_148_gpu": e["xgb148_gpu"]["fit_s"],
            "predict_109k_gpu": e["xgb148_gpu"]["pred_s"],
        },
        "note": "1-core metric rate extrapolated to the 28-core parallel "
                "extraction used by experiments/run_experiment.py; classifier "
                "per-family times in results/rq1_classifiers/*/summary.json",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    sys.path.insert(0, f"{REPO}/tools")
    try:
        from run_meta import save_run_meta
        save_run_meta(out_dir)
    except Exception as ex:
        print(f"run_meta skipped: {ex!r}")
    print(json.dumps(summary["extrapolated_gcj_corpus_wallclock_s"], indent=2),
          flush=True)


if __name__ == "__main__":
    main()
