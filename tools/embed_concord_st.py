#!/usr/bin/env python3
"""Embed ConCord test splits with 2025/2026 code embedding models (sentence-transformers).

Adds recent zero-shot code encoders to the MAP@R comparison. Each model is used in its
INTENDED retrieval configuration (correct prompts / pooling from its own ST config), so
the numbers are the model at its honest best — not undersold by a wrong recipe.

For code-to-code clone RETRIEVAL each item is both a query and a candidate, so instruction
models get an asymmetric dual encoding: a `_query` embedding (query prompt) and a `_doc`
embedding (document prompt). eval_concord_mapr.py --tag X --doc-tag X_doc then scores
Qn @ Dn.T, the standard retrieval setup these models are benchmarked with. Models with no
distinct prompts collapse to the symmetric case (query==doc).

Row order matches code_store.pkl (raw file order), asserted, so labels align in MAP@R.

Models (sizes fit the 16 GB GPU in fp16):
  coderankembed : nomic-ai/CodeRankEmbed            137M  2025 code-retrieval bi-encoder
  jinacode      : jinaai/jina-code-embeddings-0.5b  0.5B  2025 (Qwen2.5-Coder), code2code task
  qwen3emb      : Qwen/Qwen3-Embedding-0.6B         0.6B  2025/26 MTEB-topping embedder

Usage (test split only — MAP@R needs test):
  python tools/embed_concord_st.py --dataset poj104 --model jinacode
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import pickle

import numpy as np

DATA_ROOT = _REPO_ROOT + "/dataset/concord"

# model -> (hf id, query_prompt_name, doc_prompt_name). Prompt names index the model's own
# ST `prompts` dict (verified present). None -> no prompt (plain document encoding).
MODELS = {
    "coderankembed": ("nomic-ai/CodeRankEmbed", None, None),          # NL->code; code-code symmetric, no prompt
    "jinacode": ("jinaai/jina-code-embeddings-0.5b",
                 "code2code_query", "code2code_document"),            # its exact task
    "qwen3emb": ("Qwen/Qwen3-Embedding-0.6B", "query", "document"),   # instruct query / plain doc
    # SYMMETRIC jina for use as FuseCC's stage-1 BACKBONE (one representation per
    # sample, needed by the binary pair pipeline + re-rank). Uses the code2code_document
    # prompt for every sample so query==doc.
    "jinacode_sym": ("jinaai/jina-code-embeddings-0.5b",
                     "code2code_document", "code2code_document"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--splits", default="test",
                    help="comma list of splits to embed (backbone use needs train,valid,test)")
    ap.add_argument("--max-seq", type=int, default=512,
                    help="cap sequence length (512 matches our other encoders and "
                         "prevents OOM from the models' 8k-32k default context on long "
                         "snippets; mean snippet is ~230 tokens)")
    args = ap.parse_args()

    hf, qp, dp = MODELS[args.model]
    symmetric = (qp is None) or (qp == dp)   # one representation per sample
    d = os.path.join(DATA_ROOT, args.dataset)
    with open(os.path.join(d, "code_store.pkl"), "rb") as fh:
        store = pickle.load(fh)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(hf, trust_remote_code=True, device="cuda",
                                model_kwargs={"torch_dtype": "float16"})
    model.max_seq_length = args.max_seq
    prompts = getattr(model, "prompts", {}) or {}

    def encode(codes, prompt_name):
        kw = {}
        if prompt_name and prompts.get(prompt_name):
            kw["prompt"] = prompts[prompt_name]
        emb = model.encode(codes, batch_size=args.batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True, **kw)
        return emb.astype(np.float32)

    for split in args.splits.split(","):
        codes = [c for _i, _l, c in store[split]]
        print(f"{args.dataset}/{args.model} [{split}]: {len(codes):,} samples")
        if symmetric:
            emb = encode(codes, qp)      # qp may be None (no prompt) or a doc prompt
            np.save(os.path.join(d, f"embeddings_{args.model}_{split}.npy"), emb)
            print(f"  -> embeddings_{args.model}_{split}.npy {emb.shape} (symmetric)")
        else:
            q = encode(codes, qp)
            np.save(os.path.join(d, f"embeddings_{args.model}_{split}.npy"), q)
            dcs = encode(codes, dp)
            np.save(os.path.join(d, f"embeddings_{args.model}_doc_{split}.npy"), dcs)
            print(f"  -> embeddings_{args.model}[_doc]_{split}.npy {q.shape} (asymmetric)")

    meta_path = os.path.join(d, f"embeddings_{args.model}.meta.json")
    json.dump({"hf_model": hf, "symmetric": symmetric,
               "query_prompt_name": qp, "doc_prompt_name": dp,
               "query_prompt": prompts.get(qp) if qp else None,
               "doc_prompt": prompts.get(dp) if dp else None,
               "normalize": True, "dtype": "float16", "splits": args.splits,
               "max_seq": args.max_seq},
              open(meta_path, "w"), indent=2)


if __name__ == "__main__":
    main()
