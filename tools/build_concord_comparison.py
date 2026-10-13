#!/usr/bin/env python3
"""Assemble the POJ-104 / Java250 comparison table: our model vs baselines, both metrics.

Reads results/concord/<ds>/{summary.json,mapr.json} and emits a single
results/concord/comparison.json + a markdown table for the paper. Adds the CONCORD
reference rows (published + our re-run from the Paper-2 workspace logs) as clearly
labelled literature/re-run numbers — never mixed with the unified-evaluator rows.
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os

ROOT = _REPO_ROOT + "/results/concord"
DS = ["java250", "poj104"]

# CONCORD published numbers, kept only as a cross-check against our own evaluation.
# The comparison table's CONCORD row uses `concord_ours` from mapr.json — the fine-tuned
# checkpoint embedded and scored by OUR evaluator on OUR test pairs (same-environment).
CONCORD_PUBLISHED = {"java250": 0.865, "poj104": 0.915}


def main():
    out = {"datasets": {}, "notes": [
        "Binary rows: problem-disjoint pairs, threshold on valid (B2), test P/R/F1+AUC; "
        "all-clone baseline F1 = 0.667.",
        "MAP@R rows: CodeXGLUE definition (R=median true clones/query: Java250 299, "
        "POJ-104 499). Encoder rows are exact (all queries); FuseCC re-rank is "
        "sampled (500 queries at K=R+100, 300 at K=5R).",
        "2025/26 encoders (CodeRankEmbed, Qwen3-Embedding-0.6B, jina-code-embeddings-0.5b) "
        "are retrieval-specialist embedders used in their intended config (jina/Qwen3 "
        "with their own query/doc prompts, asymmetric; recipes in embeddings_<m>.meta.json). "
        "They are MUCH stronger on MAP@R than the 2020-2022 encoders and than FuseCC's "
        "two-stage re-rank -- expected: MAP@R retrieval is exactly what they are trained "
        "for. jina-code-0.5b even beats CONCORD's fine-tuned checkpoint zero-shot. This is "
        "reported straight; FuseCC's contribution is the pairwise clone task + "
        "complementarity + efficiency, NOT retrieval SOTA.",
        "FuseCC uses ZERO-SHOT UniXcoder + hand-crafted metrics, trained as a pair "
        "classifier. CONCORD is contrastively FINE-TUNED to optimise MAP@R on these "
        "train problems -> a different regime; its row is stronger on MAP@R by design.",
        "The CONCORD row is computed by OUR evaluator: its fine-tuned checkpoint "
        "(outputs/{codenet_run1,poj104_reproduce_best}) embeds our test split (raw "
        "SentencePiece code, CLS pooling) and is scored on our test pairs. Same "
        "environment as every other row -- NOT a published or self-logged number. "
        "Published values (0.865/0.915) are shown only as a cross-check: our POJ-104 "
        "0.910 reproduces it; our Java250 0.774 is below published 0.865 (their own "
        "finetune log peaked at 0.727 valid), an honest reproduction gap.",
    ]}

    for ds in DS:
        summ = json.load(open(os.path.join(ROOT, ds, "summary.json")))
        mapr = json.load(open(os.path.join(ROOT, ds, "mapr.json")))
        r = summ["results"]

        binary = {
            "handcrafted_t1_t19": r["xgb_t1_t19_only"]["test"],
            "embeddings_only_unixcoder": r["xgb_embeddings_only"]["test"],
            "fusecc_xgb148_unixcoder": r["xgb_19+emb148"]["test"],
            "fusecc_stacking_unixcoder": r["stacking_mlp"]["test_at_0.5"],
        }
        jpath = os.path.join(ROOT, ds, "summary_jina.json")
        if os.path.exists(jpath):
            rj = json.load(open(jpath))["results"]
            binary["embeddings_only_jina"] = rj["xgb_embeddings_only"]["test"]
            binary["fusecc_xgb148_jina"] = rj["xgb_19+emb148"]["test"]
            binary["fusecc_stacking_jina"] = rj["stacking_mlp"]["test_at_0.5"]

        mapr_rows = {k: mapr[k]["map_at_r"] for k in
                     ["codebert", "graphcodebert", "unixcoder",
                      "coderankembed", "qwen3emb", "jinacode",
                      "fusecc_rerank", "fusecc_rerank_bigK",
                      "jinasym", "fusecc_jina_rerank", "concord_ours"]
                     if k in mapr}
        mapr_rows["CONCORD_published_ref"] = CONCORD_PUBLISHED[ds]

        out["datasets"][ds] = {
            "language": summ["language"],
            "R": mapr["unixcoder"]["R_median"],
            "binary_pair_f1_auc": {k: {"f1": round(v["f1"], 4), "auc": round(v["auc"], 4)}
                                   for k, v in binary.items()},
            "map_at_r": {k: (round(v, 4) if v is not None else None)
                         for k, v in mapr_rows.items()},
        }

    with open(os.path.join(ROOT, "comparison.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # markdown
    lines = ["# POJ-104 / CodeNet-Java250 — FuseCC vs baselines", "",
             "## Binary pair protocol (problem-disjoint; F1 / AUC on test)", "",
             "| System | Java250 | POJ-104 |", "|---|---|---|"]
    bmap = [("Hand-crafted t1–t19 (XGB)", "handcrafted_t1_t19"),
            ("_UniXcoder backbone (2022):_", None),
            ("UniXcoder embeddings only", "embeddings_only_unixcoder"),
            ("**FuseCC 148 (UniXcoder)**", "fusecc_xgb148_unixcoder"),
            ("FuseCC + stacking (UniXcoder)", "fusecc_stacking_unixcoder"),
            ("_jina-code backbone (2025):_", None),
            ("jina embeddings only", "embeddings_only_jina"),
            ("**FuseCC 148 (jina)**", "fusecc_xgb148_jina"),
            ("FuseCC + stacking (jina)", "fusecc_stacking_jina")]
    for name, key in bmap:
        if key is None:
            lines.append(f"| {name} | | |")
            continue
        jb = out["datasets"]["java250"]["binary_pair_f1_auc"]
        pb = out["datasets"]["poj104"]["binary_pair_f1_auc"]
        if key not in jb:
            continue
        j, p = jb[key], pb[key]
        lines.append(f"| {name} | {j['f1']:.3f} / {j['auc']:.3f} | {p['f1']:.3f} / {p['auc']:.3f} |")
    lines += ["", "*All-clone baseline F1 = 0.667.*", "",
              "## MAP@R (official metric; comparable to CONCORD)", "",
              "| System | Regime | Java250 (R=299) | POJ-104 (R=499) |", "|---|---|---|---|"]
    mmap = [("CodeBERT cosine", "codebert", "zero-shot (2020)"),
            ("GraphCodeBERT cosine", "graphcodebert", "zero-shot (2020)"),
            ("UniXcoder cosine", "unixcoder", "zero-shot (2022)"),
            ("CodeRankEmbed cosine", "coderankembed", "zero-shot (2025)"),
            ("Qwen3-Embedding-0.6B", "qwen3emb", "zero-shot (2025/26)"),
            ("jina-code-embeddings-0.5b", "jinacode", "zero-shot (2025)"),
            ("**FuseCC re-rank** (UniXcoder, K=5R)", "fusecc_rerank_bigK", "frozen UniXcoder + metrics"),
            ("**FuseCC (jina backbone)**", "jinasym", "frozen jina, stage-1 only"),
            ("**FuseCC re-rank** (jina)", "fusecc_jina_rerank", "frozen jina + metrics"),
            ("CONCORD (our eval)", "concord_ours", "fine-tuned for retrieval"),
            ("CONCORD (published ref)", "CONCORD_published_ref", "fine-tuned")]
    for name, key, regime in mmap:
        jv = out["datasets"]["java250"]["map_at_r"].get(key)
        pv = out["datasets"]["poj104"]["map_at_r"].get(key)
        js = f"{jv:.3f}" if jv is not None else "—"
        ps = f"{pv:.3f}" if pv is not None else "—"
        lines.append(f"| {name} | {regime} | {js} | {ps} |")
    lines += ["", "*Encoder rows exact (all queries); FuseCC re-rank rows = 500 sampled "
              "queries. jina-backbone FuseCC: stage-1 alone 0.905/0.935; re-ranking with "
              "metrics is ~neutral-to-slightly-negative (−1 pp), i.e. a 2025 encoder largely "
              "subsumes the metrics' retrieval value. CONCORD is contrastively fine-tuned "
              "to optimise MAP@R (different regime).*"]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "comparison.md"), "w") as fh:
        fh.write(md)
    print(md)
    print("wrote", os.path.join(ROOT, "comparison.json"), "and comparison.md")


if __name__ == "__main__":
    main()
