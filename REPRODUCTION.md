# FuseCC — Reproduction Guide

Maps every reported result to the exact script, config, and output file that produces it.
All experiments are deterministic (`RANDOM_STATE=12345` everywhere; see `seeds.py`).

## Environment

```bash
conda create -n fusecc python=3.10
conda activate fusecc
pip install -r requirements-pinned.txt
```

GPU (CUDA) is used for transformer embedding extraction, encoder fine-tuning, and
XGBoost (`tree_method="hist", device="cuda"`); everything falls back to CPU but slower.
Reference hardware: Dell Precision 7920, Xeon Gold 6132, 16 GB GPU.

## Data layout expected

```
dataset/BCB/id2sourcecode_extracted/id2sourcecode/<id>.java   # BCB function sources
dataset/BCB/{clone,nonclone,type-1..5}.csv                    # BCB pair lists (see README.md, Datasets)
dataset/GCJ/googlejam4/*.java                                 # 1,665 GCJ files
dataset/GCJ/pairs_mapped.csv                                  # 544,206 GCJ pairs
```

## Determinism

All seeds are `RANDOM_STATE = 12345`. Metric t16 (winnowing) uses a stable 64-bit BLAKE2b
hash of each character k-gram, so it is reproducible run to run and across machines. The
results reported in the paper were computed with Python's built-in `hash()`, which is
randomised per process, so recomputed t16 values may differ slightly from those behind the
published numbers. The other eighteen metrics are unchanged and reproduce exactly.

## Pipeline stages → scripts → outputs

| # | Stage | Script / config | Output |
|---|---|---|---|
| 0 | BCB / GCJ pair lists with relative paths | `tools/prepare_bcb_pairs.py`, `tools/prepare_gcj_pairs.py` | `dataset/{BCB,GCJ}/pairs_mapped.csv` |
| 1 | BCB canonical pair list + clone types | `tools/build_bcb_type_mapping.py` | `dataset/BCB/pairs_with_types.csv`, `pairs_canonical.csv` |
| 2 | Hand-crafted metrics t1–t19 (BCB) | `experiments/run_experiment.py config/bcb_19_canonical.yaml` | `results/bcb_19_canonical/results.csv` |
| 3 | Hand-crafted metrics t1–t19 (GCJ) | `experiments/run_experiment.py config/gcj_19.yaml` | `results/gcj_19/results.csv` |
| 4 | Frozen-encoder embeddings (GCJ + BCB) | `tools/extract_code_embeddings_v2.py --model unixcoder [--pairs ... --out-dir ...]` | `dataset/{GCJ,BCB}/embeddings_unixcoder.pkl` |
| 5 | UniXcoder fine-tune, file-disjoint | `tools/finetune_unixcoder_gcj.py` | `models/unixcoder_gcj_ft/` |
| 6 | UniXcoder fine-tune, problem-disjoint | `tools/finetune_unixcoder_gcj.py --split problem --out-dir models/unixcoder_gcj_ft_pd` | `models/unixcoder_gcj_ft_pd/` |
| 7 | GCJ 148-feature matrices | `tools/build_gcj_embed_features.py` (zero-shot) / `tools/build_gcj_embed_features_ft.py` (ft, PCA on encoder-train files only) | `results/gcj_148{,_ft,_ft_pd}/features.npz` |

## Paper results → command → artifact

| Result (Fmeasure) | Command | Artifact |
|---|---|---|
| BCB T1–T3 0.9986 / all-pairs 0.9931; t16/t19/148 × regime matrix; WT3/T4 probe | `python tools/train_bcb_e3.py` | `results/bcb_e3/summary.json` |
| BCB per-type on published features (0.9742 repro; T1–T3 0.9932; probe 26%) | `python tools/eval_bcb_per_type.py` | `results/bcb_per_type/summary.json` |
| GCJ random-split 0.9998 / file-disjoint 0.9830 (threshold on val) | `python tools/eval_gcj_ft_protocols.py` | `results/gcj_148_ft/protocol_eval.json` |
| GCJ problem-disjoint (zero-shot 0.877/AUC 0.884; hand-crafted 0.733; pd-ft 0.639) | `python tools/eval_gcj_pd.py [--features ... --n-cols 19 --tag ...]` | `results/gcj_148_ft_pd/protocol_eval*.json`, `results/gcj_148/protocol_eval_zeroshot148.json` |
| GCJ stacked ensemble (0.9681 zero-shot / 0.9999 ft) | `python tools/train_enhanced_gcj_v2.py --features ... --out-dir ...` | `results/gcj_148{,_ft}/classification_enhanced/summary.json` |
| Baselines: CodeBERT/GraphCodeBERT/UniXcoder embedding-only | `python tools/baseline_embed_only_gcj.py --model <enc>` | `results/gcj_baselines/<enc>.json` |
| Complementarity: SHAP, permutation importance, correlations | `python tools/f1_analysis.py [--only bcb|gcj|gcj_zs]` | `results/f1_analysis/<ds>/` (figures + CSV) |
| Threshold sensitivity sweep curves | `python tools/f2_threshold_sensitivity.py` | `results/f2_threshold/` |
| Ensemble-design ablation (voting vs LR/MLP/XGB meta) | `python tools/d2_meta_comparison.py` | `results/d2_meta_comparison/summary.json` |

## Protocol definitions (see paper §evaluation)

- **random-split**: StratifiedKFold over pairs. Comparable to ASTNN/FCCA/Code2Vec numbers.
- **file-disjoint**: test pairs drawn only among held-out *files* (the encoder's
  `val_files.txt`); decision threshold picked on a disjoint validation carve-out.
- **problem-disjoint**: encoder-train / threshold-val / test split by *contest problem*
  ({1,5,6,8} / {4,7,11,12} / {2,3,9,10}); test set is 74.1% clones — report AUC+P/R/F1.

## Provenance notes

- `experiments/run_experiment.py` writes `git_sha.txt`, `config_used.yaml`,
  `pip_freeze.txt` into every output directory.
- The 16-metric BCB features (`results/real_fusecc/`, `config/real_fusecc.yaml`) come from
  an earlier tokenizer that replaced identifiers line by line, and from a pair list with
  9,936 duplicate pairs. All other experiments use the whole-file tokenizer, which keeps raw
  identifiers (abstraction is metric t17), and the de-duplicated `pairs_canonical.csv`.
