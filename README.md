# FuseCC

Code for the paper **"Fusing Unsupervised Similarity Metrics with Frozen Code Embeddings for
Efficient and Generalizable Code Clone Detection"**.

FuseCC scores each candidate code pair with nineteen unsupervised similarity metrics
(lexical, edit-distance, frequency, sequence, fingerprint and semantic-oriented measures) and
complements them with interaction features from a frozen pre-trained code encoder
(centred cosine plus |a−b| and a⊙b in a 64-dimensional PCA space). The resulting
148-dimensional vector is classified by gradient-boosted trees (XGBoost, LightGBM, CatBoost)
with a stacked meta-learner. The encoder is not fine-tuned in the headline configuration,
so it can be replaced by a stronger one without retraining anything but the trees.

## Installation

Python 3.10 is required, and a CUDA GPU is recommended (embedding extraction, encoder
fine-tuning, XGBoost with `device="cuda"`); everything also runs on CPU, more slowly.

```bash
conda create -n fusecc python=3.10
conda activate fusecc
pip install -r requirements-pinned.txt
```

Run all scripts from the repository root. Paths are resolved relative to the repository and
can be overridden with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `FUSECC_ROOT` | repository root | the checkout containing the script |
| `FUSECC_EXTERNAL` | directory holding external assets (`baselines/`, `concord_assets/`, `CONCORD_ISSTA_23/`) | parent directory of the repository |
| `PYTHON` | interpreter used by the `tools/*.sh` drivers | `python` |
| `TOKEN_CACHE_DIR` | optional on-disk token cache for metric extraction | unset (no cache) |

Every experiment is seeded (`RANDOM_STATE = 12345`, see `seeds.py`).

### Reproducibility of metric t16

Metric t16 (winnowing fingerprint similarity) hashes character k-grams. It now uses a
stable 64-bit BLAKE2b hash, so its values are identical from run to run and across
machines. The results reported in the paper were computed with Python's built-in `hash()`,
which is randomised per process, so recomputed t16 values may differ slightly from those
behind the published numbers. The other eighteen metrics are unchanged and reproduce
exactly.

## Datasets

The datasets are not redistributed. Download them and place them as shown, then run the
preparation scripts from the repository root.

### BigCloneBench (BCB)

BigCloneBench: J. Svajlenko, J. F. Islam, I. Keivanloo, C. K. Roy and M. M. Mia, "Towards a
Big Data Curated Benchmark of Inter-Project Code Clones", ICSME 2014.

We use the function-level BigCloneBench subset distributed with the Amain repository,
<https://github.com/CGCL-codes/Amain> (`dataset/BCB/`). Our copies are identical to those
files; the CSVs differ only in line endings (CRLF upstream, LF here). Rename them as follows:

| Amain file | Place as | Content |
|---|---|---|
| `id2sourcecode.zip` | unzip into `dataset/BCB/id2sourcecode_extracted/` | one `.java` file per function id |
| `BCB_clone.csv` | `dataset/BCB/clone.csv` | 270,000 clone pairs: id_a, id_b, coarse type, two similarity scores (no header) |
| `BCB_nonclone.csv` | `dataset/BCB/nonclone.csv` | 279,032 non-clone pairs (with header) |
| `T1.csv`, `T2.csv`, `ST3.csv`, `MT3.csv`, `WT3T4.csv` | `dataset/BCB/type-1.csv` … `type-5.csv` (in that order) | clone pairs per type band |

```bash
unzip id2sourcecode.zip -d dataset/BCB/id2sourcecode_extracted/
python tools/prepare_bcb_pairs.py        # -> dataset/BCB/pairs_mapped.csv (549,031 rows)
python tools/build_bcb_type_mapping.py   # -> pairs_with_types.csv, pairs_canonical.csv (548,901 pairs)
```

`type-3.csv` holds VST3 and ST3 (syntactic similarity ≥ 0.7), split at 0.9 by
`build_bcb_type_mapping.py`; `type-4.csv` is MT3 [0.5, 0.7) and `type-5.csv` is WT3/T4
(< 0.5). `prepare_bcb_pairs.py` reads `clone.csv` with a header row, so its first pair is
skipped; this reproduces the pair list behind the reported results exactly. Headline BCB
numbers are restricted to T1–T3; WT3/T4 is reported separately (Krinke & Ragkhitwetsagul,
IWSC 2022).

### Google Code Jam (GCJ)

The `googlejam4` corpus (1,665 Java files, 12 problems) of the DeepSim benchmark
(Zhao & Huang, ESEC/FSE 2018), also distributed with FA-AST
(<https://github.com/jacobwwh/graphmatch_clone>, `googlejam4_src.zip`). We use the flat
version with `package`/`import` lines removed that the Amain repository distributes as
`dataset/GCJ/googlejam4.zip` (byte-identical to ours). After removing those lines and
whitespace, 1,651 of the 1,665 files are identical to DeepSim's copy; of the remaining 14,
7 differ only in comments and 7 contain small code differences.

| Amain file | Place as | Content |
|---|---|---|
| `googlejam4.zip` | unzip into `dataset/GCJ/` | `dataset/GCJ/googlejam4/*.java` |
| `GCJ_clone.csv` | `dataset/GCJ/GCJ_clone.csv` | 274,959 same-problem pairs (identical to Amain's modulo line endings) |

The non-clone list `dataset/GCJ/GCJ_noclone_270000.csv` (270,000 pairs of files from
different problems, no header) is not distributed by Amain, and we could not identify its
original source; its origin is undocumented.

```bash
unzip googlejam4.zip -d dataset/GCJ/
python tools/prepare_gcj_pairs.py        # -> dataset/GCJ/pairs_mapped.csv (544,206 pairs)
```

### CodeNet-Java250 and POJ-104

The CONCORD release (Ding et al., ISSTA 2023), Zenodo
<https://doi.org/10.5281/zenodo.8393793>, archive `finetune_data.zip`. Extract it so that

```
$FUSECC_EXTERNAL/concord_assets/data/codenet_cc/Java250_{train,valid,test}.jsonl
$FUSECC_EXTERNAL/concord_assets/data/poj104_cc/{train,valid,test}.jsonl
```

These are the official problem-disjoint splits (POJ-104 64/16/24 problems, Java250
125/62/63). POJ-104 is C/C++; it is processed with the same lexer plus an extended keyword
list (`--cpp-keywords`).

## Pipeline

| Stage | Script | Output |
|---|---|---|
| BCB pairs + clone types | `tools/prepare_bcb_pairs.py`, `tools/build_bcb_type_mapping.py` | `dataset/BCB/pairs_canonical.csv` |
| GCJ pairs | `tools/prepare_gcj_pairs.py` | `dataset/GCJ/pairs_mapped.csv` |
| Nineteen metrics, BCB / GCJ | `python experiments/run_experiment.py config/bcb_19_canonical.yaml` (or `config/gcj_19.yaml`) | `results/bcb_19_canonical/`, `results/gcj_19/` |
| Frozen-encoder embeddings | `tools/extract_code_embeddings_v2.py --model unixcoder` | `dataset/{GCJ,BCB}/embeddings_unixcoder.pkl` |
| GCJ 148-feature matrices | `tools/build_gcj_embed_features.py`, `tools/build_gcj_embed_features_ft.py` | `results/gcj_148{,_ft,_ft_pd}/features.npz` |
| Encoder fine-tuning (GCJ analysis) | `tools/finetune_unixcoder_gcj.py [--split problem]` | `models/unixcoder_gcj_ft{,_pd}/` |
| Java250 / POJ-104 pairs, metrics, embeddings | `tools/prep_concord_pairs.py`, `tools/run_concord_features.sh`, `tools/embed_concord.py` | `dataset/concord/<ds>/` |

Quick check on 1,000 BCB pairs: `python experiments/run_experiment.py config/bcb_19_smoke.yaml`.

## Paper tables and figures → scripts

| Paper item | Script(s) | Artefact |
|---|---|---|
| Table 1, similarity measures | method definition; implemented in `similarity_kernels.py`, `similarity_dispatcher.py` | — |
| Table 2, classifier families | `tools/run_rq1_full.sh` (`tools/rq1_classifier_comparison.py`) | `results/rq1_classifiers/<ds>/` |
| Table 3, ensemble designs | `tools/run_rq1_ensemble_after.sh` (`tools/rq1_ensemble.py`), `tools/d2_meta_comparison.py` | `results/rq1_classifiers/<ds>/ensemble_*`, `results/d2_meta_comparison/` |
| Table 4, baseline configurations | baseline harnesses: `baselines/` (patches, drivers) and the `tools/` drivers listed in [baselines/README.md](baselines/README.md) | — |
| Table 5, BigCloneBench per-type recall | `tools/train_bcb_e3.py`; baselines scored by `tools/eval_pairs_bcb.py`, `tools/eval_oreo_bcb.py`, `tools/eval_deckard_bcb.py`, `tools/convert_{scc,msccd,stone}_pairs.py`, `tools/archive_faast_bcb.py`, `tools/archive_magnet_bcb.py`, `tools/llm_baseline_deepseek.py`, `baselines/drivers/astnn/train_bcb_fusecc.py` | `results/bcb_e3/`, `results/oreo_bcb/`, `results/baselines_rerun/*/` |
| Table 6, Google Code Jam results | `tools/eval_gcj_ft_protocols.py`, `tools/eval_gcj_pd.py`, `tools/baseline_embed_only_gcj.py`, `tools/f4_wilcoxon.py`, `tools/eval_pairs_gcj.py`, `tools/eval_clusters_gcj.py`, `baselines/drivers/astnn/train_gcj_fusecc.py` | `results/gcj_148*/`, `results/gcj_baselines/`, `results/f4_wilcoxon/`, `results/baselines_rerun/*/` |
| Table 7, individual measures | `tools/rq2_individual_metrics.py --dataset {bcb,gcj}` | `results/rq2_metrics/` |
| Table 8, metric-family ablation | `tools/rq_ablation_groups.py` | `results/rq_ablation_groups/` |
| Table 9, metrics vs embeddings (CodeNet-Java250, POJ-104) | `tools/train_concord.py --dataset <ds>`, `tools/make_concord_figures.py` (bootstrap tests) | `results/concord/<ds>/summary.json`, `results/concord/figures/ablation_bootstrap.json` |
| Table 10, primary results across protocols | `tools/train_bcb_pairlevel_controlled.py`, `tools/train_bcb_function_disjoint.py`, `tools/bcb_leakage_isolation.py`, `tools/eval_gcj_ft_protocols.py`, `tools/eval_gcj_pd.py`, `tools/train_concord.py` | `results/bcb_leakage_isolation/`, `results/bcb_function_disjoint/`, `results/gcj_148*/`, `results/concord/<ds>/` |
| Table 11, MAP@R retrieval | `tools/run_concord_mapr.sh` (`tools/eval_concord_mapr.py`, `tools/mapr_rerank.py`), `tools/run_concord_baseline_encoders.sh`, `tools/run_concord_st_encoders.sh`, `tools/embed_concord_with_concord.py`, `tools/build_concord_comparison.py`, `tools/diagnose_mapr_ceiling.py` | `results/concord/<ds>/mapr.json`, `results/concord/comparison.*` |
| Table 12, stage-wise cost | `tools/rq5_efficiency.py` | `results/rq5_efficiency/` |
| Table 13, backbone substitution | `tools/run_jina_backbone_embed.sh`, `tools/train_concord.py --emb-tag jinacode_sym --suffix _jina`, `tools/run_jina_rerank.sh` | `results/concord/<ds>/summary_jina.json`, `results/concord/<ds>/mapr.json` |
| Table 14, related-work positioning | literature; no script | — |
| Figure 1, pipeline | diagram; no script | — |
| Figure 2, data at each stage | `tools/make_pipeline_stage_figure.py` | `manuscript_v2/Figures/fig_pipeline_stages.{pdf,png}` |
| Figure 3, threshold plateau | `tools/f2_threshold_sensitivity.py`, then `tools/make_threshold_plateau_figure.py` | `results/f2_threshold/fig_threshold_plateau.{pdf,png}` |
| Figure 4, BCB split protocols | `tools/make_split_protocol_figure.py` | `manuscript_v2/Figures/fig_split_protocols.{pdf,png}` |
| Figure 5, family importance | `tools/f1_analysis.py`, then `tools/make_experiment_figures.py` | `results/exp_figures/fig_family_importance.{pdf,png}` |
| Figure 6, confusion matrices | `tools/make_concord_figures.py` | `results/concord/figures/confusion_{java250,poj104}.{pdf,png}` |
| Figure 7, Spearman heatmaps | `tools/make_concord_figures.py` | `results/concord/figures/heatmap_{java250,poj104}.{pdf,png}` |
| Figure 8, accuracy against training time | `tools/make_experiment_figures.py` | `results/exp_figures/fig_pareto.{pdf,png}` |

[REPRODUCTION.md](REPRODUCTION.md) gives the protocol definitions and further detail.

## Baseline re-runs

`baselines/` contains what is needed to re-run the third-party detectors on the same pairs:
`baselines/patches/*.patch` (our modifications, applied to pinned upstream commits) and
`baselines/drivers/` (our own driver scripts and configurations). See
[baselines/README.md](baselines/README.md) for upstream sources, commits and licences.

## Repository layout

| Path | Purpose |
|---|---|
| `tokenizer_core.py`, `similarity_kernels.py`, `similarity_dispatcher.py` | tokenisation and the nineteen metric kernels |
| `experiments/`, `config/` | config-driven parallel metric extraction and classification |
| `tools/` | data preparation, embedding, training, evaluation, analysis and figure scripts |
| `baselines/` | baseline patches and drivers |
| `seeds.py` | seeding |

Generated outputs go to `results/<experiment>/` (one directory per experiment, each with a
`run_meta.json` provenance record), models to `models/`, and derived data to `dataset/`;
none of these are tracked. The earlier 16-metric configuration writes to
`results/real_fusecc/` (`config/real_fusecc.yaml`, `config/classify_bcb.yaml`) and
`results/gcj_fusecc/` (`config/gcj.yaml`).

## Licence

The FuseCC code in this repository is released under the MIT licence (see [LICENSE](LICENSE)).
The files in `baselines/patches/` modify third-party code, and the baseline drivers call it;
that code keeps its upstream licence: ASTNN MIT, SourcererCC and Oreo GPL-3.0, StoneDetector
EPL-2.0. FA-AST, MAGNET, MSCCD, CCStokener and TriFusion-LLM state no licence, so obtain them
from their upstream repositories under the terms their authors set. Details are in
[baselines/README.md](baselines/README.md).

## Citation

Muhammad Tahir Riaz, Kong Weiqiang, Raja Muhammad Yaqoob, Usama Ali Shaikh, Muhammad Abdul
Rafey, Venkat Harshith Balusu. "Fusing Unsupervised Similarity Metrics with Frozen Code
Embeddings for Efficient and Generalizable Code Clone Detection". Under review at the
*Journal of Systems and Software*.

