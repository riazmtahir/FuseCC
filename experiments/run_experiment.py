#!/usr/bin/env python3
"""
Full experiment runner for FuseCC
- Reads YAML config specifying dataset (CSV of pairs) and metrics to compute
- Computes similarity scores using similarity_dispatcher.run_similarity
- Supports smoke/sample mode, parallel workers, deterministic seeds
- Supports optional streaming/batched result writes to avoid holding all results in memory
- Writes metadata (config_used.yaml, git_sha.txt, pip_freeze.txt) and results/results.csv

Config fields (example in config/real_fusecc.yaml):
  exp_id: real_fusecc
  seed: 12345
  data_path: dataset/BCB/pairs_mapped.csv            # CSV with columns: id_a,id_b,path_a,path_b,label
  sample: 1000                        # optional, integer for smoke sampling
  metrics: ["t1","t2",...]
  num_workers: 4
  batch_size: 64
  output_dir: results/real_fusecc
  stream_results: true                # optional, enable streaming append every batch

Usage:
  python experiments/run_experiment.py config/real_fusecc.yaml

The runner will create the output_dir (relative to repo root) and write:
  - config_used.yaml
  - git_sha.txt
  - pip_freeze.txt
  - results.csv (one row per pair, scores as columns)
  - run.log

Streaming mode behavior:
- If stream_results is true, the runner will process pairs in batches of batch_size and append each batch to results.csv.
- If results.csv already exists, the runner will attempt to resume by skipping pairs whose 'id' is already present in the existing file.

"""

import os
import sys
import yaml
import time
import csv
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

# Local imports (must be importable from repo root)
import similarity_dispatcher as sd


def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)


def write_metadata(outdir: Path, cfg: Dict[str, Any]):
    outdir.mkdir(parents=True, exist_ok=True)
    # save config used
    with open(outdir / 'config_used.yaml', 'w') as fh:
        yaml.safe_dump(cfg, fh)
    # pip freeze
    try:
        with open(outdir / 'pip_freeze.txt', 'w') as fh:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=fh, check=False)
    except Exception as e:
        with open(outdir / 'pip_freeze.txt', 'w') as fh:
            fh.write(f'pip freeze failed: {e}\n')
    # git sha
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        sha = 'git sha not available'
    with open(outdir / 'git_sha.txt', 'w') as fh:
        fh.write(sha + '\n')


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def prepare_dataframe(data_path: str, sample: int = None) -> pd.DataFrame:
    """Load CSV of pairs. Expect columns: id (optional), path_a, path_b, label (optional).
    If sample is provided and smaller than dataset, take sample (random, reproducible handled externally by seed).
    """
    df = pd.read_csv(data_path)
    # Accept several common column schemes; normalize to id,path_a,path_b,label
    if {'path_a', 'path_b'}.issubset(df.columns):
        # ok
        pass
    elif {'id_a', 'id_b'}.issubset(df.columns):
        # map id_a,id_b -> path_a,path_b if path columns not present later
        # but prefer path_a/path_b; caller should provide pairs with paths
        pass
    else:
        raise ValueError(f"Input CSV must contain columns: path_a, path_b (or id_a,id_b mapped later). Found: {df.columns.tolist()}")

    if sample is not None and sample > 0 and sample < len(df):
        df = df.sample(n=sample, random_state=int(os.environ.get('PYTHONHASHSEED', 0)))
    df = df.reset_index(drop=True)
    # ensure id column
    if 'id' not in df.columns:
        df.insert(0, 'id', df.index.astype(str))
    return df


def compute_scores_for_row(row: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
    """Compute metrics for a single row. row must contain 'path_a' and 'path_b'.
    Returns a dict with score_<metric_id> entries. Exceptions are caught and logged; score is None on error.
    """
    result = {}
    a = row.get('path_a') or row.get('pathA') or row.get('path1')
    b = row.get('path_b') or row.get('pathB') or row.get('path2')
    for m in metrics:
        try:
            score = sd.run_similarity(m, a, b)
        except Exception as e:
            logging.exception(f"Metric {m} failed for {a} vs {b}: {e}")
            score = None
        result[f'score_{m}'] = score
    return result


def worker_init(seed: int):
    # runs inside worker process
    set_global_seed(seed)


def append_batch_to_csv(out_csv: Path, rows: List[Dict[str, Any]]):
    df_batch = pd.DataFrame(rows)
    header = not out_csv.exists()
    df_batch.to_csv(out_csv, mode='a', index=False, header=header)


def run(cfg_path: str):
    cfg = load_config(cfg_path)
    exp_id = cfg.get('exp_id', 'exp')
    seed = int(cfg.get('seed', 12345))
    data_path = cfg.get('data_path')
    sample = cfg.get('sample', None)
    metrics = cfg.get('metrics', [])
    num_workers = int(cfg.get('num_workers', 1) or 1)
    batch_size = int(cfg.get('batch_size', 64) or 64)
    output_dir = Path(cfg.get('output_dir', f'results/{exp_id}'))
    log_file = output_dir / 'run.log'
    stream_results = bool(cfg.get('stream_results', False))

    # prepare output and logging
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logging.info('Starting experiment %s', exp_id)
    write_metadata(output_dir, cfg)

    if data_path is None:
        logging.error('No data_path specified in config')
        sys.exit(1)

    set_global_seed(seed)

    # load pairs
    df = prepare_dataframe(data_path, sample)
    logging.info('Loaded %d pairs', len(df))

    out_csv = output_dir / 'results.csv'

    # Resume support: if streaming and out_csv exists, skip already processed ids
    if stream_results and out_csv.exists():
        try:
            existing = pd.read_csv(out_csv, usecols=['id'])
            existing_ids = set(existing['id'].astype(str).tolist())
            before = len(df)
            df = df.loc[~df['id'].astype(str).isin(existing_ids)].reset_index(drop=True)
            logging.info('Resuming: skipped %d already-processed pairs, %d remaining', before - len(df), len(df))
        except Exception:
            logging.exception('Failed to read existing results for resume; will overwrite if file exists')

    # If not streaming, accumulate all results then write
    if not stream_results:
        logging.info('Running without streaming (collecting all %d results in memory)', len(df))
        results = []
        if num_workers <= 1:
            logging.info('Running sequentially')
            for _, r in df.iterrows():
                base = {k: r.get(k) for k in ['id', 'path_a', 'path_b'] if k in r}
                try:
                    scores = compute_scores_for_row(r, metrics)
                except Exception as e:
                    logging.exception('Failed computing scores for row %s: %s', r, e)
                    scores = {f'score_{m}': None for m in metrics}
                base.update(scores)
                if 'label' in r.index:
                    base['label'] = r['label']
                results.append(base)
        else:
            logging.info('Running with %d workers', num_workers)
            with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init, initargs=(seed,)) as ex:
                futures = {ex.submit(compute_scores_for_row, r[1].to_dict(), metrics): r for r in df.iterrows()}
                for fut in as_completed(futures):
                    row = futures[fut]
                    try:
                        scores = fut.result()
                    except Exception as e:
                        logging.exception('Worker failed for row %s: %s', row, e)
                        scores = {f'score_{m}': None for m in metrics}
                    base = {k: row[1].get(k) for k in ['id', 'path_a', 'path_b'] if k in row[1]}
                    base.update(scores)
                    if 'label' in row[1].index:
                        base['label'] = row[1]['label']
                    results.append(base)
        # write once
        if results:
            out_df = pd.DataFrame(results)
            out_df.to_csv(out_csv, index=False)
            logging.info('Wrote results to %s (%d rows)', out_csv, len(out_df))
        else:
            logging.warning('No results to write')

    else:
        # streaming mode: process in batches and append
        logging.info('Running in streaming mode with batch_size=%d and %d workers', batch_size, num_workers)
        total = len(df)
        processed = 0
        # create executor outside loop for reuse
        executor = None
        if num_workers > 1:
            executor = ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init, initargs=(seed,))
        try:
            for start in range(0, total, batch_size):
                batch = df.iloc[start:start + batch_size]
                rows_out = []
                if num_workers <= 1:
                    for _, r in batch.iterrows():
                        base = {k: r.get(k) for k in ['id', 'path_a', 'path_b'] if k in r}
                        try:
                            scores = compute_scores_for_row(r, metrics)
                        except Exception as e:
                            logging.exception('Failed computing scores for row %s: %s', r, e)
                            scores = {f'score_{m}': None for m in metrics}
                        base.update(scores)
                        if 'label' in r.index:
                            base['label'] = r['label']
                        rows_out.append(base)
                else:
                    futures = {executor.submit(compute_scores_for_row, r[1].to_dict(), metrics): r for r in batch.iterrows()}
                    for fut in as_completed(futures):
                        row = futures[fut]
                        try:
                            scores = fut.result()
                        except Exception as e:
                            logging.exception('Worker failed for row %s: %s', row, e)
                            scores = {f'score_{m}': None for m in metrics}
                        base = {k: row[1].get(k) for k in ['id', 'path_a', 'path_b'] if k in row[1]}
                        base.update(scores)
                        if 'label' in row[1].index:
                            base['label'] = row[1]['label']
                        rows_out.append(base)

                # append batch to CSV
                if rows_out:
                    append_batch_to_csv(out_csv, rows_out)
                processed += len(batch)
                logging.info('Processed %d/%d pairs (%.2f%%)', processed, total, processed * 100.0 / total)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        logging.info('Streaming run finished; output at %s', out_csv)

    logging.info('Experiment finished')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python experiments/run_experiment.py config/real_fusecc.yaml')
        sys.exit(1)
    run(sys.argv[1])
