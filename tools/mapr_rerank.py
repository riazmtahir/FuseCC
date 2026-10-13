#!/usr/bin/env python3
"""Parallel batched 148-feature scorer for the MAP@R two-stage re-ranker.

The re-rank bottleneck is the 19-metric similarity scorer — t14 (greedy string tiling)
alone is ~33 ms/pair, ~57 ms/pair total, i.e. 18 pairs/s single-threaded. MAP@R needs
hundreds of queries × hundreds of candidates, so this splits the flat (query, candidate)
pair list across processes. Embeddings are shared read-only via a per-worker global so
they are not re-pickled per chunk.

Feature layout matches tools/train_concord.py exactly:
    0..18   t1..t19   |   19 cosine(centered)   |   20..83 |a-b| PCA64   |   84..147 a*b PCA64
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, _REPO_ROOT)
from similarity_dispatcher import SIM_MAP, INPUT_TYPE  # noqa: E402

METRICS = [f"t{i}" for i in range(1, 20)]

_TOKS = _ECN = _PC = None      # per-worker globals


def _init(toks, ecn, pc):
    global _TOKS, _ECN, _PC
    _TOKS, _ECN, _PC = toks, ecn, pc


def _score_chunk(pairs):
    out = np.empty((len(pairs), 148), dtype=np.float32)
    for r, (qi, cj) in enumerate(pairs):
        ta, tb = _TOKS[qi], _TOKS[cj]
        na = {"tokens": ta, "string": " ".join(ta), "counter": Counter(ta)}
        nb = {"tokens": tb, "string": " ".join(tb), "counter": Counter(tb)}
        for c, m in enumerate(METRICS):
            typ = INPUT_TYPE.get(m, "tokens")
            try:
                v = SIM_MAP[m](na[typ], nb[typ])
            except Exception:
                v = 0.0
            out[r, c] = 0.0 if v is None else float(v)
        out[r, 19] = float(_ECN[qi] @ _ECN[cj])
        out[r, 20:84] = np.abs(_PC[qi] - _PC[cj])
        out[r, 84:148] = _PC[qi] * _PC[cj]
    return out


def score_pairs_parallel(pairs, toks, ecn, pc, workers=8, chunk=400):
    chunks = [pairs[i:i + chunk] for i in range(0, len(pairs), chunk)]
    parts = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(toks, ecn, pc)) as ex:
        for part in ex.map(_score_chunk, chunks):
            parts.append(part)
    return np.vstack(parts)
