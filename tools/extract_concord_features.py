#!/usr/bin/env python3
"""Compute the t1-t19 similarity features for ConCord POJ-104 / Java250 pair sets.

Reuses `similarity_kernels` + the dispatcher's SIM_MAP/INPUT_TYPE tables directly, so
the feature DEFINITIONS are byte-identical to the BCB/GCJ runs — that comparability is
the whole point, and it is why this does not reimplement any metric.

It does bypass `similarity_dispatcher.run_similarity`, which takes file PATHS and
re-reads/re-tokenises per pair. Here every sample is tokenised exactly once into memory
and pairs are scored from those cached token lists, which matters at this scale (a file
appears in many pairs) and avoids writing ~127k temp files.

LANGUAGE NOTE (POJ-104 is C/C++, not Java)
javalang's LEXER handles C/C++ fine — only its PARSER is Java-specific, and no metric
here parses. t1-t16 are lexical/fingerprint measures over token streams; t18 counts
`for/while/if/switch/try/catch/do/return`, all of which are C++ keywords too; t19 counts
loops/branches/brace-nesting/`.`. The single language-sensitive metric is t17, whose
identifier-abstraction keyword set is Java's: on C/C++ the keywords absent from that set
(`cin`, `cout`, `include`, `using`, `namespace`, `struct`, `bool`, `template`, ...) are
treated as identifiers and abstracted away. `--cpp-keywords` extends the set for C/C++;
the default is OFF so the Java runs stay bit-identical to BCB/GCJ. Whichever is used is
recorded in the output meta.

Usage:
    python tools/extract_concord_features.py --dataset java250
    python tools/extract_concord_features.py --dataset poj104 --cpp-keywords
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
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, _REPO_ROOT)
import similarity_kernels as sk                      # noqa: E402
from similarity_dispatcher import SIM_MAP, INPUT_TYPE  # noqa: E402
from tokenizer_core import _simple_tokenize_text      # noqa: E402

DATA_ROOT = _REPO_ROOT + "/dataset/concord"
METRICS = [f"t{i}" for i in range(1, 20)]

# C/C++ keywords absent from similarity_kernels' Java set, used only with --cpp-keywords
CPP_EXTRA = {
    'auto', 'bool', 'cin', 'cout', 'cerr', 'const', 'constexpr', 'delete', 'endl',
    'explicit', 'export', 'extern', 'friend', 'include', 'inline', 'mutable',
    'namespace', 'nullptr', 'operator', 'printf', 'register', 'scanf', 'signed',
    'sizeof', 'std', 'struct', 'template', 'typedef', 'typeid', 'typename', 'union',
    'unsigned', 'using', 'virtual', 'wchar_t',
}

_TOKENS = None          # per-worker global, set by _init


def tokenize(src):
    """javalang lexer with the repo's regex fallback — same path as extract_lexical_stream."""
    try:
        import javalang
        return [t.value for t in javalang.tokenizer.tokenize(src)]
    except Exception:
        return _simple_tokenize_text(src)


def _init(tokens, cpp_keywords):
    global _TOKENS
    _TOKENS = tokens
    if cpp_keywords:
        # sim_identifier_abstraction builds its keyword set locally per call, so the
        # only way to extend it without touching shared code is to wrap the kernel.
        orig = sk.sim_identifier_abstraction

        def patched(a, b, _orig=orig):
            fa = [t for t in a if t not in CPP_EXTRA]
            fb = [t for t in b if t not in CPP_EXTRA]
            return _orig(fa, fb)
        sk.sim_identifier_abstraction = patched
        SIM_MAP["t17"] = patched


def _score_chunk(chunk):
    out = np.empty((len(chunk), len(METRICS)), dtype=np.float32)
    for r, (ia, ib) in enumerate(chunk):
        ta, tb = _TOKENS[ia], _TOKENS[ib]
        norm = {"tokens": (ta, tb),
                "string": (" ".join(ta), " ".join(tb)),
                "counter": (Counter(ta), Counter(tb))}
        for c, m in enumerate(METRICS):
            a, b = norm[INPUT_TYPE.get(m, "tokens")]
            try:
                v = SIM_MAP[m](a, b)
            except Exception:
                v = 0.0
            out[r, c] = 0.0 if v is None else float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cpp-keywords", action="store_true",
                    help="extend t17's keyword set for C/C++ (use for poj104)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    d = os.path.join(DATA_ROOT, args.dataset)
    with open(os.path.join(d, "code_store.pkl"), "rb") as fh:
        store = pickle.load(fh)
    with open(os.path.join(d, "pairs.pkl"), "rb") as fh:
        pairs = pickle.load(fh)

    for split in ("train", "valid", "test"):
        rows = store[split]
        t0 = time.time()
        tokens = [tokenize(c) for _i, _l, c in rows]
        empty = sum(1 for t in tokens if not t)
        print(f"[{split}] tokenised {len(tokens):,} samples in {time.time()-t0:.1f}s "
              f"(mean {np.mean([len(t) for t in tokens]):.0f} tokens, {empty} empty)")

        pr = pairs[split]
        idx = [(a, b) for a, b, _y in pr]
        y = np.array([lab for _a, _b, lab in pr], dtype=np.int8)
        chunks = [idx[i:i + args.chunk] for i in range(0, len(idx), args.chunk)]

        t0 = time.time()
        parts = []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init,
                                 initargs=(tokens, args.cpp_keywords)) as ex:
            for k, part in enumerate(ex.map(_score_chunk, chunks), 1):
                parts.append(part)
                if k % 50 == 0 or k == len(chunks):
                    done = min(k * args.chunk, len(idx))
                    el = time.time() - t0
                    print(f"  [{split}] {done:,}/{len(idx):,} pairs  "
                          f"{el:.0f}s  ({done/max(el,1e-9):.0f} pairs/s)", flush=True)
        X = np.vstack(parts)
        np.savez_compressed(os.path.join(d, f"features_{split}.npz"),
                            X=X, y=y, pairs=np.array(idx, dtype=np.int32),
                            feature_names=np.array(METRICS))
        print(f"[{split}] X={X.shape} -> features_{split}.npz  ({time.time()-t0:.0f}s)")

    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path))
    meta["features"] = {
        "metrics": METRICS,
        "kernels": "similarity_kernels via similarity_dispatcher SIM_MAP/INPUT_TYPE "
                   "(identical definitions to the BCB/GCJ runs)",
        "tokenizer": "javalang lexer with regex fallback (the lexer is not Java-specific; "
                     "only javalang's PARSER is, and no metric parses)",
        "t17_cpp_keywords_extension": bool(args.cpp_keywords),
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    print("updated", meta_path)


if __name__ == "__main__":
    main()
