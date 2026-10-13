from tokenizer_core import extract_lexical_stream
import similarity_kernels as sk
from collections import Counter
import os
import hashlib
import pickle
import gzip
from pathlib import Path

# Map tool IDs to similarity functions
SIM_MAP = {
    "t1": sk.sim_jaccard,
    "t2": sk.sim_dice,
    "t3": sk.sim_jaro,
    "t4": sk.sim_jaro_winkler,
    "t5": sk.sim_levenshtein_distance,
    "t6": sk.sim_levenshtein_ratio,
    "t7": sk.sim_bow_cosine,
    "t8": sk.sim_fuzzy_match,
    "t9": sk.sim_lcs,
    "t10": sk.sim_ngram,
    "t11": sk.sim_tfidf,
    "t12": sk.sim_ast,
    "t13": sk.sim_rolling_hash,
    "t14": sk.sim_greedy_tiling,
    "t15": sk.sim_semantic_clone,
    "t16": sk.sim_winnow
}

SIM_MAP.update({
    "t17": sk.sim_identifier_abstraction,
    "t18": sk.sim_ast_normalization,
    "t19": sk.sim_control_flow_stats,
})

# Define which metric IDs expect which input type
# "tokens" -> list of tokens
# "string" -> joined string
# "counter" -> Counter object
INPUT_TYPE = {
    # token/counter based
    "t1": "tokens",
    "t2": "tokens",
    "t7": "tokens",
    "t10": "tokens",
    "t12": "tokens",
    "t13": "tokens",
    "t14": "tokens",
    "t16": "tokens",
    # string-based
    "t3": "string",
    "t4": "string",
    "t5": "string",
    "t6": "string",
    "t8": "string",
    "t9": "string",
    "t11": "string",
    "t15": "string",
}

# After the existing INPUT_TYPE definitions, add:
INPUT_TYPE.update({
    "t17": "tokens",
    "t18": "tokens",
    "t19": "tokens",
})

# Normalize inputs into commonly-used representations
def normalize_inputs(tokens_a, tokens_b):
    s_a = " ".join(tokens_a)
    s_b = " ".join(tokens_b)
    c_a = Counter(tokens_a)
    c_b = Counter(tokens_b)
    return {"tokens": (tokens_a, tokens_b), "string": (s_a, s_b), "counter": (c_a, c_b)}


# Token cache helpers (opt-in via TOKEN_CACHE_DIR env var)
def _get_cache_dir():
    d = os.environ.get('TOKEN_CACHE_DIR')
    if not d:
        return None
    p = Path(d)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _cache_path_for_file(cache_dir: Path, filepath: str) -> Path:
    # create a filename-safe hash for the source path
    h = hashlib.sha1(filepath.encode('utf-8')).hexdigest()
    return cache_dir / f"{h}.pkl.gz"


def _load_tokens_from_cache(cache_dir: Path, filepath: str):
    path = _cache_path_for_file(cache_dir, filepath)
    if not path.exists():
        return None
    try:
        with gzip.open(path, 'rb') as fh:
            tokens = pickle.load(fh)
        return tokens
    except Exception:
        return None


def _save_tokens_to_cache(cache_dir: Path, filepath: str, tokens):
    path = _cache_path_for_file(cache_dir, filepath)
    try:
        with gzip.open(path, 'wb') as fh:
            pickle.dump(tokens, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # ignore cache write errors
        pass


# Load files, tokenize (with optional caching), and apply selected similarity
def run_similarity(metric_id, file_a, file_b):
    cache_dir = _get_cache_dir()

    # tokens for file_a
    tokens_a = None
    if cache_dir is not None:
        tokens_a = _load_tokens_from_cache(cache_dir, file_a)
    if tokens_a is None:
        tokens_a = extract_lexical_stream(file_a)
        # ensure tokens are list
        if tokens_a is None:
            tokens_a = []
        if cache_dir is not None:
            _save_tokens_to_cache(cache_dir, file_a, tokens_a)

    # tokens for file_b
    tokens_b = None
    if cache_dir is not None:
        tokens_b = _load_tokens_from_cache(cache_dir, file_b)
    if tokens_b is None:
        tokens_b = extract_lexical_stream(file_b)
        if tokens_b is None:
            tokens_b = []
        if cache_dir is not None:
            _save_tokens_to_cache(cache_dir, file_b, tokens_b)

    norm = normalize_inputs(tokens_a, tokens_b)
    func = SIM_MAP.get(metric_id)
    if func is None:
        raise KeyError(f"Unknown metric id: {metric_id}")
    typ = INPUT_TYPE.get(metric_id, "tokens")
    a, b = norm[typ]
    return func(a, b)
