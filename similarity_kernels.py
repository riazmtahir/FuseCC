import math
import re
import difflib
import hashlib
import numpy as np
from collections import Counter
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
from pygments import lex
from pygments.lexers import JavaLexer

# Count common and total token occurrences
def _count_intersection_union(seq1, seq2):
    c1, c2 = Counter(seq1), Counter(seq2)
    intersection = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    return intersection, union

##### Basic similarity metrics

# Jaccard similarity using token frequency
def sim_jaccard(seq1, seq2):
    i, u = _count_intersection_union(seq1, seq2)
    return i / u if u else 0.0

# Dice similarity (overlap weighted)
def sim_dice(seq1, seq2):
    i, _ = _count_intersection_union(seq1, seq2)
    return (2 * i) / (len(seq1) + len(seq2)) if (seq1 or seq2) else 0.0

# Character-level Jaro similarity
def sim_jaro(seq1, seq2):
    return Levenshtein.jaro(seq1, seq2)

# Improved Jaro similarity with prefix boost
def sim_jaro_winkler(seq1, seq2):
    return Levenshtein.jaro_winkler(seq1, seq2)

##### Edit-distance based

# Raw Levenshtein edit distance
def sim_levenshtein_distance(seq1, seq2):
    return Levenshtein.distance(seq1, seq2)

# Normalized Levenshtein similarity
def sim_levenshtein_ratio(seq1, seq2):
    return Levenshtein.ratio(seq1, seq2)

##### Vector and fuzzy matching

# Cosine similarity using bag-of-words
def sim_bow_cosine(seq1, seq2):
    c1, c2 = Counter(seq1), Counter(seq2)
    vocab = set(c1) | set(c2)
    v1 = [c1.get(w, 0) for w in vocab]
    v2 = [c2.get(w, 0) for w in vocab]

    dot = sum(x * y for x, y in zip(v1, v2))
    mag = math.sqrt(sum(x*x for x in v1)) * math.sqrt(sum(y*y for y in v2))
    return dot / mag if mag else 0.0

# Fuzzy string matching similarity
def sim_fuzzy_match(seq1, seq2):
    return fuzz.ratio(" ".join(seq1), " ".join(seq2)) / 100.0

##### Structural similarity

# Longest Common Subsequence similarity
def sim_lcs(seq1, seq2):
    s1, s2 = " ".join(seq1), " ".join(seq2)
    lcs = difflib.SequenceMatcher(None, s1, s2)\
        .find_longest_match(0, len(s1), 0, len(s2)).size
    return (2 * lcs) / (len(s1) + len(s2)) if (s1 and s2) else 0.0

###### N-gram and TF-IDF

# Token-level N-gram similarity
def sim_ngram(seq1, seq2, n=15):
    def grams(seq):
        return set(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    g1, g2 = grams(seq1), grams(seq2)
    return len(g1 & g2) / len(g1 | g2) if (g1 | g2) else 0.0

# TF-IDF cosine similarity
def sim_tfidf(seq1, seq2):
    """
    TF-IDF cosine similarity robust to token lists and non-word tokens.
    Accepts either token lists or already-joined strings.
    Falls back to a token_pattern that treats any non-space chunk as a token.
    If TF-IDF still yields empty vocab, falls back to a bag-of-words cosine.
    """
    # Accept either lists of tokens or strings
    if isinstance(seq1, (list, tuple)):
        s1 = " ".join(seq1)
    else:
        s1 = str(seq1)

    if isinstance(seq2, (list, tuple)):
        s2 = " ".join(seq2)
    else:
        s2 = str(seq2)

    # Primary attempt: default vectorizer (works for normal code with word tokens)
    try:
        mat = TfidfVectorizer().fit_transform([s1, s2])
        # mat[0] @ mat[1].T is 1x1 sparse matrix; convert to scalar
        sim = (mat[0] @ mat[1].T).toarray()[0, 0]
        return float(sim)
    except ValueError:
        # Fallback: treat any non-space sequence as a token (keeps punctuation/symbols)
        try:
            mat = TfidfVectorizer(token_pattern=r"(?u)\S+").fit_transform([s1, s2])
            sim = (mat[0] @ mat[1].T).toarray()[0, 0]
            return float(sim)
        except ValueError:
            # Last fallback: compute simple bag-of-words cosine using counters
            from collections import Counter
            c1, c2 = Counter(s1.split()), Counter(s2.split())
            vocab = list(set(c1) | set(c2))
            v1 = [c1.get(w, 0) for w in vocab]
            v2 = [c2.get(w, 0) for w in vocab]
            dot = sum(x * y for x, y in zip(v1, v2))
            mag = (sum(x * x for x in v1) ** 0.5) * (sum(y * y for y in v2) ** 0.5)
            return dot / mag if mag else 0.0

##### AST-inspired and hashing

# Approximate AST similarity using Java lexer tokens
def sim_ast(seq1, seq2):
    def lex_tokens(code):
        return Counter(tok for tok, _ in lex(code, JavaLexer()))

    t1, t2 = lex_tokens(" ".join(seq1)), lex_tokens(" ".join(seq2))
    inter = sum((t1 & t2).values())
    total = sum(t1.values()) + sum(t2.values())
    return (2 * inter / total) if total else 0.0

# Rolling hash based similarity
def sim_rolling_hash(seq1, seq2, min_len=10):
    def rolling(text):
        h = 0
        for ch in text:
            h = (h * 256 + ord(ch)) % (10**9 + 7)
        return h

    s1, s2 = " ".join(seq1), " ".join(seq2)
    hashes = {rolling(s1[i:i+min_len]) for i in range(len(s1)-min_len+1)}
    matches = sum(rolling(s2[i:i+min_len]) in hashes
                  for i in range(len(s2)-min_len+1))
    return matches / (len(seq1) + len(seq2)) if (seq1 or seq2) else 0.0


##### Advanced clone detection

# Greedy string tiling similarity
def sim_greedy_tiling(seq1, seq2):
    score = 0
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            k = 0
            while i+k < len(seq1) and j+k < len(seq2) and seq1[i+k] == seq2[j+k]:
                k += 1
            score += k
    return (2 * score) / (len(seq1) + len(seq2)) if (seq1 or seq2) else 0.0

# Semantic similarity using sequence matcher
def sim_semantic_clone(seq1, seq2):
    return difflib.SequenceMatcher(
        None, " ".join(seq1), " ".join(seq2)
    ).ratio()

# Stable 64-bit hash: built-in hash() of a str is randomised per process, which made t16 non-reproducible.
def _stable_hash(key):
    return int.from_bytes(hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest(), 'big')

# Winnowing fingerprint similarity
def sim_winnow(seq1, seq2, k=5, w=10):
    def winnow(text):
        grams = [text[i:i+k] for i in range(len(text)-k+1)]
        hashes = [_stable_hash(g) for g in grams]
        return {
            min(hashes[i:i+w])
            for i in range(len(grams)-w+1)
        }

    f1, f2 = winnow(" ".join(seq1)), winnow(" ".join(seq2))
    return len(f1 & f2) / min(len(f1), len(f2)) if (f1 and f2) else 0.0

# ========== NEW SEMANTIC FEATURES ==========

def sim_identifier_abstraction(seq1, seq2):
    """
    Replace identifiers with placeholders (VAR1, VAR2, ...) and compute Jaccard.
    seq1, seq2 are token lists (strings).
    """
    import re
    from collections import Counter
    keywords = {'abstract','assert','boolean','break','byte','case','catch','char','class','const','continue','default','do','double','else','enum','extends','final','finally','float','for','goto','if','implements','import','instanceof','int','interface','long','native','new','package','private','protected','public','return','short','static','strictfp','super','switch','synchronized','this','throw','throws','transient','try','void','volatile','while'}
    
    def abstract(tokens):
        id_map = {}
        next_id = 1
        out = []
        for t in tokens:
            if t not in keywords and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', t):
                if t not in id_map:
                    id_map[t] = f'VAR{next_id}'
                    next_id += 1
                out.append(id_map[t])
            else:
                out.append(t)
        return out
    
    a1 = abstract(seq1)
    a2 = abstract(seq2)
    c1, c2 = Counter(a1), Counter(a2)
    inter = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    return inter / union if union else 0.0

def sim_ast_normalization(seq1, seq2):
    """
    Count structural patterns: for, while, if, switch, try, etc.
    Return cosine similarity of count vectors.
    """
    patterns = ['for', 'while', 'if', 'switch', 'try', 'catch', 'do', 'return']
    def count_patterns(tokens):
        cnt = {p: 0 for p in patterns}
        for t in tokens:
            if t in cnt:
                cnt[t] += 1
        return list(cnt.values())
    
    v1 = count_patterns(seq1)
    v2 = count_patterns(seq2)
    dot = sum(a*b for a,b in zip(v1,v2))
    norm1 = sum(a*a for a in v1)**0.5
    norm2 = sum(b*b for b in v2)**0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def sim_control_flow_stats(seq1, seq2):
    """
    Compute loop_count, branch_count, nesting_depth, call_count.
    Return a similarity score based on absolute differences normalized.
    """
    def stats(tokens):
        loop_count = sum(1 for t in tokens if t in ('for','while','do'))
        branch_count = sum(1 for t in tokens if t in ('if','switch'))
        depth = 0
        max_depth = 0
        for t in tokens:
            if t == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif t == '}':
                depth -= 1
        call_count = sum(1 for t in tokens if t == '.')  # crude approximation
        return loop_count, branch_count, max_depth, call_count
    
    s1 = stats(seq1)
    s2 = stats(seq2)
    diffs = [abs(a-b) / (max(a,b)+1e-6) for a,b in zip(s1,s2)]
    sim = 1 - sum(diffs) / len(diffs)
    return max(0, min(1, sim))