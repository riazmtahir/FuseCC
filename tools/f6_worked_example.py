"""Worked example of the feature extraction for the paper appendix.

Two ~10-line Java methods (same algorithm; renamed identifiers + a while->for
rewrite, i.e. Type-2/3 territory), tokenized by the real tokenizer_core, all 19
metrics computed by the real kernels, and step-by-step intermediates shown for
three representatives: t6 (edit-distance family), t13 (fingerprint family),
t17 (semantic-normalization family). Mirrors similarity_kernels.py exactly.

Output: results/f6_worked_example/{a.java,b.java,worked_example.md,vector.json}
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import re
import sys
from collections import Counter

import Levenshtein

REPO = _REPO_ROOT
sys.path.insert(0, REPO)
from similarity_dispatcher import SIM_MAP, INPUT_TYPE, normalize_inputs  # noqa
from tokenizer_core import extract_lexical_stream  # noqa

A_SRC = """public class A {
    public static int sumEven(int[] numbers) {
        int total = 0;
        int i = 0;
        while (i < numbers.length) {
            if (numbers[i] % 2 == 0) {
                total += numbers[i];
            }
            i++;
        }
        return total;
    }
}
"""

B_SRC = """public class B {
    public static int addEvenValues(int[] data) {
        int result = 0;
        for (int k = 0; k < data.length; k++) {
            if (data[k] % 2 == 0) {
                result = result + data[k];
            }
        }
        return result;
    }
}
"""

JAVA_KEYWORDS = {'abstract','assert','boolean','break','byte','case','catch',
    'char','class','const','continue','default','do','double','else','enum',
    'extends','final','finally','float','for','goto','if','implements',
    'import','instanceof','int','interface','long','native','new','package',
    'private','protected','public','return','short','static','strictfp',
    'super','switch','synchronized','this','throw','throws','transient','try',
    'void','volatile','while'}


def abstract_ids(tokens):
    id_map, out = {}, []
    for t in tokens:
        if t not in JAVA_KEYWORDS and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', t):
            id_map.setdefault(t, f"VAR{len(id_map) + 1}")
            out.append(id_map[t])
        else:
            out.append(t)
    return out, id_map


def main():
    out_dir = f"{REPO}/results/f6_worked_example"
    os.makedirs(out_dir, exist_ok=True)
    pa, pb = os.path.join(out_dir, "a.java"), os.path.join(out_dir, "b.java")
    open(pa, "w").write(A_SRC)
    open(pb, "w").write(B_SRC)

    ta, tb = extract_lexical_stream(pa), extract_lexical_stream(pb)
    norm = normalize_inputs(ta, tb)
    scores = {mid: float(SIM_MAP[mid](*norm[INPUT_TYPE.get(mid, "tokens")]))
              for mid in [f"t{i}" for i in range(1, 20)]}

    # --- t6 intermediates (Levenshtein ratio on joined token strings) ---
    sa, sb = norm["string"]
    dist = Levenshtein.distance(sa, sb)
    # --- t13 intermediates (rolling hash, window 10 chars) ---
    def rolling(text):
        h = 0
        for ch in text:
            h = (h * 256 + ord(ch)) % (10**9 + 7)
        return h
    H = {rolling(sa[i:i + 10]) for i in range(len(sa) - 9)}
    matches = sum(rolling(sb[i:i + 10]) in H for i in range(len(sb) - 9))
    # --- t17 intermediates (identifier abstraction -> Jaccard) ---
    aa, map_a = abstract_ids(ta)
    ab, map_b = abstract_ids(tb)
    ca, cb = Counter(aa), Counter(ab)
    inter, union = sum((ca & cb).values()), sum((ca | cb).values())

    md = f"""# Worked example: FuseCC feature extraction

Two Java methods implementing the same computation (sum of even array
elements); fragment B renames every identifier and rewrites the while-loop
as a for-loop — Type-2 renaming plus a Type-3 structural edit.

## 1. Tokenization (tokenizer_core, javalang, raw identifiers kept)

Fragment A -> {len(ta)} tokens:
`{' '.join(ta)}`

Fragment B -> {len(tb)} tokens:
`{' '.join(tb)}`

## 2. Three representative metric computations

### t6 - Levenshtein ratio (edit-distance family)
Token streams joined to strings of length |A|={len(sa)}, |B|={len(sb)};
edit distance = {dist}; ratio = 1 - 2d/(|A|+|B|) form used by
`Levenshtein.ratio` = **{scores['t6']:.4f}**.

### t13 - rolling-hash fingerprint (fingerprint family)
All 10-character windows of A hashed -> |H_A| = {len(H)} fingerprints;
windows of B matching H_A: {matches} of {len(sb) - 9};
score = matches / (|tokens_A|+|tokens_B|) = {matches}/{len(ta) + len(tb)}
= **{scores['t13']:.4f}**.

### t17 - identifier abstraction (semantic-normalization family, NEW)
Identifiers mapped to positional placeholders:
A: {json.dumps(map_a)}
B: {json.dumps(map_b)}
Abstracted multiset Jaccard = {inter}/{union} = **{scores['t17']:.4f}**
(vs raw-token Jaccard t1 = {scores['t1']:.4f} - abstraction recovers the
renamed-identifier similarity).

## 3. Resulting feature vector (t1..t19)

| metric | value |
|---|---|
""" + "\n".join(f"| t{i} | {scores[f't{i}']:.4f} |" for i in range(1, 20)) + """

The classifier consumes this 19-dim vector (plus, in the full system, the
UniXcoder cosine and 128 PCA interaction features).
"""
    open(os.path.join(out_dir, "worked_example.md"), "w").write(md)
    json.dump(scores, open(os.path.join(out_dir, "vector.json"), "w"), indent=2)
    print(md)


if __name__ == "__main__":
    main()
