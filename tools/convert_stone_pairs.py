"""Convert StoneDetector output to BCB function-id pairs.

StoneDetector emits `dir,file,startline,endline,dir,file,startline,endline`
per clone pair; our BCB corpus is one function per file named <id>.java, so
the id pair is just the two basenames. Deduplicates unordered pairs.

Usage: python tools/convert_stone_pairs.py <pairs_raw.csv> <out.csv>
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
pairs = set()
bad = 0
with open(src, errors="ignore") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        def to_key(name):
            try:
                return int(name.replace(".java", ""))
            except ValueError:
                return name  # non-integer corpus (e.g. GCJ) -> keep basename
        a, b = to_key(parts[1]), to_key(parts[5])
        if a != b:
            pairs.add(tuple(sorted((a, b), key=str)))
with open(dst, "w") as f:
    for a, b in sorted(pairs, key=str):
        f.write(f"{a},{b}\n")
print(f"{len(pairs)} unique id pairs written to {dst} ({bad} unparseable lines)")
