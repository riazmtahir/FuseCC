"""Convert MSCCD detection output to BCB function-id pairs.

MSCCD's pairs.file lines are `[[projectId, fileId, bagId], [projectId, fileId,
bagId]]`; fileList.txt maps (projectId, per-project line order) -> path. Our
BCB corpus is one function per file named <id>.java, so any bag-level match
between two files counts as that id pair (bags at finer granularity belong to
the same function). Deduplicates unordered pairs.

Usage: python tools/convert_msccd_pairs.py <taskDir> <detectionSubdir> <out.csv>
e.g.:  python tools/convert_msccd_pairs.py tasks/task2 detection1 out.csv
"""
import json
import os
import sys

task_dir, det, dst = sys.argv[1], sys.argv[2], sys.argv[3]

files = {}
counters = {}
with open(os.path.join(task_dir, "fileList.txt")) as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        fid = counters.get(pid, 0)
        counters[pid] = fid + 1
        files[(pid, fid)] = parts[1]

def to_key(path):
    """BCB files are <intId>.java -> int id; otherwise keep the basename."""
    base = os.path.basename(path)
    try:
        return int(base.replace(".java", ""))
    except ValueError:
        return base


pairs = set()
with open(os.path.join(task_dir, det, "pairs.file")) as f:
    for line in f:
        a, b = json.loads(line.strip())
        ida = to_key(files[(a[0], a[1])])
        idb = to_key(files[(b[0], b[1])])
        if ida != idb:
            pairs.add(tuple(sorted((ida, idb), key=str)))
with open(dst, "w") as f:
    for a, b in sorted(pairs, key=str):
        f.write(f"{a},{b}\n")
print(f"{len(pairs)} unique id pairs written to {dst}")
