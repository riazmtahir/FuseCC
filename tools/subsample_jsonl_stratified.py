#!/usr/bin/env python
"""Label-stratified subsample of a JSONL file (streaming, two-pass).

Used to cap the TriFusion-LLM test_real split at 100k pairs (GPU/API budget,
mirrors the MAGNET 100k train cap). Proportional
allocation per label via largest remainder; selected lines keep input order.
"""
import argparse
import json
import random
from collections import Counter, defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--total", type=int, default=100000)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def get_label(line):
    item = json.loads(line)
    if "label" in item:
        return int(item["label"])
    return int(item.get("original_data", {}).get("label", 0))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    labels = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labels.append(get_label(line))
    n = len(labels)
    counts = Counter(labels)
    print(f"input: {n} samples, per-label: {dict(sorted(counts.items()))}")

    if n <= args.total:
        print("input <= total; copying as-is")
        with open(args.input) as fi, open(args.output, "w") as fo:
            fo.writelines(l for l in fi if l.strip())
        return

    # proportional allocation, largest remainder
    quotas = {}
    remainders = []
    assigned = 0
    for lbl, c in counts.items():
        exact = args.total * c / n
        q = int(exact)
        quotas[lbl] = q
        assigned += q
        remainders.append((exact - q, lbl))
    remainders.sort(reverse=True)
    for _, lbl in remainders[: args.total - assigned]:
        quotas[lbl] += 1

    by_label = defaultdict(list)
    for idx, lbl in enumerate(labels):
        by_label[lbl].append(idx)

    selected = set()
    for lbl, idxs in by_label.items():
        k = min(quotas[lbl], len(idxs))
        selected.update(rng.sample(idxs, k))

    written = Counter()
    with open(args.input, "r", encoding="utf-8") as fi, \
         open(args.output, "w", encoding="utf-8") as fo:
        i = 0
        for line in fi:
            if not line.strip():
                continue
            if i in selected:
                fo.write(line)
                written[labels[i]] += 1
            i += 1

    print(f"output: {sum(written.values())} samples, per-label: {dict(sorted(written.items()))}")


if __name__ == "__main__":
    main()
