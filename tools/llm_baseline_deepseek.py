#!/usr/bin/env python
"""LLM zero-shot clone-detection baseline via the DeepSeek API.

Fills the "GPT-4 / LLM baseline missing from GCJ (and BCB) tables" gap with a
re-run LLM row in our unified environment: deepseek-chat (V3.2), zero-shot
binary clone judgment, temperature 0. GPT-4 itself stays a justified exclusion
(cost/reproducibility); this row is labelled "DeepSeek-V3.2 zero-shot (re-run)".

Protocols (seed 12345, stratified):
  BCB: --per-type N pairs per clone type (T1,T2,VST3,ST3,MT3,WT3-T4) + 4N
       non-clones -> per-type recall + labelled precision (F4 BCB convention).
  GCJ: N clone + N non-clone pairs -> P/R/F1/Acc (zero-shot row, no training).

Cost control: functions truncated at --max-chars; token usage + CNY estimate
recorded in summary.json.
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import concurrent.futures as cf
import csv
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict

import requests

REPO = _REPO_ROOT
BCB_PAIRS = f"{REPO}/dataset/BCB/pairs_with_types.csv"
GCJ_PAIRS = f"{REPO}/dataset/GCJ/pairs_mapped.csv"
BCB_TYPES = ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]
# deepseek-chat V3.2 standard (off-peak) prices, CNY per 1M tokens
PRICE_IN, PRICE_OUT = 2.0, 3.0

SYSTEM = "You are an expert in software clone detection."
PROMPT = (
    "Determine whether the following two Java functions are code clones, i.e. "
    "they implement the same or highly similar functionality (Type-1 to Type-4 "
    "clones all count). Reply with ONLY a JSON object: {{\"clone\": true}} or "
    "{{\"clone\": false}}.\n\n### Function 1:\n```java\n{c1}\n```\n\n"
    "### Function 2:\n```java\n{c2}\n```"
)

usage_lock = threading.Lock()
usage = {"in": 0, "out": 0, "errors": 0, "truncated": 0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["bcb", "gcj"], required=True)
    p.add_argument("--per-type", type=int, default=80, help="BCB: pairs per clone type")
    p.add_argument("--n-clone", type=int, default=400, help="GCJ: clone pairs")
    p.add_argument("--n-nonclone", type=int, default=400)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-chars", type=int, default=3500)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--api-key-file", default=os.path.expanduser("~/.deepseek_api_key"))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=0, help="debug: cap total calls")
    p.add_argument("--types", default=",".join(BCB_TYPES),
                   help="BCB: comma-separated clone types to sample")
    p.add_argument("--skip-nc", action="store_true", help="BCB: skip non-clone sample")
    return p.parse_args()


def read_code(path, max_chars):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            code = f.read().strip()
    except OSError:
        return None
    if len(code) > max_chars:
        with usage_lock:
            usage["truncated"] += 1
        code = code[:max_chars] + "\n// ... [truncated]"
    return code


def sample_bcb(args, rng):
    by_type = defaultdict(list)
    with open(BCB_PAIRS) as f:
        for row in csv.DictReader(f):
            key = row["clone_type"] if int(row["label"]) == 1 else "NC"
            by_type[key].append((row["path_a"], row["path_b"], int(row["label"]), key))
    picked = []
    for t in args.types.split(","):
        if not by_type[t]:
            raise SystemExit(f"no pairs found for clone type {t!r}")
        picked += rng.sample(by_type[t], min(args.per_type, len(by_type[t])))
    if not args.skip_nc:
        picked += rng.sample(by_type["NC"], min(4 * args.per_type, len(by_type["NC"])))
    return picked


def sample_gcj(args, rng):
    clones, nc = [], []
    with open(GCJ_PAIRS) as f:
        for row in csv.DictReader(f):
            rec = (os.path.join(REPO, row["path_a"]), os.path.join(REPO, row["path_b"]),
                   int(row["label"]), "GCJ")
            (clones if rec[2] == 1 else nc).append(rec)
    return rng.sample(clones, args.n_clone) + rng.sample(nc, args.n_nonclone)


def call_llm(session, args, api_key, c1, c2):
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT.format(c1=c1, c2=c2)}],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    last_err = None
    for attempt in range(4):
        try:
            r = session.post(f"{args.base_url}/chat/completions", json=payload,
                             headers={"Authorization": f"Bearer {api_key}"}, timeout=90)
            r.raise_for_status()
            data = r.json()
            u = data.get("usage", {})
            with usage_lock:
                usage["in"] += u.get("prompt_tokens", 0)
                usage["out"] += u.get("completion_tokens", 0)
            text = data["choices"][0]["message"]["content"]
            m = re.search(r'"clone"\s*:\s*(true|false)', text, re.I)
            if m:
                return 1 if m.group(1).lower() == "true" else 0
            if re.search(r"\btrue\b", text, re.I):
                return 1
            if re.search(r"\bfalse\b", text, re.I):
                return 0
            last_err = f"unparseable: {text[:80]}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:120]
        time.sleep(2 * (attempt + 1))
    with usage_lock:
        usage["errors"] += 1
    return {"error": last_err}


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    with open(args.api_key_file) as f:
        api_key = f.read().strip()

    out_dir = args.output_dir or f"{REPO}/results/baselines_rerun/deepseek_llm/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    pairs = sample_bcb(args, rng) if args.dataset == "bcb" else sample_gcj(args, rng)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{args.dataset}: {len(pairs)} pairs, composition "
          f"{dict(Counter(p[3] for p in pairs))}", flush=True)

    session = requests.Session()

    def work(rec):
        pa, pb, label, ptype = rec
        c1, c2 = read_code(pa, args.max_chars), read_code(pb, args.max_chars)
        if c1 is None or c2 is None:
            return {"path_a": pa, "path_b": pb, "label": label, "type": ptype,
                    "pred": "", "error": "missing file"}
        res = call_llm(session, args, api_key, c1, c2)
        if isinstance(res, dict):
            return {"path_a": pa, "path_b": pb, "label": label, "type": ptype,
                    "pred": "", "error": res["error"]}
        return {"path_a": pa, "path_b": pb, "label": label, "type": ptype,
                "pred": res, "error": ""}

    results, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(work, pairs):
            results.append(res)
            done += 1
            if done % 50 == 0:
                print(f"progress {done}/{len(pairs)} "
                      f"(in={usage['in']} out={usage['out']} err={usage['errors']})",
                      flush=True)

    pred_file = os.path.join(out_dir, "predictions.csv")
    with open(pred_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path_a", "path_b", "label", "type", "pred", "error"])
        w.writeheader()
        w.writerows(results)

    valid = [r for r in results if r["pred"] != ""]
    tp = sum(1 for r in valid if r["label"] == 1 and r["pred"] == 1)
    fp = sum(1 for r in valid if r["label"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in valid if r["label"] == 1 and r["pred"] == 0)
    tn = sum(1 for r in valid if r["label"] == 0 and r["pred"] == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_type = {}
    for t in sorted({r["type"] for r in valid} - {"NC", "GCJ"}):
        tv = [r for r in valid if r["type"] == t]
        per_type[t] = {"n": len(tv), "recall": sum(r["pred"] for r in tv) / len(tv)}

    cost = usage["in"] / 1e6 * PRICE_IN + usage["out"] / 1e6 * PRICE_OUT
    summary = {
        "model": args.model, "temperature": 0.0, "seed": args.seed,
        "max_chars": args.max_chars, "n_pairs": len(pairs), "n_valid": len(valid),
        "n_errors": usage["errors"], "n_truncated_functions": usage["truncated"],
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(valid) if valid else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_type_recall": per_type,
        "tokens": {"in": usage["in"], "out": usage["out"]},
        "est_cost_cny_offpeak": round(cost, 3),
        "protocol_note": (
            "zero-shot binary judgment, no training; BCB: per-type stratified "
            "sample + 4x non-clones (recall per type, labelled precision); "
            "GCJ: balanced clone/non-clone sample. Comparable to literature "
            "GPT-4 zero-shot rows, NOT to trained-classifier rows."),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    sys.path.insert(0, f"{REPO}/tools")
    try:
        from run_meta import save_run_meta
        save_run_meta(out_dir, extra={"argv": sys.argv})
    except Exception as e:  # noqa: BLE001
        print(f"run_meta skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
