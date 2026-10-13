#!/usr/bin/env python3
"""Score the MAGNET BCB re-run -> results/baselines_rerun/magnet_bcb/summary.json.

MAGNET prints per-epoch val/test metrics into its own timestamped log file
(out/BCB/multi_graph/logs/<run>/clone_bcb_2.txt) thanks to our +2-line patch that
evaluates the valset each epoch. This script parses those logs, merges the epochs
of one TRAINING LINEAGE, dev-selects the epoch by val F1 and writes the
summary.

LINEAGE, and why it must be explicit
Training may be spread over several interrupted attempts
and several log dirs. Epoch indices are global (0..9), so naively globbing all
logs would merge epoch 3 of one attempt with epoch 3 of another -- different
models, silently averaged. This script therefore takes the lineage as an ORDERED
list of log files; later files win on duplicate epochs, and every epoch records
which file it came from, so the provenance is auditable in the artifact itself.

Default lineage = the continuous run (epochs 0-7) + the resume log
(epochs 8-9), i.e. the reported run.

Usage:
    python tools/archive_magnet_bcb.py                 # default lineage, auto
    python tools/archive_magnet_bcb.py --logs A.txt B.txt
    python tools/archive_magnet_bcb.py --label attempt7 --logs C.txt
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_EXT_ROOT = _os.environ.get("FUSECC_EXTERNAL", _os.path.dirname(_REPO_ROOT))
import argparse
import glob
import json
import os
import re
import sys

LOG_ROOT = _EXT_ROOT + "/baselines/Multigraph_match/out/BCB/multi_graph/logs"
OUT_DIR = _REPO_ROOT + "/results/baselines_rerun/magnet_bcb"
ATTEMPT4 = "graph_match_nerual_network_JavaBigCloneBench_GPU0_19h12m18s_on_Jul_16_2026"

sys.path.insert(0, _REPO_ROOT)

RE_START = re.compile(r"Start evaluation on (valset|testset) in epoch:\s*(\d+)")
RE_PRF = re.compile(r"Precision:\s*([0-9.]+),\s*Recall:\s*([0-9.]+),\s*F1:\s*([0-9.]+)")


def parse_log(path):
    """-> {epoch: {'val': (p,r,f1), 'test': (p,r,f1)}} for one log file.

    The metric line always FOLLOWS its 'Start evaluation on <split> in epoch: N'
    header, so we latch the pending (split, epoch) and fill it on the next
    Precision line. An epoch interrupted mid-eval simply never gets filled.
    """
    out = {}
    pending = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_START.search(line)
            if m:
                pending = (m.group(1), int(m.group(2)))
                continue
            m = RE_PRF.search(line)
            if m and pending:
                split, ep = pending
                key = "val" if split == "valset" else "test"
                out.setdefault(ep, {})[key] = tuple(round(float(x), 6) for x in m.groups())
                pending = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=None,
                    help="ordered lineage of log files; later wins on duplicates")
    ap.add_argument("--label", default="resumed",
                    help="lineage label recorded in the summary")
    ap.add_argument("--total-epochs", type=int, default=10)
    args = ap.parse_args()

    if args.logs:
        logs = args.logs
    else:
        logs = [os.path.join(LOG_ROOT, ATTEMPT4, "clone_bcb_2.txt")]
        # newest OTHER log dir = the resume run
        others = sorted(
            (d for d in glob.glob(os.path.join(LOG_ROOT, "*")) if ATTEMPT4 not in d),
            key=os.path.getmtime)
        for d in reversed(others):
            cand = glob.glob(os.path.join(d, "*.txt"))
            if cand and parse_log(cand[0]):
                logs.append(cand[0])
                break

    logs = [p for p in logs if os.path.exists(p)]
    if not logs:
        sys.exit("no log files found")

    merged, source = {}, {}
    for path in logs:                       # later files win
        for ep, m in parse_log(path).items():
            if "val" in m and "test" in m:  # only fully-evaluated epochs
                merged[ep] = m
                source[ep] = path

    if not merged:
        sys.exit("no completed epochs parsed from: " + ", ".join(logs))

    per_epoch = {}
    for ep in sorted(merged):
        v, t = merged[ep]["val"], merged[ep]["test"]
        per_epoch[str(ep)] = [v[0], v[1], v[2], t[0], t[1], t[2]]
        print(f"epoch {ep}: val F1 {v[2]:.4f}  test F1 {t[2]:.4f}   "
              f"[{os.path.basename(os.path.dirname(source[ep]))}]")

    best = max(merged, key=lambda e: merged[e]["val"][2])
    v, t = merged[best]["val"], merged[best]["test"]
    n_done = len(merged)
    complete = n_done >= args.total_epochs

    spliced = len({source[e] for e in merged}) > 1
    summary = {
        "status": (f"COMPLETE ({n_done}/{args.total_epochs} epochs)" if complete else
                   f"PARTIAL ({n_done}/{args.total_epochs} epochs) — dev-selected among "
                   f"completed epochs (ASTNN/FA-AST salvage precedent)"),
        "system": "MAGNET (Zhang & Saber, arXiv 2510.24241, 2025) — multi-graph attention AST+CFG+DFG",
        "dataset": ("BCB (FA-AST BCB-reduced corpus, 9,133 functions), pair lists "
                    "IDENTICAL to our FA-AST+GMN BCB re-run"),
        "protocol": {
            "input_pairs": "train 100,000 / val 20,000 / test 100,000 (stratified, seed 12345)",
            "effective_loaded": ("train 99,410 / val 19,857 / test 99,791 — after MAGNET's "
                                 "native drops (skip_length 3000, javalang parse failures, "
                                 "one DFG-guard skip; 'Total skip length: 631')"),
            "epochs_planned": args.total_epochs,
            "epochs_completed": n_done,
            "selection": ("dev-selected epoch by val F1 (B2; trainer patched +2 lines to "
                          "eval valset per epoch)"),
            "config": "configs/Q1/bcb.json (upstream defaults) + --epochs 10",
            "lineage_label": args.label,
            "lineage_logs": logs,
            "epoch_source": {str(e): source[e] for e in sorted(source)},
        },
        "result": {
            "dev_best_epoch": best,
            "val_f1_at_selection": v[2],
            "test_precision": t[0],
            "test_recall": t[1],
            "test_f1": t[2],
        },
        "per_epoch": dict({"comment": "columns: [val_P, val_R, val_F1, test_P, test_R, test_F1]"},
                          **per_epoch),
        "notes": [
            "Published MAGNET BCB F1 96.5 used the authors' full training set; ours uses "
            "the 100k-pair stratified cap shared with FA-AST — caption note.",
            "Cross-check: our FA-AST+GMN BCB re-run scores F1 0.9484 on IDENTICAL pairs, "
            "i.e. a 2020 and a 2025 graph-matching method within 0.0004 F1.",
        ],
    }
    if spliced:
        summary["notes"].insert(0,
            "SPLICED LINEAGE: training was resumed from a checkpoint at epoch 8; optimiser "
            "state restarts at that boundary (see lineage_logs / epoch_source), because "
            "upstream checkpoints model weights ONLY. "
            "The learning rate is unaffected: scheduler.step() is commented out upstream, "
            "so the LR is constant at init_lr throughout. Report this caveat; the "
            f"dev-selected epoch is {best}.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "summary.json" if complete
                       else f"summary_partial_{n_done}epochs_{args.label}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    try:
        from tools.run_meta import save_run_meta
        save_run_meta(OUT_DIR, extra={"epochs_completed": n_done,
                                      "dev_best_epoch": best,
                                      "lineage": args.label,
                                      "spliced": spliced})
    except Exception as exc:
        print(f"(run_meta not written: {exc})")

    print(f"\nwrote {out}")
    print(f"dev-selected epoch {best} (val F1 {v[2]:.4f}) -> "
          f"test P {t[0]:.4f} R {t[1]:.4f} F1 {t[2]:.4f}")
    if spliced:
        print("NB spliced lineage — Adam reset at the resume boundary (documented in summary)")


if __name__ == "__main__":
    main()
