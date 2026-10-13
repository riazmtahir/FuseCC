"""MAJ-1 follow-up, item 1: isolate the leakage effect with the test distribution held constant.

WHY THIS EXISTS
`results/bcb_function_disjoint` established that a function-disjoint BCB split scores F1 0.9940
against the pair-level headline 0.9986. Those two numbers are measured on DIFFERENT pair sets
(15,707 vs 438,988) with different clone rates, so the -0.0046 gap confounds two things: the
leakage from functions recurring across splits, and the change of test distribution.

This script removes the second factor. It takes the pair-level (bcb_e3) model's out-of-fold
predictions and scores them on EXACTLY the 15,707 pairs that form the function-disjoint TEST
split. Same pairs, same labels, same clone types; the only difference left is whether the model
that produced the prediction was allowed to see the endpoint functions during training.

    pair-level OOF on the fd test pairs   <-- this script
    vs
    function-disjoint model on the same pairs (0.9940)   <-- results/bcb_function_disjoint
    = the leakage effect, test distribution held constant.

THE OOF PREDICTIONS ARE NOT ON DISK
`tools/train_bcb_e3.py` wrote only summary.json and features.npz; it never persisted its
out-of-fold vector. So they are re-derived here by replaying its protocol exactly:
features.npz supplies bcb_e3's own X in bcb_e3's own row order (including its all-files PCA),
and run_regime below is a copy of train_bcb_e3.py's, same seed, same folds, same models. The
replay is verified against the published summary.json before anything is sliced from it: if
the reproduction drifts, the script says so instead of quietly reporting a different model.

ALSO REPORTED (descriptive stats requested alongside MAJ-1)
  * clone rate of the full 438,988-pair T1-T3 set
  * the hub function that carries 33,007 pairs: its degree, its label mix, which
    function-disjoint partition it landed in, and its functionality-group membership

    NB on functionality groups: this BCB export does NOT carry BCB's functionality ids.
    dataset/BCB/clone.csv is described in the project data notes as
    (id_a, id_b, functionality_type, sim_1, sim_2), but its third column takes only the values
    {1,2,3} and reproduces the type-N.csv banding exactly (type-1.csv -> 1, type-2.csv -> 2,
    type-3/4/5.csv -> 3), i.e. it is the coarse CLONE TYPE, not one of BCB's ~43 functionality
    ids. Rather than pass that off as a functionality id, group membership is reported via a
    clearly-labelled proxy: connected components of the clone graph. BCB only ever links two
    functions with a clone edge when they implement the same functionality, so a component is
    a superset of a functionality group. Labelled as a proxy everywhere it appears.

Writes results/bcb_leakage_isolation/leakage_isolation.json (+ the replayed OOF vector, so this
never has to be recomputed). Does not touch results/bcb_e3 or results/bcb_function_disjoint.

Run: nohup python tools/bcb_leakage_isolation.py > logs/bcb_leakage_isolation.log 2>&1 &
"""
import json
import os
import sys
import time
import collections

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
OUT_DIR = "results/bcb_leakage_isolation"
TYPES = ["T1", "T2", "VST3", "ST3", "MT3"]
PROPS = (0.60, 0.20, 0.20)
SCORE_COLS = [f"score_t{i}" for i in range(1, 20)]

t_start = time.time()


def log(m):
    print(f"[{time.time()-t_start:7.1f}s] {m}", flush=True)


# ================================================================ bcb_e3 features, its row order
log("loading results/bcb_e3/features.npz (bcb_e3's own X, row order and all-files PCA)")
d = np.load("results/bcb_e3/features.npz", allow_pickle=True)
X_all, y_all, ct_all = d["X"], d["y"].astype(int), d["clone_type"]
pa_all, pb_all = d["path_a"], d["path_b"]
keep = ct_all != "WT3/T4"
X, y, ct = X_all[keep], y_all[keep], ct_all[keep]
pa, pb = pa_all[keep], pb_all[keep]
log(f"T1-T3 rows {len(y):,} of {len(y_all):,}; X {X.shape}")

clone_rate_full = float(y.mean())
log(f"clone rate of the {len(y):,}-pair T1-T3 set: {clone_rate_full:.4f} "
    f"({int(y.sum()):,} clone / {int((y == 0).sum()):,} non-clone)")


# ================================================================ replay bcb_e3's OOF
def xgb_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device="cuda",
                             random_state=RANDOM_STATE, eval_metric="logloss")


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=7, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


OOF_CACHE = os.path.join(OUT_DIR, "bcb_e3_oof_t123.npy")
if os.path.exists(OOF_CACHE):
    oof = np.load(OOF_CACHE)
    log(f"reusing cached replayed OOF {OOF_CACHE}")
else:
    log("replaying train_bcb_e3.py run_regime() on the T1-T3 regime (5-fold, seed 12345)")
    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for k, (tr, va) in enumerate(cv.split(X, y)):
        xm, lm = xgb_model(), lgb_model()
        xm.fit(X[tr], y[tr])
        lm.fit(X[tr], y[tr])
        meta_tr = np.column_stack([xm.predict_proba(X[tr])[:, 1], lm.predict_proba(X[tr])[:, 1]])
        meta_va = np.column_stack([xm.predict_proba(X[va])[:, 1], lm.predict_proba(X[va])[:, 1]])
        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(meta_tr, y[tr])
        oof[va] = lr.predict_proba(meta_va)[:, 1]
        log(f"  fold {k+1}/5 done")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OOF_CACHE, oof)
    log(f"wrote {OOF_CACHE}")


def report(y_true, prob, thr, types=None):
    pred = (prob >= thr).astype(int)
    out = {"threshold": round(float(thr), 4),
           "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
           "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
           "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
           "auc": round(float(roc_auc_score(y_true, prob)), 4),
           "n": int(len(y_true)),
           "clone_rate": round(float(np.mean(y_true)), 4)}
    if types is not None:
        out["per_type_recall"] = {
            t: {"recall": round(float(pred[(types == t) & (y_true == 1)].mean()), 4),
                "n": int(((types == t) & (y_true == 1)).sum())}
            for t in TYPES if ((types == t) & (y_true == 1)).any()}
    return out


# ---- validity gate: does the replay reproduce the published pair-level headline?
published = json.load(open("results/bcb_e3/summary.json"))["full148"]["t123"]
replayed = report(y, oof, 0.5, ct)
repro = {
    "published_bcb_e3": {"f1": round(published["f1"], 4), "precision": round(published["precision"], 4),
                         "recall": round(published["recall"], 4), "n": published["n"]},
    "replayed_here": {k: replayed[k] for k in ("f1", "precision", "recall", "n")},
    "abs_delta_f1": round(abs(replayed["f1"] - published["f1"]), 5),
    "note": "train_bcb_e3.py never persisted its OOF vector, so it is re-derived by replaying "
            "its protocol on its own saved features.npz (same seed, folds, models). This gate "
            "confirms the replay is the published model before anything is sliced from it.",
}
repro["reproduces"] = bool(repro["abs_delta_f1"] < 1e-3 and replayed["n"] == published["n"])
log(f"REPLAY GATE: published F1 {published['f1']:.4f} vs replayed {replayed['f1']:.4f} "
    f"(delta {repro['abs_delta_f1']:.5f}) -> reproduces={repro['reproduces']}")
if not repro["reproduces"]:
    log("WARNING: the replay does NOT reproduce the published pair-level number. "
        "Everything below is reported against the replayed model and must be read as such.")


# ================================================================ reproduce the fd split
log("reproducing the function-disjoint split (identical code path to train_bcb_function_disjoint.py)")
canon = pd.read_csv("dataset/BCB/pairs_canonical.csv",
                    usecols=["id_a", "id_b", "label", "path_a", "path_b", "clone_type"])
canon = canon[canon.clone_type != "WT3/T4"].reset_index(drop=True)
res = pd.read_csv("results/bcb_19_canonical/results.csv",
                  usecols=["path_a", "path_b", "label"] + SCORE_COLS).dropna()
res = res.merge(canon[["path_a", "path_b", "id_a", "id_b", "clone_type"]],
                on=["path_a", "path_b"], how="inner")

fn_ids = pd.unique(np.concatenate([res.id_a.values, res.id_b.values]))
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(len(fn_ids))
n_tr = int(PROPS[0] * len(fn_ids))
n_va = n_tr + int(PROPS[1] * len(fn_ids))
part = np.empty(len(fn_ids), dtype=np.int8)
part[perm[:n_tr]] = 0
part[perm[n_tr:n_va]] = 1
part[perm[n_va:]] = 2
fn_part = dict(zip(fn_ids, part))

pa_p = res.id_a.map(fn_part).values
pb_p = res.id_b.map(fn_part).values
keep_fd = pa_p == pb_p
split = pa_p[keep_fd]
res_fd = res[keep_fd].reset_index(drop=True)

fd_sum = json.load(open("results/bcb_function_disjoint/summary.json"))
split_check = {
    "n_pairs": {n: int((split == k).sum()) for k, n in [(0, "train"), (1, "val"), (2, "test")]},
    "expected_from_fd_summary": {n: fd_sum["splits"][n]["n_pairs"] for n in ("train", "val", "test")},
}
split_check["matches"] = split_check["n_pairs"] == split_check["expected_from_fd_summary"]
log(f"fd split reproduction: {split_check['n_pairs']} vs expected "
    f"{split_check['expected_from_fd_summary']} -> matches={split_check['matches']}")
assert split_check["matches"], "function-disjoint split did not reproduce; alignment unsafe"

# ---- map the fd TEST pairs onto rows of the bcb_e3 OOF vector
row_of_pair = {(a, b): i for i, (a, b) in enumerate(zip(pa.tolist(), pb.tolist()))}
te_mask = split == 2
te_keys = list(zip(res_fd.path_a.values[te_mask].tolist(), res_fd.path_b.values[te_mask].tolist()))
idx = np.array([row_of_pair[k] for k in te_keys])
assert len(idx) == fd_sum["splits"]["test"]["n_pairs"] == 15707, "fd test pair count mismatch"
assert len(set(idx.tolist())) == len(idx), "duplicate rows in the fd test alignment"
np.testing.assert_array_equal(y[idx], res_fd.label.values[te_mask].astype(int))
log(f"aligned all {len(idx):,} function-disjoint TEST pairs onto bcb_e3 OOF rows "
    f"(labels verified identical)")

# ================================================================ THE COMPARISON
fd_xgb = fd_sum["results"]["xgb_148"]
fd_stk = fd_sum["results"]["stacked_xgb_lgb_lr"]

pair_on_fd_test = {
    "at_0.5": report(y[idx], oof[idx], 0.5, ct[idx]),
    "at_fd_val_threshold_0.37": report(y[idx], oof[idx], fd_stk["test"]["threshold"], ct[idx]),
}
log(f"pair-level OOF on the fd TEST pairs: F1 {pair_on_fd_test['at_0.5']['f1']:.4f} "
    f"AUC {pair_on_fd_test['at_0.5']['auc']:.4f} (thr 0.5)")

comparison = {
    "held_constant": "the same 15,707 function-disjoint TEST pairs, same labels, same clone types",
    "varies": "whether the model was trained on pairs sharing endpoint functions with these pairs",
    "pair_level_bcb_e3_on_these_pairs": pair_on_fd_test["at_0.5"],
    "function_disjoint_on_these_pairs": {
        "stacked_at_0.5": fd_stk["test_at_0.5"],
        "stacked_at_val_threshold": fd_stk["test"],
        "xgb148_at_0.5": fd_xgb["test_at_0.5"],
        "xgb148_at_val_threshold": fd_xgb["test"],
        "source": "results/bcb_function_disjoint/summary.json (not recomputed)",
    },
    "leakage_effect_like_for_like_at_0.5": {
        "f1": round(pair_on_fd_test["at_0.5"]["f1"] - fd_stk["test_at_0.5"]["f1"], 4),
        "precision": round(pair_on_fd_test["at_0.5"]["precision"] - fd_stk["test_at_0.5"]["precision"], 4),
        "recall": round(pair_on_fd_test["at_0.5"]["recall"] - fd_stk["test_at_0.5"]["recall"], 4),
        "auc": round(pair_on_fd_test["at_0.5"]["auc"] - fd_stk["test_at_0.5"]["auc"], 4),
        "note": "positive = the pair-level model scores higher on the identical pairs, i.e. the "
                "benefit it gets from having seen these functions during training. Both sides "
                "use threshold 0.5 so the threshold rule is not a confound.",
    },
    "distribution_effect": {
        "f1": round(published["f1"] - pair_on_fd_test["at_0.5"]["f1"], 4),
        "note": "the SAME pair-level model, scored on all 438,988 pairs vs on the 15,707 fd test "
                "pairs. This is the part of the headline-vs-fd gap that is a change of test set, "
                "not leakage.",
    },
}


# ================================================================ hub function + descriptives
log("computing pair degrees over the 438,988-pair T1-T3 set")
deg = collections.Counter(res.id_a.tolist())
deg.update(res.id_b.tolist())
top = deg.most_common(5)

# functionality-group PROXY: connected components of the BCB clone graph
log("building the clone-graph connected-component proxy for functionality groups")
cl = pd.read_csv("dataset/BCB/clone.csv", header=None,
                 names=["id_a", "id_b", "coarse_type", "sim_lines", "sim_tokens"])
nodes = pd.unique(np.concatenate([cl.id_a.values, cl.id_b.values]))
nidx = {n: i for i, n in enumerate(nodes)}
r = cl.id_a.map(nidx).values
c = cl.id_b.map(nidx).values
g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(nodes), len(nodes)))
n_comp, comp = connected_components(g, directed=False)
comp_size = collections.Counter(comp.tolist())
log(f"clone graph: {len(nodes):,} functions, {len(cl):,} edges, {n_comp} connected components")


def hub_record(fid):
    m = (res.id_a.values == fid) | (res.id_b.values == fid)
    lab = res.label.values[m]
    types = res.clone_type.values[m]
    cid = int(comp[nidx[fid]]) if fid in nidx else None
    # of this hub's pairs, how many survive into each function-disjoint partition
    m_fd = (res_fd.id_a.values == fid) | (res_fd.id_b.values == fid)
    surv = {n: int(((split == k) & m_fd).sum()) for k, n in [(0, "train"), (1, "val"), (2, "test")]}
    return {
        "function_id": int(fid),
        "pairs_in_T1_T3_set": int(m.sum()),
        "clone_pairs": int(lab.sum()),
        "nonclone_pairs": int((lab == 0).sum()),
        "clone_rate_of_its_pairs": round(float(lab.mean()), 4),
        "clone_type_mix": {k: int(v) for k, v in collections.Counter(types.tolist()).most_common()},
        "function_disjoint_partition": {0: "train", 1: "val", 2: "test"}[int(fn_part[fid])],
        "its_pairs_surviving_into_each_fd_partition": surv,
        "pairs_discarded_as_cross_partition": int(m.sum() - sum(surv.values())),
        "functionality_group_proxy": {
            "clone_graph_component_id": cid,
            "component_size_functions": int(comp_size[cid]) if cid is not None else None,
            "in_clone_graph": fid in nidx,
        },
    }


hub_functions = [hub_record(f) for f, _ in top]
log(f"top hub {top[0][0]} carries {top[0][1]:,} pairs, lands in the "
    f"{hub_functions[0]['function_disjoint_partition']} partition")

# how much of each fd partition's pair mass the hubs account for
hub_ids = {f for f, _ in top}
hub_in_test = res_fd[(res_fd.id_a.isin(hub_ids) | res_fd.id_b.isin(hub_ids)) & (split == 2)]

descriptives = {
    "full_T1_T3_pair_set": {
        "n_pairs": int(len(y)),
        "n_clone": int(y.sum()),
        "n_nonclone": int((y == 0).sum()),
        "clone_rate": round(clone_rate_full, 4),
        "clone_type_counts": {k: int(v) for k, v in collections.Counter(ct.tolist()).most_common()},
    },
    "hub_functions_top5_by_degree": hub_functions,
    "hub_note": f"the maximum degree is {top[0][1]:,} pairs and it is a TIE between "
                f"{top[0][0]} and {top[1][0]} (both {top[1][1]:,}); the five largest hubs are all "
                f"adjacent ids in the 236771xx block, i.e. one cluster of near-identical functions",
    "hub_pairs_inside_fd_test_split": int(len(hub_in_test)),
    "functionality_group_availability": {
        "bcb_functionality_ids_present_locally": False,
        "evidence": "dataset/BCB/clone.csv column 3 is documented as functionality_type in "
                    "its source description (see README, BigCloneBench section) but takes only values {1,2,3} (48,116 / 4,234 / "
                    "217,650) and reproduces the type-N.csv banding exactly, so it is the coarse "
                    "clone type, not one of BCB's ~43 functionality ids. No other local file "
                    "carries a functionality id.",
        "proxy_used": "connected components of the BCB clone graph (BCB only links two functions "
                      "as clones when they share a functionality, so a component is a superset of "
                      "a functionality group)",
        "clone_graph_components": int(n_comp),
        "largest_components": [{"component_id": int(k), "size_functions": int(v)}
                               for k, v in comp_size.most_common(5)],
        "proxy_limitation": f"the proxy does NOT resolve individual functionalities: "
                            f"{n_comp} components, but one giant component holds "
                            f"{comp_size.most_common(1)[0][1]:,} of {len(nodes):,} clone-graph "
                            f"functions ({100*comp_size.most_common(1)[0][1]/len(nodes):.1f}%), so "
                            f"'component 0' is close to 'anything reachable through clone edges' "
                            f"rather than a single BCB functionality. Read the hub's group "
                            f"membership as 'inside the dominant clone cluster', nothing finer. "
                            f"Recovering true functionality ids needs BCB's own functionality "
                            f"table, which is not in this export.",
    },
}

# ================================================================ write
summary = {
    "purpose": "MAJ-1 item 1: leakage isolated with the test distribution held constant "
               "(pair-level bcb_e3 OOF scored on the function-disjoint TEST pairs)",
    "replay_gate": repro,
    "fd_split_reproduction": split_check,
    "comparison": comparison,
    "pair_level_on_fd_test_all_thresholds": pair_on_fd_test,
    "pair_level_on_full_438988": {k: replayed[k] for k in
                                  ("threshold", "precision", "recall", "f1", "auc", "n",
                                   "clone_rate", "per_type_recall")},
    "descriptives": descriptives,
    "inputs": {
        "features": "results/bcb_e3/features.npz (bcb_e3's own X and row order)",
        "fd_reference": "results/bcb_function_disjoint/summary.json",
        "oof_vector": OOF_CACHE,
    },
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "leakage_isolation.json"), "w") as f:
    json.dump(summary, f, indent=2)
log(f"wrote {OUT_DIR}/leakage_isolation.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_meta import save_run_meta  # noqa: E402

prior = {}
rm_path = os.path.join(OUT_DIR, "run_meta.json")
if os.path.exists(rm_path):
    prior = json.load(open(rm_path))
save_run_meta(OUT_DIR, extra={"component": "leakage_isolation (item 1)",
                              "outputs": ["leakage_isolation.json", os.path.basename(OOF_CACHE)],
                              "prior_runs": prior.get("prior_runs", []) +
                                            ([{k: prior[k] for k in ("component", "timestamp_utc", "argv")
                                               if k in prior}] if prior else [])})
log(f"wrote {rm_path}")
print(json.dumps({k: summary[k] for k in ("replay_gate", "comparison", "descriptives")}, indent=2),
      flush=True)
log("done")
