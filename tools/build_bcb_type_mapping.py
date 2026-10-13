"""Map every BCB pair to its clone type (needed for the Krinke-compliant evaluation).

Sources (dataset/BCB/): type-1..type-5.csv with columns
  id_a, id_b, functionality_type, sim_1, sim_2   (no header)
The five files partition clone.csv (270,000 pairs). The creator banded them by
sim_1 (col 4): type-3 = [0.7,1.0), type-4 = [0.5,0.7), type-5 = [0.0,0.5),
matching BigCloneBench's syntactic-similarity bands. We map to standard names:
  type-1.csv -> T1, type-2.csv -> T2,
  type-3.csv -> VST3 (sim_1 >= 0.9) or ST3 (0.7 <= sim_1 < 0.9),
  type-4.csv -> MT3, type-5.csv -> WT3/T4
Non-clone pairs (label 0) get NC.

Writes dataset/BCB/pairs_with_types.csv = pairs_mapped.csv + clone_type column,
and prints the coverage/type distribution for the paper.
"""
import pandas as pd

BCB = "dataset/BCB"

def load_types(name, band):
    df = pd.read_csv(f"{BCB}/{name}", header=None,
                     names=["id_a", "id_b", "ftype", "sim_1", "sim_2"])
    if callable(band):
        df["clone_type"] = df["sim_1"].map(band)
    else:
        df["clone_type"] = band
    return df[["id_a", "id_b", "clone_type", "sim_1"]]

parts = [
    load_types("type-1.csv", "T1"),
    load_types("type-2.csv", "T2"),
    load_types("type-3.csv", lambda s: "VST3" if s >= 0.9 else "ST3"),
    load_types("type-4.csv", "MT3"),
    load_types("type-5.csv", "WT3/T4"),
]
types = pd.concat(parts, ignore_index=True)
key = pd.DataFrame({
    "lo": types[["id_a", "id_b"]].min(axis=1),
    "hi": types[["id_a", "id_b"]].max(axis=1),
    "clone_type": types["clone_type"],
    "syn_sim": types["sim_1"],
})
dups = key.duplicated(subset=["lo", "hi"]).sum()
print(f"type rows: {len(key)}, duplicate id-pairs across files: {dups}")
key = key.drop_duplicates(subset=["lo", "hi"])

pairs = pd.read_csv(f"{BCB}/pairs_mapped.csv")
pairs["lo"] = pairs[["id_a", "id_b"]].min(axis=1)
pairs["hi"] = pairs[["id_a", "id_b"]].max(axis=1)
merged = pairs.merge(key, on=["lo", "hi"], how="left").drop(columns=["lo", "hi"])
merged.loc[merged["label"] == 0, "clone_type"] = "NC"

n_clones = (merged["label"] == 1).sum()
unmapped = merged[(merged["label"] == 1) & merged["clone_type"].isna()]
print(f"pairs: {len(merged)} ({n_clones} clones); unmapped clones: {len(unmapped)}")
print(merged["clone_type"].value_counts(dropna=False).to_string())

merged.to_csv(f"{BCB}/pairs_with_types.csv", index=False)
print(f"saved {BCB}/pairs_with_types.csv")
