#!/usr/bin/env python
"""Build dataset/BCB/pairs_mapped.csv from the raw BigCloneBench inputs.

Inputs (see README, BigCloneBench section):
  dataset/BCB/clone.csv      clone pairs, no header
  dataset/BCB/nonclone.csv   non-clone pairs, header FUNCTION_ID_ONE,FUNCTION_ID_TWO
  dataset/BCB/id2sourcecode_extracted/id2sourcecode/<id>.java

Output columns: id_a,id_b,label,path_a,path_b. Paths are relative to the repository
root (run the pipeline from there). Clone rows come first, then non-clone rows, each in
input-file order.

Note: clone.csv is read with pandas' default header handling, so its first row
(20601756,23594635) is consumed as a header and not emitted. This reproduces the pair
list used for all reported results (269,999 clone + 279,032 non-clone = 549,031 rows).

Usage: python tools/prepare_bcb_pairs.py
"""
import os

import pandas as pd

BCB_DIR = os.path.join('dataset', 'BCB')
SRC_DIR = os.path.join(BCB_DIR, 'id2sourcecode_extracted', 'id2sourcecode')

clone = pd.read_csv(os.path.join(BCB_DIR, 'clone.csv'))
nonclone = pd.read_csv(os.path.join(BCB_DIR, 'nonclone.csv'))

pairs = pd.concat([
    pd.DataFrame({'id_a': clone.iloc[:, 0], 'id_b': clone.iloc[:, 1], 'label': 1}),
    pd.DataFrame({'id_a': nonclone.iloc[:, 0], 'id_b': nonclone.iloc[:, 1], 'label': 0}),
], ignore_index=True)
pairs['path_a'] = pairs['id_a'].map(lambda i: os.path.join(SRC_DIR, f'{i}.java'))
pairs['path_b'] = pairs['id_b'].map(lambda i: os.path.join(SRC_DIR, f'{i}.java'))

out = os.path.join(BCB_DIR, 'pairs_mapped.csv')
pairs.to_csv(out, index=False)
print(f'Wrote {len(pairs)} pairs ({int(pairs.label.sum())} clone) to {out}')
