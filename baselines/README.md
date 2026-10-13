# Baseline re-runs

The third-party detectors are not included. Clone each upstream repository at the listed
commit next to this repository (default `$FUSECC_EXTERNAL/baselines/<name>`, i.e. a sibling
`baselines/` directory), then apply our patch with `git apply baselines/patches/<name>.patch`.
The MIT licence in the top-level `LICENSE` covers FuseCC's own code only. The files in
`patches/`, and any upstream code they modify or the drivers call, keep their upstream
licences: ASTNN MIT, SourcererCC and Oreo GPL-3.0, StoneDetector EPL-2.0. FA-AST, MAGNET,
MSCCD, CCStokener and TriFusion-LLM state no licence, so obtain them from their upstream
repositories under the terms their authors set. ASTNN's MIT notice is reproduced in
`drivers/astnn/LICENSE-ASTNN` and at the top of every file derived from ASTNN.

| Baseline | Upstream | Commit | Upstream licence | Our files |
|---|---|---|---|---|
| ASTNN | <https://github.com/zhangj111/astnn> | `55f3dd5` | MIT (© 2019 zhangj1994) | `patches/astnn.patch`, `drivers/astnn/*.py` (copy into `clone/`), `drivers/astnn/LICENSE-ASTNN` |
| CCStokener | <https://github.com/CCStokener/CCStokener> | `33a6a8d` | none stated | (run as shipped) |
| Deckard | <https://github.com/skyhover/Deckard> | `24164dc` | see upstream `LICENSE` (© University of California / SMU) | `drivers/deckard/config_{bcb,gcj}` |
| FA-AST + GMN | <https://github.com/jacobwwh/graphmatch_clone> | `0080fdb` | none stated | `patches/graphmatch_clone.patch` |
| MAGNET | <https://github.com/ZixianReid/Multigraph_match> | `48b4ce3` | none stated | `patches/Multigraph_match.patch` |
| MSCCD | <https://github.com/zhuwq585/MSCCD> | `6158751` | none stated | `patches/MSCCD.patch` (edit the `<FUSECC_ROOT>` / `<MSCCD_DIR>` placeholders) |
| SourcererCC | <https://github.com/Mondego/SourcererCC> | `88314f5` | GPL-3.0 | `patches/SourcererCC.patch` |
| StoneDetector | <https://github.com/StoneDetector/StoneDetector> | `de8a5eb` | EPL-2.0 | `patches/StoneDetector.patch` |
| TriFusion-LLM | <https://github.com/LMD4790/TriFusion-LLM> | `51a091f` | none stated | `patches/TriFusion-LLM.patch` (upstream `train.py` is empty; the patch reconstructs it from the authors' components) |
| Oreo | artefact of Saini et al., ESEC/FSE 2018 | — | GPL-3.0 | scored by `tools/eval_oreo_bcb.py`, `tools/convert_scc_pairs.py` |

Drivers in `tools/` (`run_faast_bcb.sh`, `run_magnet_bcb_resume.sh`, `run_trifusion_*.sh`,
`prep_magnet_data.py`, `prep_faast_bcb_from_magnet.py`, `archive_*_bcb.py`,
`eval_trifusion_arbitration.py`) expect the checkouts under `$FUSECC_EXTERNAL/baselines/`.
`drivers/deckard/config_*` and `drivers/astnn/*.py` read `FUSECC_ROOT` (and `DECKARD_DIR`)
from the environment. The DeepSeek-based runs read the API key from the file named by
`DEEPSEEK_API_KEY_FILE` (default `~/.deepseek_api_key`); no key is included.
