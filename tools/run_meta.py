"""Persist run provenance next to every experiment output.

Usage (from any tools/ script, after writing its summary):
    from run_meta import save_run_meta
    save_run_meta("results/<exp>")

Writes results/<exp>/run_meta.json with git SHA, dirty flag, argv, timestamp,
and key library versions. Never raises — provenance must not kill a finished run.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def save_run_meta(out_dir: str, extra: dict | None = None) -> None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        import numpy, pandas, sklearn, xgboost  # noqa: PLC0415
        meta = {
            "git_sha": sha, "git_dirty": dirty,
            "argv": sys.argv,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__,
                         "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        }
        if extra:
            meta.update(extra)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"run_meta: could not save provenance ({e!r})", file=sys.stderr)
