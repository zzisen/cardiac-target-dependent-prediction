"""Assemble the closure-level all-target A table from the two restart records."""
import json
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_A_missing_target_restart as run

if __name__ == "__main__":
    original = json.loads(run.SOURCE.read_text(encoding="utf-8"))
    missing = json.loads(run.RESULT.read_text(encoding="utf-8"))
    previous = json.loads(run.EXISTING.read_text(encoding="utf-8"))
    run.merge_full_matrix(original, missing, previous)
    print(run.CLOSURE_ROOT / "A_FULL_RESTART_AUDIT.csv")
