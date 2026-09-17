#!/usr/bin/env python3
"""Single-source guard for the Survival Simulator controller.

WHY THIS EXISTS
On 2026-09-18 this repo held TWO controller files with the same params:

  * best_controller.py            (repo root)  -> COPYed into the Docker image, served by the VPS
  * experiments/best_controller.py             -> the file every parameter sweep actually tuned

They had drifted apart. Sweeps were measuring the experiments copy while the grader scored the root
copy, so every leaderboard number in that period (580.96 / 655.95 / 771.8 / 860.2 / 631.06) came
from code that nobody had benchmarked. The relay/absolute-energy-gate work never shipped at all.
Worse, the root copy had no reset_memory(), so it carried per-agent state ACROSS episodes inside
one process -- its score depended on seed ORDER, which the grader controls and we cannot reproduce.

The fix is not "remember to copy the file". It is one canonical file plus this check, so drift is
loud instead of silent.

CHECKS
  1. exactly ONE best_controller.py exists in the repo
  2. exactly ONE best_controller/params.json; any other path to it is a SYMLINK to the canonical
     file (so a "useful copy" cannot be edited independently again)
  3. the canonical controller's sha256 matches the hash recorded in best_controller.sha256, and the
     Docker build re-verifies that same hash (see Dockerfile.vps)

USAGE
  python tools/check_controller.py            # verify; exit 0 = consistent, 1 = violation
  python tools/check_controller.py --update   # re-record the hash after an INTENTIONAL change
"""
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANON_PY = REPO / "best_controller.py"
CANON_PARAMS = REPO / "best_controller" / "params.json"
LEDGER = REPO / "best_controller.sha256"
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_all(filename: str):
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if filename in files:
            hits.append(Path(root) / filename)
    return sorted(hits)


def read_ledger():
    if not LEDGER.exists():
        return None
    try:
        line = LEDGER.read_text().strip().split("\n")[0]
        return line.split()[0]
    except Exception:
        return None


def main():
    update = "--update" in sys.argv
    fails, notes = [], []

    if not CANON_PY.exists():
        fails.append("canonical controller missing: %s" % CANON_PY)
    if not CANON_PARAMS.exists():
        fails.append("canonical params missing: %s" % CANON_PARAMS)
    if fails:
        for f in fails:
            print("FAIL  %s" % f)
        return 1

    # ---- check 1: exactly one controller file ----
    controllers = find_all("best_controller.py")
    if len(controllers) != 1:
        fails.append("expected exactly 1 best_controller.py, found %d: %s"
                     % (len(controllers), [str(p.relative_to(REPO)) for p in controllers]))
    elif controllers[0].resolve() != CANON_PY.resolve():
        fails.append("the only controller is not the canonical one: %s" % controllers[0])
    notes.append("controller file: %s" % controllers[0].relative_to(REPO))

    # ---- check 2: params reachable only through the canonical file (copies must be symlinks) ----
    params = [p for p in find_all("params.json") if p.parent.name == "best_controller"]
    for p in params:
        rel = p.relative_to(REPO)
        if p.resolve() == CANON_PARAMS.resolve():
            if p.is_symlink():
                notes.append("params alias OK (symlink -> canonical): %s" % rel)
            else:
                notes.append("params canonical: %s" % rel)
        else:
            fails.append("a SECOND, independently-editable params.json exists: %s" % rel)

    # ---- check 3: recorded hash ----
    actual = sha256(CANON_PY)
    recorded = read_ledger()
    if update:
        LEDGER.write_text("%s  best_controller.py\n" % actual)
        print("UPDATED %s -> %s" % (LEDGER.relative_to(REPO), actual))
        recorded = actual
        notes.append("hash re-recorded")
    if recorded is None:
        fails.append("no hash ledger (run with --update to record)")
    elif recorded != actual:
        fails.append("controller hash MISMATCH: recorded %s, actual %s\n"
                     "      (if this change was intentional, re-run with --update so the Docker\n"
                     "       build check accepts the new file)" % (recorded[:16], actual[:16]))
    else:
        notes.append("hash matches ledger: %s" % actual[:16])

    for n in notes:
        print("ok    %s" % n)
    for f in fails:
        print("FAIL  %s" % f)
    print("\n%s" % ("CONSISTENT: exactly one canonical controller." if not fails
                    else "INCONSISTENT: %d problem(s) -- fix before deploying." % len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
