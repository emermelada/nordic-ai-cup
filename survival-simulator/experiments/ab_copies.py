"""RECORD + GUARD for the 2026-09-18 two-controller incident.

For most of the competition this repo held TWO controllers carrying the SAME params:

  * best_controller.py             (repo root) -> COPYed into the Docker image, served by the VPS
  * experiments/best_controller.py             -> the file every sweep actually tuned

They drifted. Every sweep measured the experiments copy; the grader scored the root copy. So the
relay / absolute-energy-gate work never shipped, and the leaderboard numbers of that period
(580.96 / 655.95 / 771.8 / 860.2 / 631.06) came from code nobody had benchmarked. The root copy
also lacked reset_memory(), so it carried per-agent state across episodes within one process -- its
score depended on seed ORDER, which the grader controls and we cannot reproduce offline.

This script used to A/B the two files. The duplicate is now deleted, so it instead VERIFIES that
the incident cannot recur. `tools/check_controller.py` is the superset check (it also validates the
params aliases and the recorded build hash); this one is the narrow, dependency-free version.

Measured A/B outcome taken just BEFORE the fix (8 TRAIN seeds, horizon 16000, identical params):

    seed   ROOT (live on VPS)   EXPERIMENTS (tuned)   delta
     100          5286                4674            -612
     200          5454                5320            -134
     300          9676                9130            -546
     400          6558                6558               0
     500          3620                3620               0
     600          5768                7997           +2229
     700         10345               11882           +1537
     800          5826                5826               0

    ROOT mean 6566.6 median 5797.0   |   EXP mean 6875.9 median 6192.0
    5/8 seeds differed.

Interpretation: the copies were behaviourally DIFFERENT (so production was scored on a controller
that was never tuned), but the aggregate gap (+309 ticks on the mean) sits inside seed noise
(SE ~800). This was a reproducibility defect, not a demonstrated score improvement.

Usage: python ab_copies.py     (exit 0 = single source intact)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP_PY = os.path.join(HERE, "best_controller.py")          # must NOT exist any more
CANON_PY = os.path.join(REPO, "best_controller.py")        # the only controller
EXP_PARAMS = os.path.join(HERE, "best_controller", "params.json")


def main():
    bad = False
    if os.path.exists(EXP_PY):
        print("FAIL: the duplicate controller is BACK at %s" % EXP_PY)
        bad = True
    if not os.path.exists(CANON_PY):
        print("FAIL: canonical controller missing at %s" % CANON_PY)
        bad = True
    if os.path.exists(EXP_PARAMS) and not os.path.islink(EXP_PARAMS):
        print("FAIL: %s is a real file again (must be a symlink to the canonical params)" % EXP_PARAMS)
        bad = True

    if bad:
        print("\nINCONSISTENT: run `python tools/check_controller.py` for the full report.")
        return 1
    print("OK: exactly one controller (%s); experiments/ holds no copy." % CANON_PY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
