"""Isolate the nondeterminism: SIM or POLICY?

det_probe.py showed the same policy+seed giving 6576 / 5743 / 4856 / 4674 / 6576 inside ONE
process, and three harnesses disagree on seed 100 (4674 / 5363 / 5368). Two processes of the same
harness agree exactly, so something varies per EPISODE within a process.

This separates the two candidate causes using a CONSTANT policy that has no memory at all:
  - constant policy varies per episode  -> the SIMULATOR is order-dependent (src/), and no amount
    of policy hygiene fixes it.
  - constant policy stable, real policy varies -> the POLICY holds unreset state.

Run it in two separate processes and compare (PYTHONHASHSEED printed to rule out hash effects).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode

SEED = 100
H = 16000


def constant_policy(state):
    return [0.0, 0.0, 0.0, 0.0]      # never move, never spawn


if __name__ == "__main__":
    print("PYTHONHASHSEED=%s  pid=%d" % (os.environ.get("PYTHONHASHSEED", "<unset>"), os.getpid()))
    print("=== constant policy, seed %d, horizon %d, THREE episodes in one process ===" % (SEED, H))
    for k in range(3):
        r = run_eval_episode(constant_policy, n_agents=5, seed=SEED, horizon=H,
                             stop_on_death=True, reset_fn=None)
        print("  episode %d: ticks=%6d score=%8.1f spawns=%d fruits=%d predated=%d"
              % (k + 1, r["steps"], r["score"], r["spawns"], r["fruits_eaten"], r["predated"]))
