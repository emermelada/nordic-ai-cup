"""Validate the phase-policy additions.

1. SIM-TIME ESTIMATOR ACCURACY -- the controller reconstructs sim time from `age` + per-agent call
   counts because the state carries no clock. A phase boundary is worthless if that estimate drifts,
   so compare it against ground truth (the true step index) every tick.
2. BEHAVIOUR PRESERVATION -- phase_mode=0 must reproduce the pre-patch result exactly (seed 100 gave
   462.3278 on this harness).
3. A first phase-ON measurement, to see whether the famine logic helps at all before spending
   evolution time on it.

Usage: python phase_check.py [seed] [horizon]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from env_wrapper import make_action
from src.core import SimulationCore
import best_controller as bc

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 100
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16000

with open(os.path.join(REPO, "best_controller", "params.json")) as f:
    FILE_PARAMS = json.load(f)


def params(overrides):
    P = dict(bc.DEFAULT_PARAMS)
    P.update(FILE_PARAMS)
    P.update(overrides)
    return P


def run(P, label, trace_est=False):
    bc.reset_memory()
    fn = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=5, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=SEED)
    states = core.env.get_agent_state
    errs = []
    phases = []
    n = 0
    for i in range(H):
        live = [states(a.agent_id) for a in core.env.agents]
        if not live:
            break
        acts = [(s["agent_id"], make_action(s, fn(s))) for s in live]
        core.step(acts)
        n = i + 1
        errs.append(bc._SIM_TICK - n)
        phases.append(bc._phase_debug if hasattr(bc, "_phase_debug") else 0.0)
    print("%-26s ticks=%5d score=%7.1f  | sim-time est err: max=%d mean=%+.2f final=%+d"
          % (label, n, core.env.score, max(abs(e) for e in errs),
             sum(errs) / max(1, len(errs)), errs[-1] if errs else 0))
    return n, core.env.score


if __name__ == "__main__":
    print("seed=%d horizon=%d controller=%s" % (SEED, H, bc.__file__))
    print()
    print("=== 1+2. estimator accuracy AND behaviour preservation (phase OFF) ===")
    off_ticks, _ = run(params({}), "phase_mode=0 (baseline)")
    print("   expected from pre-patch run: 462.3278 score / 4472 ticks (macOS seed 100)")
    print()
    print("=== 3. phase ON, a few switch points (exploratory, macOS) ===")
    for lo, hi in ((4000, 7500), (3000, 6000), (5000, 9000)):
        run(params({"phase_mode": 1.0, "famine_tick_lo": lo, "famine_tick_hi": hi}),
            "phase on lo=%d hi=%d" % (lo, hi))
