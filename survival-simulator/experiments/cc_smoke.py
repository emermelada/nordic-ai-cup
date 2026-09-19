#!/usr/bin/env python3
"""cc_smoke.py - verify the carrying-capacity rule activates AND changes the population.

Two things must hold before spending 40 seeds:
  1. the arm is not inert (a silent no-op has bitten this project four times)
  2. it does what it claims: LOWERS the fleet's population in the mid-game
"""
import os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import best_controller as bc  # noqa: E402
import oracle_probe as op  # noqa: E402
from env_wrapper import make_action  # noqa: E402

BASE = os.path.join(os.path.dirname(HERE), "best_controller", "params.json")
import json  # noqa: E402
P0 = dict(bc.DEFAULT_PARAMS)
P0.update(json.load(open(BASE)))


def run(seed, over, ticks=14000):
    P = dict(P0)
    P.update(over)
    random.seed(seed)
    np.random.seed(seed)
    core = op.build_core(seed)
    f = bc.make_policy(P)
    bc.reset_memory()
    pops = []
    for t in range(ticks):
        states = [core.env.get_agent_state(a.agent_id) for a in list(core.env.agents)]
        if not states:
            return t, pops
        acts = [(s["agent_id"], make_action(s, f(s))) for s in states]
        core.step(acts)
        if t % 500 == 0:
            pops.append(len(core.env.agents))
    return ticks, pops


if __name__ == "__main__":
    for seed in (3500, 3501, 3502):
        t0, p0 = run(seed, {})
        t1, p1 = run(seed, {"cc_k0": 12.0, "cc_floor": 2.0})
        mid0 = sum(p0[8:16]) / max(1, len(p0[8:16]))
        mid1 = sum(p1[8:16]) / max(1, len(p1[8:16]))
        print(f"seed {seed}: OFF ticks={t0:6d} mid-pop={mid0:5.1f} | CC ticks={t1:6d} mid-pop={mid1:5.1f} "
              f"| differs={'YES' if (t0 != t1 or p0 != p1) else 'NO <<< INERT'}")
