#!/usr/bin/env python3
"""psp_smoke.py - verify the phase-speed rule actually activates (guards against a silent no-op).

Three silent no-ops have already bitten this project: unknown-knob overrides, an empty candidate file,
and a builder calling a non-existent function. All three LOOK like ordinary negative results. So before
spending 40 seeds: run the same seeds with the rule off and on, and require that the outcomes differ.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402
import best_controller as bc  # noqa: E402
from src.core import SimulationCore  # noqa: E402
from env_wrapper import make_action  # noqa: E402

DEPLOYED = os.path.join(ROOT, "best_controller", "params.json")


def run(seed, horizon, overrides):
    P = dict(bc.DEFAULT_PARAMS)
    P.update(json.load(open(DEPLOYED)))
    P.update(overrides)
    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    fn = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_trees=50, seed=seed)
    for _ in range(horizon):
        states = [core.env.get_agent_state(a.agent_id) for a in list(core.env.agents)]
        acts = [(s["agent_id"], make_action(s, fn(s))) for s in states]
        core.step(acts)
        if not core.env.agents:
            break
    return round(float(core.env.score), 3), len(core.env.agents), round(float(core.env.time) * 10)


print("phase-speed rule smoke test | knob exists in DEFAULT_PARAMS:",
      "psp_mode" in bc.DEFAULT_PARAMS)
diffs = 0
for sd in (3500, 3501, 3502):
    off = run(sd, 4000, {})
    on = run(sd, 4000, {"psp_mode": 1.0})
    same = (off == on)
    diffs += (not same)
    print(f"  seed {sd}: rule OFF score={off[0]:>8} agents={off[1]:>2} tick={off[2]:>5} | "
          f"ON score={on[0]:>8} agents={on[1]:>2} tick={on[2]:>5} | "
          f"{'IDENTICAL (silent no-op!)' if same else 'differs -> the rule is live'}")
print(f"\n{sum(1 for _ in range(diffs))}/{3} seeds differ. "
      f"{'RULE IS LIVE' if diffs else 'WARNING: the rule appears INERT - do not run the 40-seed test'}")
