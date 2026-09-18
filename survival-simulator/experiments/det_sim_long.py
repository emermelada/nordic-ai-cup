"""Decisive test: is the SIMULATOR itself nondeterministic across episodes?

The policy is a pure constant function -- it holds NO state whatsoever and returns the same action
every call. So if repeated episodes in one process disagree, the nondeterminism lives in src/, not
in our code.

Suspected mechanism: the world keeps entities in `defaultdict(set)` spatial grids
(environment.py: `self.grid_fruits = defaultdict(set)`), and Python hashes objects by identity
(id()-based). Iterating a set of objects therefore yields an order that depends on MEMORY ADDRESSES,
which differ for every newly allocated object -- so the order can differ episode to episode (and
process to process) even with an identical seed. Any logic that walks a grid cell (fruit pickup
order, local-agent lists) then sees a different order and can diverge chaotically.

Scores identical across episodes => sim deterministic (and the variance we see is policy-side).
Scores differ => the sim is order-dependent, which matters a lot: the grader runs the SAME sim, so
its per-run scores for an identical policy are also partly a coin flip.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import env_wrapper
from env_wrapper import run_eval_episode
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

SEED = 100
H = 16000


def constant(state):
    return [0.0, 0.0, 0.0, 0.0]


def raw_loop(seed, horizon, label):
    """Own loop, one agent, no policy state, no stop_on_death: cheapest possible chaos detector."""
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=1, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    states = core.env.get_agent_state
    for _ in range(horizon):
        live = [states(a.agent_id) for a in core.env.agents]
        if not live:
            break
        acts = [(s["agent_id"], ActionRequest(agent_id=s["agent_id"], move_distance=0.0,
                                              move_direction=0.0, turn_angle=0.0,
                                              spawn_agent=False)) for s in live]
        core.step(acts)
    # a fingerprint of the final world state: positions rounded, fruit count, score
    pos = sorted((round(a.x, 3), round(a.y, 3)) for a in core.env.agents)
    print("  %-14s score=%10.4f  agents=%d  fruits=%d  fingerprint=%s"
          % (label, core.env.score, len(core.env.agents), len(core.env.fruits),
             hash(tuple(pos)) & 0xFFFFFF))
    return core.env.score


if __name__ == "__main__":
    print("=== stateless policy via env_wrapper.run_eval_episode (stop_on_death=False, h=%d) ===" % H)
    out = []
    for k in range(3):
        r = run_eval_episode(constant, n_agents=5, seed=SEED, horizon=H, stop_on_death=False,
                             reset_fn=None)
        out.append(r["score"])
        print("  ep%d score=%.4f ticks=%d" % (k + 1, r["score"], r["steps"]))
    print("  -> %s" % ("IDENTICAL" if len(set(out)) == 1 else "DIFFERENT: %s" % out))

    print()
    print("=== raw loop, ONE agent, no wrapper (isolates the sim completely) ===")
    raw = [raw_loop(SEED, H, "run%d" % (k + 1)) for k in range(3)]
    print("  -> %s" % ("IDENTICAL" if len(set(raw)) == 1 else "DIFFERENT: %s" % raw))
