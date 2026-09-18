"""Find the FIRST tick where two supposedly-identical episodes diverge.

Established: sim is deterministic (stateless policy identical x3, raw loop fingerprint identical);
policy state is empty at the first call with reset_fn (verified); yet the real policy gives
5286, 5286, 6576 across three episodes in one process.

So record a per-tick fingerprint of the whole world for two fresh episodes and diff them. The first
differing tick localises the cause: print the pre-tick state of BOTH runs there (agent energies,
ages, positions, agent ids, fruit count) so we can see WHAT input differed -- which tells us whether
a policy-side value leaked or the world itself advanced differently.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import env_wrapper
from env_wrapper import make_action
from src.core import SimulationCore
import best_controller as bc

SEED = 100
H = 16000

with open(os.path.join(REPO, "best_controller", "params.json")) as f:
    FILE_PARAMS = json.load(f)


def fingerprint(core):
    return (
        tuple(sorted((int(a.agent_id), round(a.x, 6), round(a.y, 6), round(a.energy, 6),
                      round(a.age, 6)) for a in core.env.agents)),
        len(core.env.fruits),
        round(core.env.score, 6),
        round(core.env.time, 6),
    )


def episode(label, horizon=H):
    bc.reset_memory()
    P = dict(bc.DEFAULT_PARAMS)
    P.update(FILE_PARAMS)
    fn = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=5, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=SEED)
    states = core.env.get_agent_state
    frames = []
    for i in range(horizon):
        live = [states(a.agent_id) for a in core.env.agents]
        if not live:
            frames.append(("dead", i))
            break
        acts = [(s["agent_id"], make_action(s, fn(s))) for s in live]
        core.step(acts)
        f = fingerprint(core)
        if i < 3 or i % 100 == 0:
            frames.append(f)
    return label, core.env.score, len(frames), frames, core


if __name__ == "__main__":
    print("seed=%d horizon=%d controller=%s" % (SEED, H, bc.__file__))
    runs = []
    for k in range(4):
        label, score, nframes, frames, core = episode("run%d" % (k + 1))
        runs.append((label, score, frames))
        print("  %s: score=%.4f  frames=%d  final_agents=%d fruits=%d"
              % (label, score, nframes, len(core.env.agents), len(core.env.fruits)))

    base_label, base_score, base_frames = runs[0]
    print()
    for label, score, frames in runs[1:]:
        if score == base_score:
            print("  %s == %s (%.4f)" % (label, base_label, score))
            continue
        # find first differing recorded frame
        diff = None
        for j in range(min(len(base_frames), len(frames))):
            if base_frames[j] != frames[j]:
                diff = j
                break
        print("  %s (%.4f) DIVERGES from %s (%.4f) at recorded frame #%s"
              % (label, score, base_label, base_score, diff))
        if diff is not None:
            print("    base frame: %s" % (str(base_frames[diff])[:220],))
            print("    new  frame: %s" % (str(frames[diff])[:220],))
