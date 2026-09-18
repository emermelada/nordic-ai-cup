"""ABLATE the phase policy: test ONE component at a time on Linux.

The first phase attempt coupled movement + spawning and lost 1,700-3,800 ticks. Movement turned out
to be the INCOME mechanism (access-limited food), not discretionary burn, so this isolates the
reproduction side, which has a clean mechanism: a spawn costs the parent 100 energy, the child starts
at 75, and every extra agent adds 0.1/tick metabolism forever while score stays +0.1/tick regardless
of population.

To isolate spawning, movement is left EXACTLY as baseline in every arm below:
  famine_move_frac = 1.0 and famine_blind_stop = 0.0  -> movement untouched
so any difference is attributable to the spawn gate alone.

Usage: python phase_ablate.py <seed> <horizon>
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

# movement identical to baseline in ALL arms -> the spawn gate is the only variable
NEUTRAL_MOVE = {"famine_move_frac": 1.0, "famine_blind_stop": 0.0}

ARMS = [
    ("A0 baseline (phase off)", {}),
    ("A1 spawn-gate only (pop<=3, bank .9)", dict(NEUTRAL_MOVE, phase_mode=1.0, famine_tick_lo=4000, famine_tick_hi=7500)),
    ("A5 spawn-gate, BANK RELAXED to .5", dict(NEUTRAL_MOVE, phase_mode=1.0, famine_tick_lo=4000, famine_tick_hi=7500, famine_bank_frac=0.5)),
    ("A6 spawn-gate, BANK OFF (0.0) + pop<=3", dict(NEUTRAL_MOVE, phase_mode=1.0, famine_tick_lo=4000, famine_tick_hi=7500, famine_bank_frac=0.0)),
]


def params(overrides):
    P = dict(bc.DEFAULT_PARAMS)
    P.update(FILE_PARAMS)
    P.update(overrides)
    return P


def run(P, label):
    bc.reset_memory()
    fn = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=5, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=SEED)
    states = core.env.get_agent_state
    n, maxn = 0, 0
    for i in range(H):
        live = [states(a.agent_id) for a in core.env.agents]
        if not live:
            break
        acts = [(s["agent_id"], make_action(s, fn(s))) for s in live]
        core.step(acts)
        n = i + 1
        maxn = max(maxn, len(core.env.agents))
    print("%-40s ticks=%5d score=%7.1f pop_peak=%d" % (label, n, core.env.score, maxn), flush=True)
    return n


if __name__ == "__main__":
    print("PHASE-ABLATION seed=%d horizon=%d" % (SEED, H))
    for label, ov in ARMS:
        run(params(ov), label)
