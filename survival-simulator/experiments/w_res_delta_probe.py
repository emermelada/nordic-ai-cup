"""w_res_delta_probe.py - how BIG is the learned correction actually? (chaos test for W1)

If the residual only nudges the base action by ~0.1%, then any eval swing is chaos and the +35% seen in
training is not learning. This runs one real episode with base+residual and reports the distribution of
the APPLIED deltas in the action space, plus the fraction of decisions where the correction is
effectively zero.

Usage: python w_res_delta_probe.py <weights.npz> <seed> <horizon> <warmup>
"""
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
import best_controller as bc
from env_wrapper import build_obs
from rl_residual import NpResidual, res_obs, base_action, apply_residual, BOUND_FRAC, BOUND_DIR, BOUND_TURN


def main():
    npz, seed, horizon, warmup = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    p = dict(bc.DEFAULT_PARAMS)
    blob = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "experiments", "params_H1_rollback.json")))
    p.update(blob.get("params", blob))
    bc.reset_memory()
    fn = bc.make_policy(p)
    pol = NpResidual.from_npz(npz)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    rng = random.Random(seed)
    d_frac, d_dir, d_turn, raw_abs = [], [], [], []
    for t in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a) for a in live) if s]
        if not states:
            break
        acts = []
        for s in states:
            b = base_action(fn, s)
            if t < warmup:
                acts.append((s["agent_id"], ActionRequest(agent_id=s["agent_id"], move_distance=b[0],
                                                          move_direction=b[1], turn_angle=b[2],
                                                          spawn_agent=bool(b[3]))))
                continue
            o = res_obs(s, b)
            raw, _lp, _z = pol.act(o, rng, greedy=True)
            raw_abs.append(max(abs(float(raw[0])), abs(float(raw[1])), abs(float(raw[2]))))
            d_frac.append(abs(BOUND_FRAC * math.tanh(float(raw[0]))))
            d_dir.append(abs(BOUND_DIR * math.tanh(float(raw[1]))))
            d_turn.append(abs(BOUND_TURN * math.tanh(float(raw[2]))))
            ex = apply_residual(b, raw)
            acts.append((s["agent_id"], ActionRequest(agent_id=s["agent_id"], move_distance=ex[0],
                                                       move_direction=ex[1], turn_angle=ex[2],
                                                       spawn_agent=bool(ex[3]))))
        core.step(acts)
    n = max(1, len(d_frac))
    print(f"seed={seed} decisions measured: {n} (after warmup {warmup})", flush=True)
    for label, arr in (("|delta dist frac|", d_frac), ("|delta dir| rad", d_dir), ("|delta turn| rad", d_turn)):
        arr = sorted(arr)
        print(f"  {label:18s} mean {statistics.mean(arr):.5f}  median {arr[n//2]:.5f}  "
              f"p90 {arr[int(0.9*n)]:.5f}  max {arr[-1]:.5f}", flush=True)
    tiny = sum(1 for x in d_dir if x < 0.01) / n
    print(f"  fraction of decisions with |delta dir| < 0.01 rad (effectively untouched): {tiny:.1%}", flush=True)


if __name__ == "__main__":
    main()
