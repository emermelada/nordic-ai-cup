"""Run ONE episode with ONE policy in a fresh process; print one JSON line.

Keeps no state between runs (the digest is accumulated, not stored) so allocator/GC pressure is
identical across runs. Usage: python _ab_episode.py {shipped|fast} <seed> <horizon>
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
import best_controller_fast as bcf
from env_wrapper import run_eval_episode

which, seed, horizon = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mod = {"shipped": bc, "fast": bcf}[which]
h = hashlib.sha256()
nacts = 0


def rec(i, livestates, acts, out):
    global nacts
    for aid, a in acts:
        h.update(repr((aid, a.move_distance, a.move_direction, a.turn_angle,
                       bool(a.spawn_agent))).encode())
        nacts += 1


t0 = time.perf_counter()
r = run_eval_episode(mod.make_policy(mod._load_params()), n_agents=5, seed=seed, horizon=horizon,
                     stop_on_death=True, recorder=rec, reset_fn=mod.reset_memory)
wall = time.perf_counter() - t0
print(json.dumps({"policy": which, "seed": seed, "horizon": horizon,
                  "ticks": r["steps"], "score": round(r["score"], 6), "spawns": r["spawns"],
                  "fruits": r["fruits_eaten"], "predated": r["predated"],
                  "final_agents": r["final_agents"], "n_actions": nacts,
                  "action_sha": h.hexdigest()[:16], "wall_s": round(wall, 3),
                  "ms_per_tick": round(1000.0 * wall / max(1, r["steps"]), 3)}))
