"""Break down where a tick's wall time actually goes: controller vs sim core.

Usage: python _breakdown.py [seed]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
from env_wrapper import make_action, run_eval_episode
from src.core import SimulationCore

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 100
HORIZON = 6000
N = 5

# ---------- (1) full episode with a timed policy ----------
P = bc._load_params()
fn = bc.make_policy(P)
acc = {"pol": 0.0, "calls": 0, "json": 0.0, "json_calls": 0}


def timed_fn(state):
    t = time.perf_counter()
    a = fn(state)
    acc["pol"] += time.perf_counter() - t
    acc["calls"] += 1
    return a


bc.reset_memory()
t0 = time.perf_counter()
r = run_eval_episode(timed_fn, n_agents=N, seed=SEED, horizon=HORIZON,
                     stop_on_death=True, reset_fn=bc.reset_memory)
wall = time.perf_counter() - t0
steps = r["steps"]
print("seed=%d steps=%d wall=%.3fs  ms/tick=%.3f" % (SEED, steps, wall, 1000 * wall / steps))
print("policy: %d calls, total %.3fs -> %.4f ms/call, %d calls/tick -> %.4f ms/tick (%.2f%% of tick)"
      % (acc["calls"], acc["pol"], 1000 * acc["pol"] / max(1, acc["calls"]),
         acc["calls"] / steps, 1000 * acc["pol"] / steps, 100 * acc["pol"] / wall))

# ---------- (2) served-path wrapper cost (reads params.json per call) ----------
st = {"agent_id": 1, "energy": 300.0, "max_energy": 500.0, "age": 10.0, "biome": "forest",
      "speed": 20.0, "sprint_speed": 40.0, "vision_range": 400.0, "vision_angle": 1.6,
      "hearing_radius": 100.0,
      "observations": [{"type": "Edge", "coords": ((10.0, 10.0), (20.0, 20.0))} for _ in range(13)]
                      + [{"type": "Tree", "distance": 50.0, "angle": 0.3}]}
K = 3000
t0 = time.perf_counter()
for _ in range(K):
    bc._load_params()
jl = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(K):
    bc.best_controller(st)          # served path: json read EVERY call
served = time.perf_counter() - t0
bc.reset_memory()
t0 = time.perf_counter()
for _ in range(K):
    fn(st)                          # pre-bound params
pre = time.perf_counter() - t0
print("served best_controller(state): %.5f ms/call | _load_params: %.5f ms/call | make_policy fn: %.5f ms/call"
      % (1000 * served / K, 1000 * jl / K, 1000 * pre / K))

# ---------- (3) sim-core share: replay the same episode with a constant policy ----------
def const_policy(state):
    return [10.0, 0.0, 0.0, 0.0]


run_eval_episode(const_policy, n_agents=N, seed=SEED, horizon=2000, stop_on_death=True)
t0 = time.perf_counter()
r2 = run_eval_episode(const_policy, n_agents=N, seed=SEED, horizon=HORIZON,
                      stop_on_death=True)
wall2 = time.perf_counter() - t0
print("SIM-ONLY (constant policy) seed=%d steps=%d wall=%.3fs ms/tick=%.3f"
      % (SEED, r2["steps"], wall2, 1000 * wall2 / r2["steps"]))
