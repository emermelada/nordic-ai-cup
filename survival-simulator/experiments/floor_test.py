"""C1: null-action floor. Does a do-nothing agent already match our 7.4k-tick controller?

Also tests the audit's proposed architecture: park-and-graze + a spawn relay (no electric field).
Variants:  F0 idle            - never move, never spawn
           F1 idle+relay      - never move, spawn when energy > 50%
           F2 graze+relay     - move ONLY toward a visible fruit (else stand still), relay spawn
           V0 deployed        - current shipped controller (reference)
Horizon 16000 so the floor is measured on the same standard surface.
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

H = 16000
SEEDS = [100, 200, 300]


def f0_idle(state):
    return [0.0, 0.0, 0.0, 0.0]


def _spawn_ok(state, frac=0.5):
    e = state.get("energy", 0.0)
    me = max(state.get("max_energy", 1.0) or 1.0, 1.0)
    return 1.0 if e > frac * me else 0.0


def f1_idle_relay(state):
    return [0.0, 0.0, 0.0, _spawn_ok(state)]


def f2_graze_relay(state):
    """Park unless a fruit is visible; then step straight at the nearest one. Spawn when rich."""
    obs = state.get("observations") or []
    fruits = [o for o in obs if o.get("type") == "Fruit"]
    spawn = _spawn_ok(state)
    if not fruits:
        return [0.0, 0.0, 0.0, spawn]
    f = min(fruits, key=lambda o: o["distance"])
    d = f["distance"]
    dist = min(abs(d), _speed(state)) if d < 0 else min(d, _speed(state))
    return [float(dist), float(f["angle"]), 0.0, spawn]


def _speed(state):
    return float(state.get("speed", 10.0))


VARIANTS = [("F0 idle", f0_idle), ("F1 idle+relay", f1_idle_relay),
            ("F2 graze+relay", f2_graze_relay)]

P = dict(DEFAULT_PARAMS)
with open(os.path.join(HERE, "best_controller", "params.json")) as f:
    P.update(json.load(f))

print("NULL-ACTION FLOOR  horizon=%d seeds=%s" % (H, SEEDS))
for tag, fn in VARIANTS:
    ticks = []
    for s in SEEDS:
        r = run_eval_episode(fn, n_agents=5, seed=s, horizon=H, stop_on_death=True,
                             reset_fn=bc.reset_memory)
        ticks.append(r["steps"])
    print("  %-16s median=%7.0f mean=%7.0f ticks=%s" % (tag, st.median(ticks), st.mean(ticks), ticks))

fn = bc.make_policy(P)
ticks = []
for s in SEEDS:
    r = run_eval_episode(fn, n_agents=5, seed=s, horizon=H, stop_on_death=True,
                         reset_fn=bc.reset_memory)
    ticks.append(r["steps"])
print("  %-16s median=%7.0f mean=%7.0f ticks=%s" % ("V0 deployed", st.median(ticks), st.mean(ticks), ticks))

print("\nFOOD-SUPPLY DECAY (environment.py:738-740: tree_spawn_chance = 0.5**(time/300))")
for t in (0, 150, 300, 600, 750, 1500, 3000):
    print("  t=%5ds  fruit production = %6.2f%% of opening rate" % (t, 100 * 0.5 ** (t / 300.0)))