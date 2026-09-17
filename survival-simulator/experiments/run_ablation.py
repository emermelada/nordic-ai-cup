"""Controlled-component ablation on the FINAL winner params.
Toggles each gate on/off (default all on = full controller). Reports mean score over seeds.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bench import multi_eval, summarize
from best_controller import make_policy, DEFAULT_PARAMS

H = 3500
SEEDS = [100, 200, 300]

with open("best_controller/params.json") as f:
    BASE = json.load(f)


def full(**tw):
    P = dict(DEFAULT_PARAMS)
    P.update(BASE)
    P.update(tw)
    return P


ablations = [
    ("FULL", full()),
    ("no_predator", full(use_predator=False)),
    ("no_flee", full(use_flee=False)),
    ("no_fruit", full(use_fruit=False)),
    ("no_wall", full(use_wall=False)),
    ("no_disperse", full(use_disperse=False)),
    ("no_explore", full(use_explore=False)),
    ("no_repro", full(use_repro=False)),
    ("no_energy", full(use_energy=False)),
]

for name, P in ablations:
    r = multi_eval(make_policy(P), seeds=SEEDS, horizon=H)
    s = summarize(r, name)
    print(f"{name:12s} mean={s['mean']:.1f} fruit={s['fruits']:.0f} pred={s['predated']:.1f} "
          f"final={s['final_agents']:.1f} alive={s['alive']}/{s['n']}")