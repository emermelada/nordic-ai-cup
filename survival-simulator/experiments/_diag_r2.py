"""Diagnose why batch_eval round-2 crashed: run one config at short horizon."""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

P = dict(DEFAULT_PARAMS)
try:
    with open(os.path.join(HERE, "best_controller", "params.json")) as f:
        P.update(json.load(f))
except Exception as e:
    print("params load error:", e)

overrides = {"repro_urgency": 0.0, "forage_nearest": 0.0}
P.update(overrides)
print("param keys added:", sorted(set(overrides) - set(DEFAULT_PARAMS)))
print("total params:", len(P))

try:
    fn = bc.make_policy(P)
    print("make_policy OK")
except Exception:
    print("make_policy FAILED")
    traceback.print_exc()
    sys.exit(1)

try:
    r = run_eval_episode(fn, n_agents=5, seed=100, horizon=200, stop_on_death=True)
    print("eval OK:", {k: r[k] for k in ("steps", "score") if k in r})
except Exception:
    print("eval FAILED")
    traceback.print_exc()
    sys.exit(1)