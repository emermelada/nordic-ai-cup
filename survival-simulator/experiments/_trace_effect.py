"""Does enabling trace change the episode? (measurement must not perturb the measured system)
Runs the SAME seed with trace off/on, in the same process, then prints both."""
import json
import os
import sys

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

fn = bc.make_policy(P)
for trace in (False, True, False, True):
    r = run_eval_episode(fn, n_agents=5, seed=100, horizon=3000, stop_on_death=True, trace=trace)
    print("trace=%-5s steps=%5d score=%9.3f fruit=%4d pred=%3d spawns=%3d"
          % (trace, r["steps"], r["score"], r["fruits_eaten"], r["predated"], r["spawns"]))