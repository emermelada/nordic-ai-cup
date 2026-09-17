"""Long-horizon determinism probe: one seed at 16000 ticks, deployed params."""
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
with open(os.path.join(HERE, "best_controller", "params.json")) as f:
    P.update(json.load(f))

fn = bc.make_policy(P)
for s in (300, 700):
    r = run_eval_episode(fn, n_agents=5, seed=s, horizon=16000, stop_on_death=True, trace=False)
    print("seed=%-5d steps=%6d score=%9.3f fruit=%5d pred=%4d spawns=%4d"
          % (s, r["steps"], r["score"], r["fruits_eaten"], r["predated"], r["spawns"]))