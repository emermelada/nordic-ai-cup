"""Order-invariance test: seed 300 must give the SAME survival whether it runs first or after
other seeds. Before reset_memory it gave 9362 / 7581 / 11873 depending on position."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

assert hasattr(bc, "reset_memory"), "wrong module: %s" % bc.__file__
print("controller:", bc.__file__)
P = dict(DEFAULT_PARAMS)
with open(os.path.join(HERE, "best_controller", "params.json")) as f:
    P.update(json.load(f))
fn = bc.make_policy(P)
H = 8000


def run(s):
    r = run_eval_episode(fn, n_agents=5, seed=s, horizon=H, stop_on_death=True,
                         reset_fn=bc.reset_memory)
    return r["steps"]


a = run(300)
print("A) seed 300 alone      -> %d" % a)
run(100); run(200)
b = run(300)
print("B) after 100,200 -> 300 -> %d" % b)
run(700); run(500)
c = run(300)
print("C) after 700,500 -> 300 -> %d" % c)
print("RESULT: %s" % ("ORDER-INVARIANT" if a == b == c else "STILL ORDER-DEPENDENT"))