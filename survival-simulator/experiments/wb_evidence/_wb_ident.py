"""Decisive identity / determinism probe: is the H1-vs-H1 difference caused by my edit, or is the
harness itself order/process dependent? Run each module TWICE on the same seed, in one process."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from env_wrapper import run_eval_episode  # noqa: E402


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mods = {}
for nm, path in [("orig", "/tmp/wB/best_controller_ORIG.py"),
                 ("new", os.path.join(ROOT, "best_controller.py"))]:
    mods[nm] = load_mod(nm, path)

H1 = json.load(open(os.path.join(HERE, "wH1_winEARN_liveSPEND.json")))
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 102
horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

for label in ["orig", "new", "orig", "new"]:
    m = mods[label]
    P = dict(m.DEFAULT_PARAMS)
    P.update(H1)
    r = run_eval_episode(m.make_policy(P), n_agents=5, seed=seed, horizon=horizon,
                         stop_on_death=True, reset_fn=m.reset_memory)
    print(f"{label:5s} seed={seed} steps={r['steps']} fruits={r['fruits_eaten']} spawns={r['spawns']} pred={r['predated']}", flush=True)
