"""Identity check, one PROCESS PER MODULE (the only fair comparison: this project's own notes record
residual in-process order-dependence between consecutive episodes, which made the same seed score
differently depending on which seeds ran before it).

Usage: python _wb_ident2.py <module_path> <seed1,seed2,...> [horizon]
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from env_wrapper import run_eval_episode  # noqa: E402

path = sys.argv[1]
seeds = [int(x) for x in sys.argv[2].split(",")]
horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 6000

spec = importlib.util.spec_from_file_location("bc_x", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["bc_x"] = mod
spec.loader.exec_module(mod)

H1 = json.load(open(os.path.join(HERE, "wH1_winEARN_liveSPEND.json")))
P = dict(mod.DEFAULT_PARAMS)
P.update(H1)
fn = mod.make_policy(P)
tag = os.path.basename(path)
for sd in seeds:
    r = run_eval_episode(fn, n_agents=5, seed=sd, horizon=horizon,
                         stop_on_death=True, reset_fn=mod.reset_memory)
    print(f"{tag} seed={sd} steps={r['steps']} fruits={r['fruits_eaten']} spawns={r['spawns']} "
          f"pred={r['predated']}", flush=True)
