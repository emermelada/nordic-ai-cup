import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env_wrapper import run_eval_episode, HORIZON
from policies import heuristic_policy, heuristic_v2, random_policy, dummy_policy

SEEDS = [100, 200, 300, 400, 500]

def bench(name, fn):
    res = [run_eval_episode(fn, n_agents=5, seed=s, horizon=HORIZON) for s in SEEDS]
    sc = [r["score"] for r in res]; st = [r["steps"] for r in res]
    fr = [r["fruits_eaten"] for r in res]; pr = [r["predated"] for r in res]
    alive = sum(r["alive"] for r in res)
    print(f"{name:12s} score={np.mean(sc):.0f}±{np.std(sc):.0f}  steps={np.mean(st):.0f}  fruit={np.mean(fr):.0f}  pred={np.mean(pr):.1f}  alive={alive}/5")
    return np.mean(sc), np.std(sc)

bench("random", random_policy)
bench("dummy", dummy_policy)
bench("heuristic", heuristic_policy)
bench("heuristic_v2", heuristic_v2)