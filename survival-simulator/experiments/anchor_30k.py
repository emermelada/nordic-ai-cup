import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env_wrapper import run_eval_episode, HORIZON
from policies import random_policy, dummy_policy, heuristic_policy

def bench(name, fn, seeds, stop):
    res = []
    for s in seeds:
        t0=time.time()
        r = run_eval_episode(fn, n_agents=5, seed=s, horizon=HORIZON, stop_on_death=stop)
        r["t"]=time.time()-t0; res.append(r)
    sc=[r["score"] for r in res]
    print(f"{name:12s} fixed={np.mean(sc):.1f}±{np.std(sc):.1f}  surv.eqs={[r['steps'] for r in res]}  fruit={np.mean([r['fruits_eaten'] for r in res]):.0f}  pred={np.mean([r['predated'] for r in res]):.1f}  alive={sum(r['alive'] for r in res)}/5  {np.mean([r['t'] for r in res]):.0f}s/eps")
    return res

SEED = [100, 200, 300, 400, 500]
print(f"horizon={HORIZON} steps")
print("=== FIXED-horizon grading proxy ===")
bench("random", random_policy, SEED, stop=False)
bench("heuristic", heuristic_policy, SEED, stop=False)
print("=== SURVIVORSHIP grading proxy (episode ends when team wiped) ===")
bench("heuristic", heuristic_policy, SEED, stop=True)