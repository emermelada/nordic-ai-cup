import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import run_eval_episode
from policies import heuristic_policy

for h in (3000, 6000):
    t0 = time.time()
    r = run_eval_episode(heuristic_policy, n_agents=5, seed=100, horizon=h)
    dt = time.time() - t0
    print(f"horizon={h}: {r}  time={dt:.2f}s  ~{h/dt:.0f} steps/s")