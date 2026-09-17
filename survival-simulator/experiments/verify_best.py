import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from env_wrapper import run_eval_episode, HORIZON
from policies import heuristic_policy
from best_controller import best_controller

SEEDS = [100, 200, 300]
H = 8000  # reduced for speed; survivorship + collapse ticks

def bench(name, fn):
    res = [run_eval_episode(fn, n_agents=5, seed=s, horizon=H, stop_on_death=True) for s in SEEDS]
    sc = [r["score"] for r in res]; tick = [r["steps"] for r in res]
    print(f"{name:16s} surv.score mean={np.mean(sc):7.1f}  collapse_ticks={tick}  median={int(np.median(tick))}  fruit={np.mean([r['fruits_eaten'] for r in res]):.0f}  pred={np.mean([r['predated'] for r in res]):.1f}")

print(f"horizon={H}, seeds={SEEDS}, survivorship")
bench("heuristic(v1)", heuristic_policy)
bench("best_controller", best_controller)