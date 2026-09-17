"""Final validation: best_controller vs heuristic_policy on seeds {100..500} at FULL horizon 30000,
reported under BOTH grading metrics (fixed-horizon and survivorship). Writes JSON + prints table.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bench import multi_eval, failure_analysis, HORIZON
from policies import heuristic_policy
from best_controller import best_controller

SEEDS = [100, 200, 300, 400, 500]
OUT = "best_controller/final_30k.json"


def run_block(name, fn, stop):
    res = multi_eval(fn, seeds=SEEDS, horizon=HORIZON, stop=stop)
    sc = np.array([r["score"] for r in res])
    return {
        "name": name, "metric": "survivorship" if stop else "fixed",
        "scores": [round(float(s), 1) for s in sc],
        "mean": round(float(sc.mean()), 1), "median": round(float(np.median(sc)), 1),
        "std": round(float(sc.std()), 1), "min": round(float(sc.min()), 1),
        "max": round(float(sc.max()), 1),
        "fruits_eaten": [r["fruits_eaten"] for r in res],
        "predated": [r["predated"] for r in res],
        "final_agents": [r["final_agents"] for r in res],
        "steps": [r["steps"] for r in res],
        "alive": [bool(r["alive"]) for r in res],
        "latency_s_per_episode": [round(r["t"], 1) for r in res],
        "median_survival_ticks": (round(float(np.median([r["steps"] for r in res])), 0)
                                  if stop else None),
        "survival_ticks": ([r["steps"] for r in res] if stop else None),
    }


def fmt(b):
    return (f"{b['name']:14s} {b['metric'][:6]:6s} mean={b['mean']:.1f}±{b['std']:.1f} "
            f"med={b['median']:.1f} min={b['min']:.1f} max={b['max']:.1f} "
            f"fruit={np.mean(b['fruits_eaten']):.0f} pred={np.mean(b['predated']):.1f} "
            f"final={np.mean(b['final_agents']):.1f} alive={sum(b['alive'])}/5 "
            f"lat={np.mean(b['latency_s_per_episode']):.0f}s/eps")


if __name__ == "__main__":
    print(f"horizon={HORIZON} ticks, seeds={SEEDS}")
    print("=== FIXED-horizon grading ===")
    rows = []
    for name, fn in [("heuristic", heuristic_policy), ("best_controller", best_controller)]:
        b = run_block(name, fn, stop=False)
        print(fmt(b)); rows.append(b)
    print("=== SURVIVORSHIP grading (stop when team wiped) ===")
    for name, fn in [("heuristic", heuristic_policy), ("best_controller", best_controller)]:
        b = run_block(name, fn, stop=True)
        print(fmt(b)); rows.append(b)
    print("=== deaths-by-cause (seed=100, full 30k) ===")
    for name, fn in [("heuristic", heuristic_policy), ("best_controller", best_controller)]:
        fa = failure_analysis(fn, seed=100, horizon=HORIZON)
        print(f"  {name:14s} deaths={fa['deaths']} births={fa['births']} peak={fa['peak']} "
              f"fruit={fa['fruit_score']:.0f} predPen={fa['pred_penalty']:.1f} score={fa['score']:.0f} "
              f"final={fa['final_agents']} collapse_ticks={fa['collapse_ticks']}")
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)
    print("saved", OUT)