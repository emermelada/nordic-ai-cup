"""Parallel survival evaluator: sweeps configs x seeds across CPU cores (one process per episode).

Usage: python batch_par.py <configs.json> <horizon> <seed,seed,...> [workers]
Config file: [{"tag": "...", "params": {...}}, ...]
Prints one line per config (median/mean survival ticks, survivorship score, fruit, predated).
Appends results to batch_results.json in the experiments dir.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "best_controller", "params.json")


def deployed():
    from best_controller import DEFAULT_PARAMS
    P = dict(DEFAULT_PARAMS)
    try:
        P.update(json.load(open(PARAMS)))
    except Exception:
        pass
    return P


def run_task(task):
    tag, overrides, seed, horizon = task
    import best_controller as bc
    P = deployed(); P.update(overrides)
    fn = bc.make_policy(P)
    from env_wrapper import run_eval_episode
    r = run_eval_episode(fn, n_agents=5, seed=seed, horizon=horizon, stop_on_death=True)
    return {"tag": tag, "seed": seed, "steps": r["steps"], "score": r["score"],
            "fruit": r["fruits_eaten"], "pred": r["predated"]}


def main():
    cfgs = json.load(open(sys.argv[1]))
    H = int(sys.argv[2])
    seeds = [int(x) for x in sys.argv[3].split(",")]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    tasks = [(c["tag"], c.get("params", {}), s, H) for c in cfgs for s in seeds]
    with Pool(workers) as pool:
        res = pool.map(run_task, tasks)
    by = {}
    for r in res:
        by.setdefault(r["tag"], []).append(r)
    out = []
    for c in cfgs:
        rs = sorted(by[c["tag"]], key=lambda r: r["seed"])
        steps = [r["steps"] for r in rs]; scores = [r["score"] for r in rs]
        d = {"tag": c["tag"], "params": c.get("params", {}),
             "med_steps": float(np.median(steps)), "mean_steps": float(np.mean(steps)),
             "med_score": float(np.median(scores)), "mean_score": float(np.mean(scores)),
             "min_steps": int(np.min(steps)), "max_steps": int(np.max(steps)),
             "fruit": float(np.mean([r["fruit"] for r in rs])),
             "pred": float(np.mean([r["pred"] for r in rs])),
             "steps": steps, "scores": [round(s, 1) for s in scores]}
        out.append(d)
        print(f"{d['tag']:30s} med_ticks={d['med_steps']:7.0f} mean_ticks={d['mean_steps']:7.0f} "
              f"med_score={d['med_score']:7.1f} min={d['min_steps']:6d} max={d['max_steps']:6d} "
              f"fruit={d['fruit']:6.0f} pred={d['pred']:6.1f} ticks={steps}", flush=True)
    hist = []
    hp = os.path.join(HERE, "batch_results.json")
    if os.path.exists(hp):
        try:
            hist = json.load(open(hp))
        except Exception:
            hist = []
    hist.extend(out)
    json.dump(hist, open(hp, "w"), indent=1)


if __name__ == "__main__":
    main()