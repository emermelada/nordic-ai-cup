"""Environment-shape ROBUSTNESS check (final-eval generalisation).

The real grader does ONE run on an UNSEEN dataset (README: "one final evaluation per use case on a
DIFFERENT dataset"), so a config that wins on 1600x1200 only is worthless. This script re-runs the
SAME controller params under several world shapes (env_width/env_height/n_agents) using the
standard episode runner (run_eval_episode with reset_fn=bc.reset_memory), and reports
mean/median/std/min/p5 for each (config, shape) cell.

Usage:
    python robust_env.py <spec.json>
spec.json:
    {"horizon": 16000, "seeds": [100..800], "out": "robust_env_results.jsonl",
     "shapes": [{"name":"default","w":1600,"h":1200,"n":5}, ...],
     "configs": [{"tag":"...","params":{...}}, ...]}   # params merge over best_controller/params.json

Same import discipline as bench_std.py: experiments/ first, guard against the repo-root deployment
copy, and reset_fn=bc.reset_memory so episodes are order-invariant.
"""
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

if not hasattr(bc, "reset_memory"):
    raise SystemExit("FATAL: imported the wrong best_controller (%s). Expected %s"
                     % (getattr(bc, "__file__", "?"), os.path.join(HERE, "best_controller.py")))

PARAMS_PATH = os.path.join(HERE, "best_controller", "params.json")


def baseline():
    P = dict(DEFAULT_PARAMS)
    try:
        with open(PARAMS_PATH) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def agg(xs):
    xs = list(xs)
    return {"mean": round(st.mean(xs), 1), "median": round(st.median(xs), 1),
            "std": round(st.pstdev(xs), 1), "min": min(xs), "max": max(xs),
            "p5": round(pct(xs, 0.05), 1), "p20": round(pct(xs, 0.2), 1), "n": len(xs)}


def evaluate(overrides, seeds, horizon, shape):
    P = baseline()
    P.update(overrides or {})
    fn = bc.make_policy(P)
    ticks, spawns, finals, peaks, walls = [], [], [], [], []
    for s in seeds:
        t0 = time.perf_counter()
        r = run_eval_episode(fn, n_agents=shape["n"], seed=s, horizon=horizon,
                             env_width=shape["w"], env_height=shape["h"],
                             stop_on_death=True, trace=True, reset_fn=bc.reset_memory)
        walls.append(time.perf_counter() - t0)
        ticks.append(r["steps"]); spawns.append(r["spawns"]); finals.append(r["final_agents"])
        tr = r["traces"] or [{"n": 0}]
        peaks.append(max(x["n"] for x in tr))
    return {"ticks": agg(ticks), "raw_ticks": [int(x) for x in ticks],
            "spawns_mean": round(st.mean(spawns), 1), "final_mean": round(st.mean(finals), 1),
            "pop_peak_mean": round(st.mean(peaks), 1),
            "ms_per_tick": round(1000.0 * sum(walls) / max(1, sum(ticks)), 3), "params": P}


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    H = int(spec.get("horizon", 16000))
    seeds = spec["seeds"]
    out = os.path.join(HERE, spec.get("out", "robust_env_results.jsonl"))
    print("ROBUST-ENV horizon=%d seeds=%s controller=%s" % (H, seeds, bc.__file__))
    rows = []
    for shape in spec["shapes"]:
        for c in spec["configs"]:
            rec = evaluate(c.get("params", {}), seeds, H, shape)
            row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": c["tag"], "shape": shape,
                   "horizon": H, "seeds": list(seeds), **rec}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            t = rec["ticks"]
            print("%-22s shape=%-14s mean=%7.1f median=%7.1f std=%6.1f min=%6d p5=%7.1f "
                  "max=%6d sp=%.1f pop=%.1f ms/t=%.2f"
                  % (c["tag"], shape["name"], t["mean"], t["median"], t["std"], t["min"],
                     t["p5"], t["max"], rec["spawns_mean"], rec["pop_peak_mean"],
                     rec["ms_per_tick"]), flush=True)
    print("results appended -> %s" % out)
