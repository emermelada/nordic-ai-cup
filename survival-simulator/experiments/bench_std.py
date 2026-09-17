"""STANDARDIZED evaluation harness for the Survival Simulator controller (Experiment Rules 1-6).

One harness, one horizon, one metric definition, disjoint seed sets, full telemetry.

Usage:
    python bench_std.py <configs.json> <horizon> <seeds>
      <seeds> = "train" | "eval" | comma list, e.g. 1000,1100,1200

configs.json: [{"tag": "...", "params": {...}}, ...]   (params merge over best_controller/params.json)

Rules enforced:
  * merges over experiments/best_controller/params.json (the source of truth) as baseline
  * reports mean AND median AND std AND min/max (never a single lucky seed)
  * train/eval seed sets are DISJOINT and named
  * records horizon + seeds in every result line, so experiments can't be silently compared
  * resets policy module state per episode (order-invariant) and imports THIS directory's
    best_controller.py, not the repo-root deployment copy
  * records ms/tick (latency proxy) and mean population per config
"""
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# SINGLE SOURCE OF TRUTH: there is exactly ONE controller -- <repo>/best_controller.py, the same
# file the Dockerfile ships and the VPS serves. experiments/ deliberately holds NO copy; this
# directory is only for measurement code. `tools/check_controller.py` enforces that invariant.
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

if not hasattr(bc, "reset_memory"):
    raise SystemExit("FATAL: imported a controller lacking reset_memory (%s). Expected %s"
                     % (getattr(bc, "__file__", "?"), os.path.join(REPO, "best_controller.py")))
if os.path.realpath(getattr(bc, "__file__", "")) != os.path.realpath(os.path.join(REPO, "best_controller.py")):
    raise SystemExit("FATAL: canonical controller is shadowed by %s (a stale duplicate?)" % bc.__file__)

PARAMS_PATH = os.path.join(REPO, "best_controller", "params.json")
RESULTS_PATH = os.path.join(HERE, "bench_std_results.jsonl")
TRAIN_SEEDS = list(range(100, 900, 100))       # 100..800   - tuning only
EVAL_SEEDS = list(range(1000, 1800, 100))      # 1000..1700 - HELD OUT, never tuned on
STD_HORIZON = 16000                            # ~ the grader's 600s budget at ~37 ms/tick
N_AGENTS = 5


def baseline():
    P = dict(DEFAULT_PARAMS)
    try:
        with open(PARAMS_PATH) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def agg(xs):
    xs = list(xs)
    return {"mean": round(st.mean(xs), 1), "median": round(st.median(xs), 1),
            "std": round(st.pstdev(xs), 1), "min": min(xs), "max": max(xs), "n": len(xs)}


def evaluate(overrides, seeds, horizon, tag=""):
    P = baseline()
    P.update(overrides or {})
    fn = bc.make_policy(P)
    ticks, scores, fruits, preds, spawns, finals, peaks, walls = [], [], [], [], [], [], [], []
    final_trace = []
    for s in seeds:
        t0 = time.perf_counter()
        r = run_eval_episode(fn, n_agents=N_AGENTS, seed=s, horizon=horizon,
                             stop_on_death=True, trace=True, reset_fn=bc.reset_memory)
        walls.append(time.perf_counter() - t0)
        ticks.append(r["steps"]); scores.append(r["score"]); fruits.append(r["fruits_eaten"])
        preds.append(r["predated"]); spawns.append(r["spawns"]); finals.append(r["final_agents"])
        tr = r["traces"] or [{"n": 0}]
        peaks.append(max(x["n"] for x in tr))
        final_trace.append({"seed": s, "ticks": r["steps"], "traces": tr})

    total_ticks = sum(ticks)
    ms_per_tick = round(1000.0 * sum(walls) / max(1, total_ticks), 3)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "horizon": horizon,
           "seeds": list(seeds), "params": P,
           "ticks": agg(ticks), "score": agg(scores),
           "fruit_mean": round(st.mean(fruits), 1), "pred_mean": round(st.mean(preds), 1),
           "spawns_mean": round(st.mean(spawns), 1), "final_mean": round(st.mean(finals), 1),
           "pop_peak_mean": round(st.mean(peaks), 1), "ms_per_tick": ms_per_tick,
           "raw_ticks": [int(x) for x in ticks], "trace": final_trace}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")

    print("%-24s horizon=%d seeds=%d" % (tag, horizon, len(seeds)))
    print("   survival ticks: mean=%7.1f median=%7.1f std=%7.1f min=%d max=%d"
          % (rec["ticks"]["mean"], rec["ticks"]["median"], rec["ticks"]["std"],
             rec["ticks"]["min"], rec["ticks"]["max"]))
    print("   score:          mean=%7.1f median=%7.1f std=%7.1f"
          % (rec["score"]["mean"], rec["score"]["median"], rec["score"]["std"]))
    print("   fruit=%.1f pred=%.1f spawns=%.1f pop_peak=%.1f final=%.1f  ms/tick=%.3f"
          % (rec["fruit_mean"], rec["pred_mean"], rec["spawns_mean"], rec["pop_peak_mean"],
             rec["final_mean"], ms_per_tick))
    print("   per-seed ticks: %s" % rec["raw_ticks"], flush=True)
    return rec


if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1]))
    H = int(sys.argv[2]) if len(sys.argv) > 2 else STD_HORIZON
    arg = sys.argv[3] if len(sys.argv) > 3 else "eval"
    if arg == "train":
        seeds = TRAIN_SEEDS
    elif arg == "eval":
        seeds = EVAL_SEEDS
    else:
        seeds = [int(x) for x in arg.split(",")]
    print("BENCH-STD horizon=%d seeds=%s configs=%d controller=%s"
          % (H, seeds, len(cfgs), bc.__file__))
    for c in cfgs:
        evaluate(c.get("params", {}), seeds, H, c["tag"])
    print("results appended -> %s" % RESULTS_PATH)