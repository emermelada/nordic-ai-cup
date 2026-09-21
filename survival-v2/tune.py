"""CMA-ES over Hive parameters on the fast simulator.

    python tune.py --workers 30 --seeds-per-gen 16 --popsize 12 --gens 40 --out runs/tune1

Every candidate of a generation plays the same seed batch (runs are deterministic per seed, so candidates are
compared on identical worlds); the batch rotates every generation so the search does not fit a fixed set.
The objective is the mean game score (extinctions count in full). The best mean so far is re-checked on a
held-out seed range before being written to best_params.json.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool

import cma
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import play  # noqa: E402
from hive import DEFAULT_PARAMS  # noqa: E402

# name: (low, high, log-scale)  -- the knobs with the most measured effect / least certainty
SPACE = {
    "w_hear": (0.0, 6.0, False),
    "w_energy": (0.0, 3.0, False),
    "breed_gap": (0.1, 2.0, True),
    "breed_reserve_early": (5.0, 150.0, True),
    "breed_reserve_late": (5.0, 250.0, True),
    "breed_reserve_span": (0.0, 250.0, False),
    "pop_min": (2.0, 16.0, False),
    "pop_cap_early": (15.0, 60.0, False),
    "pop_cap_mid": (6.0, 40.0, False),
    "ripe_age": (5.0, 21.0, False),
    "starve_frac": (0.02, 0.4, True),
    "dist_cost": (0.02, 0.15, True),
    "food_range": (60.0, 700.0, True),
    "food_range_hungry": (100.0, 900.0, True),
    "min_gain": (0.0, 30.0, False),
    "weak_food_w": (0.0, 1.0, False),
    "alert_close": (60.0, 200.0, False),
    "keep_dist": (95.0, 260.0, False),
    "backoff_speed": (5.0, 20.0, False),
    "scan_rate": (0.0, 0.5, False),
    "spread_pen": (0.0, 400.0, False),
    "barren_watch": (5.0, 60.0, True),
    "breed_colony_e": (40.0, 200.0, False),
    "charge_zone": (80.0, 140.0, False),
    "face_tol": (0.2, 1.5, False),
    "tree_stick": (0.0, 300.0, False),
    "slow_tree_pen": (0.0, 600.0, False),
    "alert_always": (40.0, 120.0, False),
    "camp_spacing": (20.0, 150.0, False),
    "brake_min_pop": (6.0, 40.0, False),
}
INT_KEYS = {"pop_min", "pop_cap_early", "pop_cap_mid", "brake_min_pop"}
NAMES = list(SPACE)


def to_params(z):
    """z in [0, 1]^n (CMA works on the unit cube) -> parameter dict."""
    out = {}
    for name, u in zip(NAMES, z):
        lo, hi, log = SPACE[name]
        u = min(1.0, max(0.0, float(u)))
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if log else lo + u * (hi - lo)
        out[name] = int(round(v)) if name in INT_KEYS else v
    return out


def from_params(params):
    z = []
    for name in NAMES:
        lo, hi, log = SPACE[name]
        v = float(params.get(name, DEFAULT_PARAMS[name]))
        v = min(hi, max(lo, v))
        z.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log else (v - lo) / (hi - lo))
    return z


def evaluate(pool, cands, seeds, ticks):
    jobs = [(s, to_params(z), ticks, "hive") for z in cands for s in seeds]
    rows = pool.map(play, jobs, chunksize=1)
    scores = []
    for i in range(len(cands)):
        rs = rows[i * len(seeds):(i + 1) * len(seeds)]
        scores.append(statistics.fmean(r["score"] for r in rs))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds-per-gen", type=int, default=16)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--ticks", type=int, default=30000)
    ap.add_argument("--holdout", default="5001-5032")
    ap.add_argument("--start", default="", help="json file with starting params")
    ap.add_argument("--out", default="runs/tune")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    start = dict(DEFAULT_PARAMS)
    if args.start:
        start.update(json.load(open(args.start)))
    es = cma.CMAEvolutionStrategy(from_params(start), args.sigma,
                                  {"popsize": args.popsize, "bounds": [0, 1], "seed": 1, "verbose": -9})
    rng = random.Random(7)
    a, b = (int(x) for x in args.holdout.split("-"))
    holdout = list(range(a, b + 1))
    best = (-1e9, None)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    with Pool(args.workers) as pool:
        base = evaluate(pool, [from_params(start)], holdout, args.ticks)[0]
        print(f"start params on holdout: {base:.1f}", flush=True)
        log.write(json.dumps({"gen": -1, "holdout_start": base}) + "\n")
        for gen in range(args.gens):
            t0 = time.time()
            seeds = rng.sample(range(10000, 100000), args.seeds_per_gen)
            cands = es.ask()
            # always include the incumbent mean so progress is measured on the same seeds
            scores = evaluate(pool, cands + [es.mean], seeds, args.ticks)
            es.tell(cands, [-s for s in scores[:-1]])
            mean_score = scores[-1]
            i_best = int(np.argmax(scores[:-1]))
            rec = {"gen": gen, "seeds": seeds, "mean_score": mean_score, "best_cand": scores[i_best],
                   "cands": scores[:-1], "sigma": es.sigma, "mean_params": to_params(es.mean), "secs": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"gen {gen:3d} incumbent {mean_score:7.1f} best cand {scores[i_best]:7.1f} "
                  f"median {statistics.median(scores[:-1]):7.1f} sigma {es.sigma:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if (gen + 1) % 5 == 0:
                hold = evaluate(pool, [es.mean], holdout, args.ticks)[0]
                print(f"   holdout of incumbent: {hold:.1f} (start {base:.1f})", flush=True)
                log.write(json.dumps({"gen": gen, "holdout": hold}) + "\n")
                if hold > best[0]:
                    best = (hold, to_params(es.mean))
                    json.dump({"holdout": hold, "params": best[1]}, open(os.path.join(args.out, "best_params.json"), "w"),
                              indent=1)


if __name__ == "__main__":
    main()
