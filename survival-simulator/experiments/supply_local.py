#!/usr/bin/env python3
"""supply_local.py - policy-free LOCAL food supply at the agents' starting positions.

The map-wide supply test (supply_map.py) explains only ~7% of the outcome variance, but the winning
runs SEE ~25% more fruit per agent than the losers. Seeing is a function of where the agents are, so the
question splits in two:

  (a) the starting neighbourhood is seed-determined and policy-free -> if winners start in fruit-denser
      areas, the divergence is environmental and no behaviour can fix it
  (b) the starting neighbourhoods are equivalent -> the winners POSITION themselves better, and the
      divergence is behaviour, i.e. fixable

This measures (a) directly: build the world for each seed, take the 5 starting agent positions, and
measure fruit/tree density and distances around those positions BEFORE any agent acts. No stepping, so
it is seconds per seed, not minutes.

  PYTHONHASHSEED=0 /opt/nacv/bin/python supply_local.py --seeds 1-400 --workers 16 --out /opt/nac_traj/suploc
"""
import argparse
import json
import math
import os
import random
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from multiprocessing import get_context  # noqa: E402

RADII = (100.0, 200.0, 400.0, 800.0)


def run_seed(seed, horizon, bucket):
    from src.core import SimulationCore

    random.seed(seed)
    np.random.seed(seed)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    env = core.env
    start = [(float(a.x), float(a.y)) for a in env.agents]
    fruits = [(float(f.x), float(f.y), float(f.energy)) for f in env.fruits]
    trees = [(float(t.x), float(t.y)) for t in env.trees]
    # the pre-behaviour choice set: nearest fruit/tree distance, and density within radii
    out = {"seed": seed, "n_start": len(start),
           "map_fruit": len(fruits), "map_fruit_E": sum(f[2] for f in fruits), "map_trees": len(trees)}
    for r in RADII:
        nf, nfe, nt, df, dt = [], [], [], [], []
        for (ax, ay) in start:
            cf = cfE = ct = 0.0
            for (fx, fy, fe) in fruits:
                d = math.hypot(fx - ax, fy - ay)
                if d <= r:
                    cf += 1.0
                    cfE += fe
                    df.append(d)
            for (tx, ty) in trees:
                if math.hypot(tx - ax, ty - ay) <= r:
                    ct += 1.0
                    dt.append(math.hypot(tx - ax, ty - ay))
            nf.append(cf)
            nfe.append(cfE)
            nt.append(ct)
        out["r%d_fruit" % int(r)] = sum(nf) / len(nf)
        out["r%d_fruit_E" % int(r)] = sum(nfe) / len(nfe)
        out["r%d_tree" % int(r)] = sum(nt) / len(nt)
        out["r%d_d_fruit" % int(r)] = (sum(df) / len(df)) if df else 9999.0
        out["r%d_d_tree" % int(r)] = (sum(dt) / len(dt)) if dt else 9999.0
    # pairwise spread of the start positions (how clustered the fleet begins)
    if len(start) > 1:
        ds = [math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(start) for b in start[i + 1:]]
        out["start_pair_dist"] = sum(ds) / len(ds)
    else:
        out["start_pair_dist"] = 0.0
    return out


def _run(job):
    return run_seed(job[0], job[1], job[2])


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--bucket", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "raw.jsonl")
    done = set()
    if os.path.exists(raw):
        for line in open(raw):
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass
    seeds = [s for s in parse_seeds(args.seeds) if s not in done]
    ctx = get_context("spawn")
    with open(raw, "a") as f, ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_run, [(s, 0, 1) for s in seeds]):
            f.write(json.dumps(res) + "\n")
            f.flush()
    print("[supply_local] done, %d new (total %d)" % (len(seeds), len(done) + len(seeds)), flush=True)


if __name__ == "__main__":
    main()
