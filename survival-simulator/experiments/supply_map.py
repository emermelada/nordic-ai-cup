#!/usr/bin/env python3
"""supply_map.py - measure each seed's POLICY-INDEPENDENT food supply.

Why: the 400-seed baseline says ~25% of hive runs die before t=700 s, and the losers show a low
energy/food-income state at t=200-300. Before designing any rescue, one must know whether that state
is imposed by the SEED (sparse fruit around the start region -> no policy can fix it) or produced by
hive's own choices (fixable). Fruit spawning depends only on trees, biome and time - never on agent
actions - so it can be measured with the agents idling.

Runs the real simulator on each seed for H ticks with NO agent actions (agents idle and starve out) and
records, per 10 s bucket: fruit count, total fruit energy on the map, tree count, and the same for the
8 biome types present. Output is one JSON record per seed, same resumable jsonl pattern as traj_map.py.

  PYTHONHASHSEED=0 /opt/nacv/bin/python supply_map.py --seeds 1-400 --workers 36 --out /opt/nac_traj/sup
"""
import argparse
import json
import os
import random
import sys
import time

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


def run_seed(seed, horizon, bucket):
    from src.core import SimulationCore

    random.seed(seed)
    np.random.seed(seed)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    env = core.env
    rows = []
    acc = {"n_fruit": 0.0, "fruit_E": 0.0, "n_tree": 0.0, "n_agents": 0.0}
    nb = 0
    for i in range(horizon):
        core.step([])                       # no actions: agents idle (policy-independent supply)
        acc["n_fruit"] += len(env.fruits)
        acc["fruit_E"] += sum(float(f.energy) for f in env.fruits)
        acc["n_tree"] += len(env.trees)
        acc["n_agents"] += len(env.agents)
        nb += 1
        if (i + 1) % bucket == 0:
            rows.append({"t": (i + 1) / 10.0, "n_fruit": acc["n_fruit"] / nb,
                         "fruit_E": acc["fruit_E"] / nb, "n_tree": acc["n_tree"] / nb,
                         "n_agents": acc["n_agents"] / nb})
            acc = {"n_fruit": 0.0, "fruit_E": 0.0, "n_tree": 0.0, "n_agents": 0.0}
            nb = 0
    return {"seed": seed, "T": horizon, "buckets": rows}


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
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--bucket", type=int, default=100)
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
    print("[supply] %d seeds to run (%d done)" % (len(seeds), len(done)), flush=True)
    t0 = time.time()
    ctx = get_context("spawn")
    with open(raw, "a") as f, ctx.Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_run, [(s, args.horizon, args.bucket) for s in seeds]), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if k % 25 == 0:
                print("[supply] %d/%d (%.1f s)" % (k, len(seeds), time.time() - t0), flush=True)
    print("[supply] DONE %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
