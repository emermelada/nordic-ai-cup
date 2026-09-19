#!/usr/bin/env python3
"""steer_diag.py - does the deployed controller WASTE COVERAGE by re-selecting its foraging
target every tick?

Tests the "persistent target commitment beats per-tick reactive steering" hypothesis on the
DEPLOYED controller. Instrumentation only: the policy is called exactly as served.

WHAT IS MEASURED (per agent per tick, aggregated into 100-tick buckets)
  straightness   net displacement / path length over the bucket. 1.0 = dead straight.
  flip_rate      share of ticks where the ABSOLUTE heading turned more than 90 deg
  away_frac      share of fruit-VISIBLE ticks whose heading points >90 deg away from the nearest
                 visible fruit (fruit abandoned while visible)
  fv_hist        distribution of VISIBLE FRUIT COUNT - the decisive quantity: thrashing requires
                 >=2 visible fruits, and the fleet mean is 0.21-0.48, so the hypothesis lives or
                 dies on whether the distribution is bimodal.
  thrash_rate    among ticks with >=2 visible fruits: share where the SECOND-nearest fruit is
                 within 15% of the nearest (i.e. a near-tie that re-selection can flip on)
  cov_per_e      net ground covered per unit of movement energy = the quantity oscillation
                 actually destroys (environment.py charges the COMMANDED distance, so oscillation
                 costs no extra energy - it costs coverage per energy).

  python3 steer_diag.py --seeds 5000-5039 --horizon 18000 --workers 60 --out steer.jsonl
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import random  # noqa: E402
import numpy as np  # noqa: E402

BUCKET = 100
WALK_COST = 0.05          # environment.py:501
SPRINT_COST = 0.5         # environment.py:502


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part and "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_episode(args):
    seed, horizon, params_path = args
    import best_controller as bc
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    P = dict(bc.DEFAULT_PARAMS)
    with open(params_path) as f:
        P.update(json.load(f))

    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    policy = bc.make_policy(P)

    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)

    speed_default = 10.0
    B = {}

    def acc(t):
        b = t // BUCKET
        if b not in B:
            B[b] = {"t0": b * BUCKET, "ticks": 0, "n": 0,
                    "path": 0.0, "netx": 0.0, "nety": 0.0,
                    "flips": 0, "hdg_n": 0,
                    "fvis_n": 0, "away": 0,
                    "fv_hist": {}, "thrash_n": 0, "thrash_tot": 0,
                    "cmd": 0.0, "e_move": 0.0, "pop": 0,
                    "spin": 0.0, "abs_turn": 0.0}
        return B[b]

    prev = {}      # agent_id -> (abs heading, x, y, e_move)
    for i in range(horizon):
        live = list(core.env.agents)
        if not live:
            break
        states = []
        for a in live:
            st = core.env.get_agent_state(a.agent_id)
            if st:
                st["_x"] = float(a.x)
                st["_y"] = float(a.y)
                st["_dir"] = float(a.direction)
                states.append(st)

        b = acc(i)
        b["ticks"] += 1
        b["pop"] += len(states)
        acts = []
        for st in states:
            aid = st["agent_id"]
            obs = st.get("observations") or []
            fruits = [o for o in obs if o.get("type") == "Fruit"]
            dist, dr, turn, spawn = policy(st)
            sprint = float(st.get("sprint_speed", 20.0) or 20.0)
            spd = float(st.get("speed", speed_default) or speed_default)

            # energy the sim will charge for this commanded distance (environment.py:515-518)
            if dist <= spd:
                e_move = dist * WALK_COST
            else:
                e_move = spd * WALK_COST + (dist - spd) * SPRINT_COST

            abs_hdg = st["_dir"] + dr
            nf = len(fruits)
            b["n"] += 1
            b["cmd"] += dist
            b["e_move"] += e_move
            b["abs_turn"] += abs(turn)

            b["fv_hist"][str(min(nf, 6))] = b["fv_hist"].get(str(min(nf, 6)), 0) + 1
            if nf >= 1:
                b["fvis_n"] += 1
                nearest = min(fruits, key=lambda f: f["distance"])
                bearing = st["_dir"] + nearest["angle"]
                if abs(wrap(abs_hdg - bearing)) > math.pi / 2:
                    b["away"] += 1
                if nf >= 2:
                    ds = sorted(f["distance"] for f in fruits)
                    b["thrash_tot"] += 1
                    if ds[1] <= ds[0] * 1.15:
                        b["thrash_n"] += 1

            p = prev.get(aid)
            if p is not None:
                phdg, px, py, pe = p
                dh = abs(wrap(abs_hdg - phdg))
                b["hdg_n"] += 1
                if dh > math.pi / 2:
                    b["flips"] += 1
                b["spin"] += dh
                dx = st["_x"] - px
                dy = st["_y"] - py
                b["path"] += math.hypot(dx, dy)
                b["netx"] += dx
                b["nety"] += dy
            prev[aid] = (abs_hdg, st["_x"], st["_y"], e_move)

            acts.append((aid, ActionRequest(agent_id=aid, move_distance=dist, move_direction=dr,
                                            turn_angle=turn, spawn_agent=bool(spawn))))

        core.step(acts)
        keep = {a.agent_id for a in core.env.agents}
        prev = {k: v for k, v in prev.items() if k in keep}

    rows = []
    for bi in sorted(B):
        b = B[bi]
        n = max(1, b["n"])
        net = math.hypot(b["netx"], b["nety"])
        rows.append({
            "seed": seed, "t0": b["t0"], "ticks": b["ticks"], "pop": b["pop"] / max(1, b["ticks"]),
            "straightness": net / b["path"] if b["path"] > 0 else None,
            "net_per_1k_ticks": net / max(1, b["ticks"]) * 1000,
            "flip_rate": b["flips"] / max(1, b["hdg_n"]),
            "mean_abs_turn": b["abs_turn"] / n,
            "away_frac": b["away"] / max(1, b["fvis_n"]),
            "fvis_frac": b["fvis_n"] / n,
            "fv_hist": b["fv_hist"],
            "thrash_rate_ge2": b["thrash_n"] / max(1, b["thrash_tot"]),
            "cmd_per_tick": b["cmd"] / n,
            "e_move_per_tick": b["e_move"] / n,
            "net_per_e": net / b["e_move"] if b["e_move"] > 0 else None,
        })
    summary = {"seed": seed, "ticks": i + 1}
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="5000-5039")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "steer.jsonl"))
    ap.add_argument("--sumout", default=os.path.join(HERE, "steer_sum.json"))
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    print(f"steer_diag | {len(seeds)} seeds | horizon {args.horizon} | workers {args.workers}",
          flush=True)
    from multiprocessing import get_context
    ctx = get_context("spawn")
    jobs = [(s, args.horizon, args.params) for s in seeds]
    n = 0
    with open(args.out, "w") as fh, open(args.sumout, "w") as sh:
        with ctx.Pool(args.workers) as pool:
            for rows, summary in pool.imap_unordered(run_episode, jobs, chunksize=1):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                n += 1
                sh.write(json.dumps(summary) + "\n")
                if n % 10 == 0:
                    print(f"  {n}/{len(jobs)} episodes", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()