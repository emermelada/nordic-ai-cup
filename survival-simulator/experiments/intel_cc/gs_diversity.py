#!/usr/bin/env python3
"""gs_diversity.py - can our breeding gate collapse the colony's genetic diversity?

COMPETITOR CLAIM BEING TESTED: their selection rule ("within 0.4 of the best") retired 21 of 23
agents at t=240, collapsing the colony onto a few elites and destroying exploration; they proposed
gating against the 75th percentile instead of the single best.

WHAT WE HAVE (read from the live artifact):
  * our gate is ALREADY a fraction-of-the-living-fleet gate: rank(genome utility) <= gs_topk, re-ranked
    every epoch over a TTL-pruned table, so it cannot retire a fixed cohort at t=240. Deployed
    gs_topk=3.
  * guards already exist: gpop <= gs_rescue_pop (never gate the relay) and len(alive) < gs_min_known
    (no gating while the ranking is uninformative).
  * their proposed fix IS implemented: gs_mode=2 = breed if own utility > the LIVING fleet's mean
    (~the 50th percentile, strictly weaker than their 75th).
  * what is NOT known is the DIVERSITY cost. Measured already: gs_topk=3 cuts births to 91.6-99.9
    per run against 173.9-175.7 without it (gsconf/gsconf2 ledgers) - the same "fewer births, less
    exploration" mechanism they describe.

This measures the diversity directly, from the SAME genome table the gate itself uses
(controller module global `_TRAITS`, populated from the agent's own DTO on every call), sampled
through the episode:
    n_living        living genomes known to the table at the sample
    distinct        distinct rounded genome signatures among them
    pair_d          mean normalised pairwise L1 distance between living genomes (0 = one clone)
    top1_share      share of the living fleet held by the single most common signature
    max_e / vis     mean max_energy and vision_range of the living fleet (ratchet evidence)

Usage:
  ./gs_diversity.py --seeds 700-711 --horizon 12000 --workers 8 --out gs_div.json
"""
import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts live in experiments/intel_cc/, so walk UP until the repo root (the dir holding src/) is
# found rather than assuming a fixed depth
ROOT = HERE
while ROOT != "/" and not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
for p in (HERE, ROOT, os.path.join(ROOT, "experiments")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

SERVED_CTRL = os.path.join(HERE, "served_controller_252f0ba1.py")
SERVED_PARAMS = os.path.join(HERE, "served_params_252f0ba1.json")
ARMS = [("GS_TOPK3_live", {}),
        ("GS_MODE2_mean", {"gs_mode": 2.0}),
        ("GS_OFF", {"genome_select": 0.0})]
SAMPLE_EVERY = 500
CAPS = (400.0, math.pi / 2, 100.0, 1000.0, 20.0, 40.0)   # the sim's trait caps, for normalisation


def _load_controller(path):
    import importlib.util
    name = "ctrl_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sample_env(agents):
    """Summarise the TRUE living fleet's genome diversity straight from the simulator entities.

    Deliberately NOT read from the controller's `_TRAITS` table: that table only exists when
    genome_select > 0, so it could not measure the no-selection control at all, and it is a TTL
    proxy (agents unseen for gs_ttl epochs drop out) rather than the real population. The entity
    stores vision_range/vision_angle under the ENTITY names vision_radius/cone_angle."""
    alive = []
    for a in agents:
        alive.append([float(getattr(a, "vision_radius", 0.0) or 0.0),
                      float(getattr(a, "cone_angle", 0.0) or 0.0),
                      float(getattr(a, "hearing_radius", 0.0) or 0.0),
                      float(getattr(a, "max_energy", 0.0) or 0.0),
                      float(getattr(a, "speed", 0.0) or 0.0),
                      float(getattr(a, "sprint_speed", 0.0) or 0.0)])
    n = len(alive)
    if n == 0:
        return None
    sigs = {}
    for v in alive:
        s = tuple(round(x, 2) for x in v)
        sigs[s] = sigs.get(s, 0) + 1
    norm = [[v[i] / CAPS[i] for i in range(6)] for v in alive]
    if n > 24:
        idx = list(range(0, n, max(1, n // 24)))[:24]
        norm = [norm[i] for i in idx]
    tot = cnt = 0.0
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            tot += sum(abs(norm[i][k] - norm[j][k]) for k in range(6)) / 6.0
            cnt += 1
    return {"n_living": n, "distinct": len(sigs), "top1_share": max(sigs.values()) / n,
            "pair_d": (tot / cnt) if cnt else 0.0,
            "max_e": sum(v[3] for v in alive) / n,
            "vis": sum(v[0] for v in alive) / n}


def run_episode(task):
    arm, seed, horizon, workers_unused = task
    from src.core import SimulationCore
    from env_wrapper import make_action
    ovr = dict(ARMS)[arm]

    random.seed(seed)
    np.random.seed(seed)
    mod = _load_controller(SERVED_CTRL)
    mod.reset_memory()
    P = dict(mod.DEFAULT_PARAMS)
    P.update(json.load(open(SERVED_PARAMS)))
    P.update(ovr)
    policy = mod.make_policy(P)

    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    ttl = int(P.get("gs_ttl", 60) or 60)
    samples = []
    births = 0
    last_score = 0.0
    steps = 0
    for i in range(horizon):
        steps = i + 1
        live = list(core.env.agents)
        before = len(live)
        acts = []
        for a in live:
            s = core.env.get_agent_state(a.agent_id)
            if s is None:
                continue
            acts.append((a.agent_id, make_action(s, policy(s))))
        core.step(acts)
        after = len(core.env.agents)
        if after > before:
            births += after - before
        if steps % SAMPLE_EVERY == 0:
            sm = _sample_env(core.env.agents)
            if sm:
                samples.append(sm)
        if after == 0:
            break

    def avg(k):
        v = [s[k] for s in samples]
        return sum(v) / len(v) if v else float("nan")

    early = samples[:3]
    late = samples[-3:]
    def avg_of(ss, k):
        v = [s[k] for s in ss]
        return sum(v) / len(v) if v else float("nan")

    return {"arm": arm, "seed": seed, "steps": steps, "score": round(core.env.score, 3),
            "births": births, "samples": len(samples),
            "n_living": avg("n_living"), "distinct": avg("distinct"),
            "top1_share": avg("top1_share"), "pair_d": avg("pair_d"),
            "ratio": avg("distinct") / max(1e-9, avg("n_living")),
            "max_e_early": avg_of(early, "max_e"), "max_e_late": avg_of(late, "max_e"),
            "vis_early": avg_of(early, "vis"), "vis_late": avg_of(late, "vis")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="700-711")
    ap.add_argument("--horizon", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", default=",".join(a for a, _ in ARMS))
    ap.add_argument("--out", default="gs_div.json")
    args = ap.parse_args()
    a0, a1 = args.seeds.split("-")
    seeds = list(range(int(a0), int(a1) + 1))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    jobs = [(a, s, args.horizon, None) for a in arms for s in seeds]
    print(f"arms={arms} seeds={len(seeds)} horizon={args.horizon} jobs={len(jobs)} "
          f"workers={args.workers} ctrl={os.path.basename(SERVED_CTRL)}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_episode, jobs):
            rows.append(r)
            print("  %-17s seed=%-5d steps=%-6d score=%-7s births=%-4d n_liv=%-6.1f distinct=%-6.1f "
                  "ratio=%-6.3f top1=%-6.3f pair_d=%-6.3f maxE %.0f->%.0f vis %.0f->%.0f"
                  % (r["arm"], r["seed"], r["steps"], r["score"], r["births"], r["n_living"],
                     r["distinct"], r["ratio"], r["top1_share"], r["pair_d"],
                     r["max_e_early"], r["max_e_late"], r["vis_early"], r["vis_late"]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)

    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    print("\n=== SUMMARY (means over seeds; ratio = distinct genomes / living genomes) ===")
    print("%-17s %8s %8s %8s %8s %8s %8s %9s %9s" %
          ("arm", "steps", "births", "n_living", "distinct", "ratio", "top1", "maxE_late", "vis_late"))
    for arm in arms:
        rs = by.get(arm, [])
        if not rs:
            continue
        m = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
        print("%-17s %8.0f %8.1f %8.1f %8.1f %8.3f %8.3f %9.0f %9.1f" %
              (arm, m("steps"), m("births"), m("n_living"), m("distinct"), m("ratio"),
               m("top1_share"), m("max_e_late"), m("vis_late")))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
