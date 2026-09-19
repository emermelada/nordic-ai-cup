#!/usr/bin/env python3
"""oracle_percept.py - perception-vs-decision and travel-economics test (competitor intel).

WHY THIS EXISTS
A competitor's oracle experiment claimed that perfect knowledge of all food still leaves colonies
dying at ~1630 s, so their bottleneck was decision/energy logic, not perception. They also suspected
their own 90 px food-search radius made agents camp next to nothing while ripe fruit stood farther
away, and wanted to widen it to 500 px.

WHAT WE ACTUALLY HAVE (read from source, not assumed)
  * Our controller has NO self-imposed food-search radius. It steers at the NEAREST OBSERVED fruit
    (`forage_nearest=1.0` is live) and has no distance cutoff at all.
  * The radius is imposed by the SIM's sensor: hearing_radius 50 (omnidirectional) + vision_radius
    200 in a cone_angle pi/3 (= +-30 deg) with river-edge occlusion (creature.py:138-139). One agent
    therefore surveys 0.5*200^2*(pi/3) = 20,944 px^2 = 1.09% of the 1600x1200 world per tick.
  * Both radii are HERITABLE GENES with hard caps: hearing <= chunk/4 = 100, vision <= chunk = 400
    (environment.py:335-344), cone <= pi/2. So the largest sensor the sim permits is
    0.5*400^2*(pi/2) = 125,664 px^2 = 6.5% of the world - 6x the default. That is the only legal
    version of "widen the search radius" in this simulator, and it is reachable only through
    reproduction (the policy cannot set a gene), which is what genome_select + gs_w_vision are for.

ARMS (same policy, same seeds, same world; only the OBSERVATION or the SENSOR changes)
  BASE          normal perception.
  ORACLE_FRUIT  observation augmented with EVERY fruit in the world at its true relative
                distance/angle -> perfect food knowledge, world otherwise untouched (predators ON).
  ORACLE_NOPRED ORACLE_FRUIT + predator spawning disabled -> a faithful copy of the competitor's
                oracle (perfect food knowledge, no predators).
  ORACLE_RIPE   only GROWN fruit (energy >= 40 of a 60 cap) is injected: the direct analogue of
                "travel farther for the ripe fruit".
  WIDE_VISION   sensors forced every tick to the genetic cap (vision 400, cone pi/2, hearing 100).
                This is the ceiling of what vision-selection could ever buy.

DIAGNOSTICS on every episode (from the same loop, so they are free):
  blind_frac            share of agent-ticks with NO fruit in view (income is access-limited?)
  fruit_vis_mean        fruits visible per agent-tick
  away_frac             share of agent-ticks where the agent moved AWAY from a visible fruit
  starve_food_frac      share of agent-ticks at <20% energy WITH a fruit in view
  travel_per_fruit      total distance moved / fruits eaten
  income_per_1k         fruit energy absorbed per 1,000 ticks (score delta x 1000)
  move_e_per_1k         energy spent moving/turning per 1,000 ticks (0.05/unit walk, 0.5/unit beyond
                        walk speed, |turn|/2pi per turn)
  metab_per_1k          passive drain (0.1/tick at biome rate 1.0)

Usage:
  ./oracle_percept.py --arms BASE,ORACLE_FRUIT --seeds 500-509 --horizon 12000 --workers 8 --out o1.json
"""
import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

# MUST precede any worker interpreter start (read at interpreter startup).
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
# walk UP to the repo root (the dir holding src/) so this works from experiments/ or a subdir of it
ROOT = HERE
while ROOT != "/" and not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
for p in (HERE, ROOT, os.path.join(ROOT, "experiments")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

DEPLOYED_GS = os.path.join(HERE, "DEPLOY_GS_params.json")
DEPLOYED_C6 = os.path.join(ROOT, "best_controller", "params.json")
# The LIVE artifact: pulled from the running container on the serving box by the sibling agent
# (sha 252f0ba1). Using it makes BASE literally the policy that is answering requests today.
SERVED_CTRL = os.path.join(HERE, "served_controller_252f0ba1.py")
SERVED_PARAMS = os.path.join(HERE, "served_params_252f0ba1.json")
ARMS = ("BASE", "ORACLE_FRUIT", "ORACLE_FRUIT_SLOW", "ORACLE_RIPE", "ORACLE_NOPRED", "WIDE_VISION")

# The control that separates "we cannot see" from "we mis-spend what we see": under perfect
# perception the controller's direct-forage branch fires on EVERY tick, so it travels at
# forage_speed=1.0 permanently and never enters the energy-throttled blind branch. If that is the
# reason an oracle arm dies early, throttling the speed restores it and the bottleneck is the SPEED
# RULE, not perception.
ARM_OVERRIDES = {"ORACLE_FRUIT_SLOW": {"forage_speed": 0.35},
                 "ORACLE_RIPE_SLOW": {"forage_speed": 0.35}}

# SIM-LEVEL PATCHES, applied to the live agents every tick. The competitor read "colonies still died
# with perfect knowledge" as proof that DECISION/energy logic is the bottleneck - but this sim gives
# every agent a hard max_age of 60-120 s (agent.py:26), i.e. an agent dies of old age after
# 600-1,200 ticks no matter how well fed it is. A colony can therefore be bounded by its
# breeding/aging economy rather than by perception OR by decisions, and the two are indistinguishable
# unless aging is held off. These arms do exactly that.
ARM_PATCHES = {"BASE_NOAGE": "noage", "ORACLE_NOPRED_NOAGE": "noage"}


def _apply_patch(kind, live):
    if kind == "noage":
        for a in live:
            a.max_age = 1e9


def _load_controller(path):
    """Import a controller module by FILE PATH so an experiment can pin the exact live artifact
    instead of whatever best_controller.py happens to be in the shared repo right now (a sibling
    agent edits it concurrently)."""
    import importlib.util
    name = "ctrl_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_params(mod, path):
    P = dict(mod.DEFAULT_PARAMS)
    P.update(json.load(open(path)))
    return P


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_episode(task):
    arm, seed, horizon, params_path, povr, ctrl_path = task
    from src.core import SimulationCore
    from env_wrapper import make_action

    random.seed(seed)
    np.random.seed(seed)
    mod = _load_controller(ctrl_path)
    mod.reset_memory()
    P = load_params(mod, params_path)
    P.update(povr or {})
    policy = mod.make_policy(P)

    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=5, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    env = core.env
    nopred = "NOPRED" in arm
    if nopred:
        env.spawn_predator = lambda *a, **k: None   # their oracle: no predators at all

    # EXACT fruit income. remove_fruit() is called from exactly two places: agent eating
    # (environment.py:675, pays fruit.energy) and rot at age>100 (line 736, pays nothing), so
    # hooking it removes all guesswork (and avoids id() reuse when a spawn frees a slot).
    removed = []
    _orig_rm = env.remove_fruit

    def _rm(fr):
        removed.append((float(getattr(fr, "energy", 0.0)), float(getattr(fr, "age", 0.0))))
        return _orig_rm(fr)

    env.remove_fruit = _rm

    M = dict(agent_ticks=0, blind_ticks=0, fruit_vis_sum=0, away_ticks=0, starve_food_ticks=0,
             move_e=0.0, turn_e=0.0, metab_e=0.0, income_e=0.0, dist=0.0,
             fruit_stock=0, sensors=None)
    last_score = 0.0
    eaten = spawns = n_fruit = 0
    steps = 0
    after = len(env.agents)

    for i in range(horizon):
        steps = i + 1
        live = list(env.agents)
        pk = ARM_PATCHES.get(arm)
        if pk:
            _apply_patch(pk, live)
        if arm == "WIDE_VISION":
            # force the sensor to the genetic cap every tick (the sim reads these at observe())
            for a in live:
                a.vision_radius = 400.0
                a.cone_angle = math.pi / 2
                a.hearing_radius = 100.0

        acts = []
        for a in live:
            s = env.get_agent_state(a.agent_id)
            if s is None:
                continue
            obs = s.get("observations") or []
            fruits_o = [o for o in obs if o.get("type") == "Fruit"]
            if arm.startswith("ORACLE"):
                # perfect food knowledge: EVERY fruit in the world, at its true relative bearing.
                # ORACLE_RIPE injects only GROWN fruit (energy >= 40 of a 60 cap), i.e. it answers
                # "would knowing where the ripe fruit is, and ignoring the rest, help?" - the
                # question the competitor's 90->500 px change was really trying to ask.
                inj = list(fruits_o)
                for f in env.fruits:
                    if arm == "ORACLE_RIPE" and float(getattr(f, "energy", 0.0)) < 40.0:
                        continue
                    dx, dy = f.x - a.x, f.y - a.y
                    d = math.hypot(dx, dy)
                    if d > 1e-6:
                        inj.append({"type": "Fruit", "distance": float(d),
                                    "angle": float(_wrap(math.atan2(dy, dx) - a.direction))})
                fruits_o = inj

            # ---- diagnostics (pre-step) ------------------------------------------------
            ef = float(s.get("energy", 0.0)) / max(1.0, float(s.get("max_energy", 1.0)))
            M["agent_ticks"] += 1
            M["fruit_vis_sum"] += len(fruits_o)
            if not fruits_o:
                M["blind_ticks"] += 1
            elif ef < 0.20:
                M["starve_food_ticks"] += 1

            if arm.startswith("ORACLE"):
                s = dict(s)
                s["observations"] = [o for o in obs if o.get("type") != "Fruit"] + fruits_o

            act = policy(s)
            # "camping" measure: does the CHOICE move toward the nearest visible fruit?
            # Comparable across arms because it is per-tick and normalised by the fruit in view.
            if fruits_o and float(act[0]) > 0.1:
                na = min(fruits_o, key=lambda o: o["distance"])["angle"]
                if math.cos(_wrap(na - float(act[1]))) < -0.2:
                    M["away_ticks"] += 1
            # nominal energy accounting for the chosen action
            dist = float(act[0])
            spd = float(s.get("speed", 10.0) or 10.0)
            spmax = float(s.get("sprint_speed", 20.0) or 20.0)
            dist_eff = min(dist, spmax)
            me = dist_eff * 0.05 if dist_eff <= spd else spd * 0.05 + (dist_eff - spd) * 0.5
            M["move_e"] += me
            M["turn_e"] += min(math.pi, abs(float(act[2]))) / (2 * math.pi)
            M["dist"] += dist_eff
            acts.append((a.agent_id, make_action(s, act)))

        before = len(env.agents)
        del removed[:]
        out = core.step(acts)
        after = len(env.agents)
        for e, ag in removed:
            if ag <= 100.0:          # eaten (rot only fires at age>100 and pays nothing)
                n_fruit += 1
                M["income_e"] += e
        if after < before:
            eaten += before - after
        elif after > before:
            spawns += after - before
        M["metab_e"] += 0.1 * len(env.agents)
        M["fruit_stock"] += len(env.fruits)
        if arm == "WIDE_VISION" and live:
            M["sensors"] = [round(live[0].vision_radius, 1), round(live[0].cone_angle, 3),
                            round(live[0].hearing_radius, 1)]
        if nopred:
            env.spawn_predator = lambda *a, **k: None
        if after == 0:
            break

    at = max(1, M["agent_ticks"])

    def per1k(v):
        return round(v * 1000.0 / steps, 2) if steps else 0.0

    return {
        "arm": arm, "seed": seed, "steps": steps, "score": round(core.env.score, 3),
        "eaten": n_fruit, "deaths": eaten, "spawns": spawns, "final_agents": after,
        "blind_frac": round(M["blind_ticks"] / at, 4),
        "fruit_vis_mean": round(M["fruit_vis_sum"] / at, 3),
        "away_frac": round(M["away_ticks"] / at, 4),
        "starve_food_frac": round(M["starve_food_ticks"] / at, 4),
        "travel_per_fruit": round(M["dist"] / max(1, n_fruit), 1),
        "income_per_1k": per1k(M["income_e"]),
        "move_e_per_1k": per1k(M["move_e"] + M["turn_e"]),
        "metab_per_1k": per1k(M["metab_e"]),
        "fruit_stock_mean": round(M["fruit_stock"] / max(1, steps), 1),
        "sensors": M["sensors"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="BASE,ORACLE_FRUIT")
    ap.add_argument("--seeds", default="500-509")
    ap.add_argument("--horizon", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--params", default=SERVED_PARAMS)
    ap.add_argument("--controller", default=SERVED_CTRL)
    ap.add_argument("--out", default="oracle_percept.json")
    args = ap.parse_args()

    a0, a1 = args.seeds.split("-")
    seeds = list(range(int(a0), int(a1) + 1))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    jobs = [(arm, s, args.horizon, args.params, ARM_OVERRIDES.get(arm), args.controller)
            for arm in arms for s in seeds]
    print(f"arms={arms} seeds={len(seeds)} horizon={args.horizon} jobs={len(jobs)} "
          f"workers={args.workers} params={os.path.basename(args.params)} "
          f"ctrl={os.path.basename(args.controller)}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_episode, jobs):
            rows.append(r)
            print("  %-13s seed=%-5d steps=%-6d score=%-7s blind=%-6s fvis=%-6s away=%-7s starve+food=%s"
                  % (r["arm"], r["seed"], r["steps"], r["score"], r["blind_frac"],
                     r["fruit_vis_mean"], r["away_frac"], r["starve_food_frac"]), flush=True)

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)

    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    base = by.get("BASE", {})
    print("\n=== PAIRED SUMMARY (steps survived; paired against BASE on the same seeds) ===")
    for arm in arms:
        d = by.get(arm, {})
        if not d:
            continue
        ks = sorted(d)
        st = [d[s]["steps"] for s in ks]
        pd = [d[s]["steps"] - base[s]["steps"] for s in ks if s in base]
        w = sum(1 for v in pd if v > 0)
        l = sum(1 for v in pd if v < 0)
        pm = sum(pd) / len(pd) if pd else 0.0
        av = lambda k: sum(d[s][k] for s in ks) / len(ks)
        print("%-13s mean_steps=%8.0f  paired_vs_BASE=%+8.1f  (%dW/%dL)  blind=%.3f fvis=%.2f "
              "away=%.4f travel/fruit=%.1f income/1k=%.1f move/1k=%.1f metab/1k=%.1f"
              % (arm, sum(st) / len(st), pm, w, l, av("blind_frac"), av("fruit_vis_mean"),
                 av("away_frac"), av("travel_per_fruit"), av("income_per_1k"),
                 av("move_e_per_1k"), av("metab_per_1k")))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
