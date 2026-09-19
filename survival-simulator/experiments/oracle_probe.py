#!/usr/bin/env python3
"""oracle_probe.py - feasibility gate for counterfactual search, then the smallest oracle experiment.

THE GATE. Everything in the counterfactual/oracle direction depends on one capability: snapshot a world
state, roll it forward, and get EXACTLY the continuation the uninterrupted run would have produced.
If that fails, an "oracle" is just a second simulator producing different worlds, and any decisions it
recommends are worthless. So this tests three things, in order:

  1. snapshot -> restore -> K ticks is BIT-IDENTICAL to the uninterrupted run (same positions, energies,
     ages, score). Without this the direction is dead.
  2. a DIFFERENT action from the same snapshot produces a DIFFERENT outcome (the branch is causal, not
     a no-op). Without this the oracle has nothing to compare.
  3. restore is repeatable: the same snapshot can be branched many times.

Then THE SMALLEST ORACLE EXPERIMENT: sample states from real runs, and for each one ask whether a fixed
alternative behaviour (e.g. "this agent heads north for the next K ticks") beats what the controller
actually did, scored on the focal agent's survival / energy / lockout time. If the oracle systematically
disagrees with the controller, there is something to learn; if it never disagrees, the controller is
already near-optimal on this action set and the direction is not worth building out.

Usage:
  ./oracle_probe.py --selftest --seed 2600 --ticks 1500
  ./oracle_probe.py --probe --seed 2600 --states 6 --horizon 400 --every 150
"""
import argparse
import copy
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402

import best_controller as bc  # noqa: E402
from src.core import SimulationCore  # noqa: E402
from env_wrapper import make_action  # noqa: E402

DEPLOYED = os.path.join(ROOT, "best_controller", "params.json")
N_AGENTS = 5


def load_params(path=DEPLOYED):
    import json
    p = dict(bc.DEFAULT_PARAMS)
    p.update(json.load(open(path)))
    return p


def build_core(seed):
    random.seed(seed)
    np.random.seed(seed)
    return SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=N_AGENTS,
                          starting_predators=0, starting_trees=50, seed=seed)


# The environment holds six pygame view surfaces (biome/shadow/obstacle/world/vision/leaf), created in
# __init__ purely for rendering. They cannot be deep-copied, and nothing in the simulation reads them
# (verified: no `.surface` reference exists outside draw(...) signatures), so we detach them across the
# copy and re-attach the live ones on restore. This is what makes snapshot/restore possible at all.
SURFACE_ATTRS = ("biome_surface", "static_surface", "shadow_surface", "obstacle_surface",
                 "world_surface", "vision_screen", "leaf_screen")


def snapshot(env):
    """A deep copy of the world with the view surfaces temporarily detached (render-only, not state)."""
    saved = {}
    for k in SURFACE_ATTRS:
        if hasattr(env, k):
            saved[k] = getattr(env, k)
            setattr(env, k, None)
    try:
        snap = copy.deepcopy(env)
    finally:
        for k, v in saved.items():
            setattr(env, k, v)
    return snap


def restore(core, snap):
    """Install a fresh copy of the snapshot, re-attaching the live view surfaces."""
    s = copy.deepcopy(snap)
    for k in SURFACE_ATTRS:
        if getattr(s, k, "missing") is None and hasattr(core.env, k):
            setattr(s, k, getattr(core.env, k))
    core.env = s
    return core


def signature(core):
    """A precise fingerprint of the world: every agent's id/position/energy/age, plus the score."""
    sig = []
    for a in sorted(core.env.agents, key=lambda x: x.agent_id):
        sig.append((a.agent_id, round(float(a.x), 6), round(float(a.y), 6),
                    round(float(a.energy), 6), round(float(a.age), 6)))
    return (round(float(core.env.score), 8), tuple(sig))


def roll(core, fn, ticks, override=None):
    """Advance `ticks`. `override` = (agent_id, action) where action is a tuple OR a callable f(state)
    -> action. A callable is recomputed every tick, which is what a sustained BEHAVIOUR needs: the agent
    turns as it moves, so a fixed relative heading stops meaning the same thing after a few ticks."""
    for _ in range(ticks):
        states = [core.env.get_agent_state(a.agent_id) for a in list(core.env.agents)]
        acts = []
        for st in states:
            aid = st["agent_id"]
            if override is not None and aid == override[0]:
                act = override[1](st) if callable(override[1]) else override[1]
                acts.append((aid, make_action(st, act)))
            else:
                acts.append((aid, make_action(st, fn(st))))
        core.step(acts)
        if not core.env.agents:
            break
    return core


def act_toward(target_kind, sprint=True):
    """A sustained behaviour: every tick, head for the nearest visible `target_kind` (Fruit/Predator).
    Falls back to standing still when nothing of that kind is visible."""
    import math

    def f(st):
        obs = [o for o in (st.get("observations") or []) if o.get("type") == target_kind]
        if not obs:
            return (0.0, 0.0, 0.0, False)
        o = min(obs, key=lambda x: x.get("distance", 1e9))
        bearing = float(o.get("angle", 0.0))          # relative bearing to the target
        dist = 20.0 if sprint else min(10.0, st.get("speed", 10.0))
        return (dist, max(-math.pi, min(math.pi, bearing)), 0.0, False)
    return f


def act_away_from(kind="Predator", sprint=True):
    """A sustained behaviour: sprint directly away from the nearest visible predator."""
    import math

    def f(st):
        obs = [o for o in (st.get("observations") or []) if o.get("type") == kind]
        if not obs:
            return (0.0, 0.0, 0.0, False)
        o = min(obs, key=lambda x: x.get("distance", 1e9))
        bearing = float(o.get("angle", 0.0)) + math.pi
        if bearing > math.pi:
            bearing -= 2 * math.pi
        return (20.0 if sprint else 10.0, bearing, 0.0, False)
    return f


def selftest(seed, ticks, branch_ticks, override_action=(20.0, 0.0, 0.0, False)):
    P = load_params()
    fn = bc.make_policy(P)
    core = build_core(seed)
    bc.reset_memory()
    roll(core, fn, ticks)

    print(f"selftest | seed {seed} | {ticks} ticks warm-up | branches of {branch_ticks} ticks")
    try:
        snap = snapshot(core.env)
    except Exception as exc:
        print(f"  SNAPSHOT FAILED: {type(exc).__name__}: {exc}")
        print("  -> counterfactual search is NOT available without a custom state-capture layer")
        return False
    print(f"  deepcopy of the environment: OK  ({len(core.env.agents)} agents, "
          f"{len(core.env.predators)} predators, {len(core.env.fruits)} fruits)")

    # 1. uninterrupted continuation
    fn_a = bc.make_policy(P)
    bc.reset_memory()
    roll(core, fn_a, branch_ticks)
    sig_a = signature(core)
    print(f"  uninterrupted : score={sig_a[0]:.2f} agents={len(sig_a[1])}")

    # 2. restored continuation, same policy
    restore(core, snap)
    fn_b = bc.make_policy(P)
    bc.reset_memory()
    roll(core, fn_b, branch_ticks)
    sig_b = signature(core)
    print(f"  restored      : score={sig_b[0]:.2f} agents={len(sig_b[1])}")
    identical = (sig_a == sig_b)
    print(f"  GATE 1 (bit-identical continuation): {'PASS' if identical else 'FAIL'}")
    if not identical:
        print(f"    first difference: {next((x for x, y in zip(sig_a[1], sig_b[1]) if x != y), None)}"
              f" vs {next((y for x, y in zip(sig_a[1], sig_b[1]) if x != y), None)}")
        return False

    # 3. a different action must produce a different world (the branch must be causal)
    restore(core, snap)
    fn_c = bc.make_policy(P)
    bc.reset_memory()
    focal = min((a.agent_id for a in core.env.agents), default=0)
    roll(core, fn_c, branch_ticks, override=(focal, override_action))
    sig_c = signature(core)
    print(f"  branch (agent {focal} forced): score={sig_c[0]:.2f} agents={len(sig_c[1])}")
    causal = (sig_a != sig_c)
    print(f"  GATE 2 (a different action changes the world): {'PASS' if causal else 'FAIL'}")

    # 4. restore is repeatable
    restore(core, snap)
    fn_d = bc.make_policy(P)
    bc.reset_memory()
    roll(core, fn_d, branch_ticks)
    repeatable = (signature(core) == sig_a)
    print(f"  GATE 3 (restore is repeatable): {'PASS' if repeatable else 'FAIL'}")
    ok = identical and causal and repeatable
    print(f"\nVERDICT: counterfactual search is {'AVAILABLE' if ok else 'NOT AVAILABLE'} on this build")
    return ok


def probe(seed, warm, states, horizon, every, out_path=None):
    """The smallest oracle experiment: does a fixed alternative behaviour ever beat the controller?"""
    P = load_params()
    fn = bc.make_policy(P)
    core = build_core(seed)
    bc.reset_memory()
    print(f"oracle probe | seed {seed} | {states} sampled states | {horizon}-tick branches\n")
    rows = []
    for i in range(states):
        roll(core, fn, warm if i == 0 else every)
        if not core.env.agents:
            print(f"  state {i}: fleet extinct at tick {warm + i * every}, stopping")
            break
        snap = snapshot(core.env)
        focal = sorted(core.env.agents, key=lambda a: a.energy)[len(core.env.agents) // 2].agent_id
        opts = {"sprint_to_fruit": act_toward("Fruit", True),
                "walk_to_fruit": act_toward("Fruit", False),
                "flee_predator": act_away_from("Predator", True),
                "toward_predator": act_toward("Predator", True),
                "north": (20.0, 0.0, 0.0, False),
                "still": (0.0, 0.0, 0.0, False)}
        res = {}
        for name, act in opts.items():
            restore(core, snap)
            f = bc.make_policy(P)
            bc.reset_memory()
            roll(core, f, horizon, override=(focal, act))
            a = next((x for x in core.env.agents if x.agent_id == focal), None)
            res[name] = {"alive": a is not None,
                         "energy": round(float(a.energy), 1) if a else 0.0,
                         "age": round(float(a.age), 1) if a else 0.0,
                         "score": round(float(core.env.score), 1)}
        # what did the controller actually do over the same window?
        restore(core, snap)
        f = bc.make_policy(P)
        bc.reset_memory()
        roll(core, f, horizon)
        a = next((x for x in core.env.agents if x.agent_id == focal), None)
        ctrl = {"alive": a is not None, "energy": round(float(a.energy), 1) if a else 0.0,
                "score": round(float(core.env.score), 1)}
        best = max(res.items(), key=lambda kv: (kv[1]["alive"], kv[1]["energy"]))
        beats = (best[1]["alive"] and not ctrl["alive"]) or (best[1]["energy"] > ctrl["energy"] + 5)
        rows.append({"state": i, "tick": warm + i * every, "focal": focal, "controller": ctrl,
                     "best_alt": best[0], "best": best[1], "oracle_beats": bool(beats), "all": res})
        flag = "ORACLE BEATS CONTROLLER" if beats else "controller fine"
        print(f"  t~{warm + i*every:>5}  focal a{focal:<3} ctrl alive={int(ctrl['alive'])} e={ctrl['energy']:>7.1f}"
              f" | best alt {best[0]:<6} alive={int(best[1]['alive'])} e={best[1]['energy']:>7.1f}  {flag}")
    if rows:
        n = sum(r["oracle_beats"] for r in rows)
        print(f"\n{n}/{len(rows)} sampled states: an alternative action beat what the controller did")
        print("that fraction is the size of the opportunity. If it is ~0, the controller is already good")
        print("on this action set and the oracle direction is not worth building out.")
    if out_path:
        import json
        json.dump(rows, open(out_path, "w"), indent=1)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--seed", type=int, default=2600)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--branch-ticks", type=int, default=300)
    ap.add_argument("--warm", type=int, default=1200)
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--every", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        ok = selftest(a.seed, a.ticks, a.branch_ticks)
        sys.exit(0 if ok else 1)
    if a.probe:
        probe(a.seed, a.warm, a.states, a.horizon, a.every, a.out)
