#!/usr/bin/env python3
"""oracle_sweep.py - scale the counterfactual oracle from 3 states to hundreds, properly.

Q2 of AUTOPILOT.md. The single-seed probe answered "does an alternative ever beat the controller" with
1-of-3 states, which is an anecdote. This scales it and, more importantly, records WHEN the oracle
disagrees, so the disagreement can be turned into a rule:

    for each seed, for each sampled state:
        snapshot -> for each candidate behaviour: restore, roll H ticks, score the FOCAL agent
                 -> restore, roll H ticks with the controller, score the same way
        disagreement = best alternative beats the controller on (alive, then energy)

Aggregation is by behaviour and by state features (focal energy fraction, predator visible, fruit
visible, time), so the output is "behaviour X wins when the world looks like Y" rather than a p-value
about nothing. Everything is measured on x86; snapshot/restore is verified bit-identical (oracle_probe.py
--selftest).

Usage: ./oracle_sweep.py --seeds 2700-2711 --states-per-seed 5 --horizon 400 --out oracle_sweep.json
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import oracle_probe as op  # noqa: E402


def behaviours():
    """The candidate decisions. Kept small and interpretable on purpose: a rule we can encode, not a
    network we cannot debug. The first probe found 'sprint to nearest visible fruit' beating the
    controller, so goal-directed options are the interesting half."""
    return {
        "sprint_to_fruit": op.act_toward("Fruit", True),
        "walk_to_fruit": op.act_toward("Fruit", False),
        "flee_predator": op.act_away_from("Predator", True),
        "toward_predator": op.act_toward("Predator", True),
        "north": (20.0, 0.0, 0.0, False),
        "still": (0.0, 0.0, 0.0, False),
    }


def score_focal(core, aid, t0):
    """The focal agent's outcome over the branch: alive, energy, and lockout share."""
    a = next((x for x in core.env.agents if x.agent_id == aid), None)
    if a is None:
        return {"alive": False, "energy": 0.0, "ticks_survived": 0}
    return {"alive": True, "energy": round(float(a.energy), 1), "ticks_survived": 1}


def state_features(st, core):
    obs = st.get("observations") or []
    preds = [o for o in obs if o.get("type") == "Predator"]
    fruit = [o for o in obs if o.get("type") == "Fruit"]
    max_e = float(st.get("max_energy", 500.0) or 500.0)
    return {
        "energy_frac": round(float(st.get("energy", 0.0)) / max_e, 3),
        "in_lockout": bool(float(st.get("energy", 0.0)) < 0.2 * max_e),
        "pred_visible": len(preds),
        "pred_dist": min([o.get("distance", 1e9) for o in preds], default=None),
        "fruit_visible": len(fruit),
        "fleet_pop": len(core.env.agents),
        "tick": int(round(float(core.env.time) * 10)),
    }


def run_seed(seed, states, horizon, every, warm, rows, out):
    P = op.load_params()
    fn = op.bc.make_policy(P)
    core = op.build_core(seed)
    op.bc.reset_memory()
    opts = behaviours()
    for i in range(states):
        op.roll(core, fn, warm if i == 0 else every)
        if not core.env.agents:
            break
        snap = op.snapshot(core.env)
        states_list = [core.env.get_agent_state(a.agent_id) for a in core.env.agents]
        # rotate the focal agent so we are not always measuring the same individual
        focal_st = sorted(states_list, key=lambda s: s["agent_id"])[i % len(states_list)]
        aid = focal_st["agent_id"]
        feats = state_features(focal_st, core)

        res = {}
        for name, act in opts.items():
            op.restore(core, snap)
            f = op.bc.make_policy(P)
            op.bc.reset_memory()
            op.roll(core, f, horizon, override=(aid, act))
            res[name] = score_focal(core, aid, feats["tick"])

        op.restore(core, snap)
        f = op.bc.make_policy(P)
        op.bc.reset_memory()
        op.roll(core, f, horizon)
        ctrl = score_focal(core, aid, feats["tick"])

        def better(a, b):
            if a["alive"] != b["alive"]:
                return a["alive"]
            return a["energy"] > b["energy"] + 5.0

        winners = [n for n, r in res.items() if better(r, ctrl)]
        rows.append({"seed": seed, "focal": aid, "features": feats, "controller": ctrl,
                     "alternatives": res, "winners": winners,
                     "beats": bool(winners),
                     "best": max(res.items(), key=lambda kv: (kv[1]["alive"], kv[1]["energy"]))[0]})
        print(f"  seed {seed} t~{feats['tick']:>5} focal a{aid:<3} lockout={int(feats['in_lockout'])} "
              f"pred={feats['pred_visible']} fruit={feats['fruit_visible']:>2} "
              f"ctrl_alive={int(ctrl['alive'])} ctrl_e={ctrl['energy']:>7.1f}"
              f"  -> {('BEATS: ' + ','.join(winners[:2])) if winners else 'controller fine'}", flush=True)
    json.dump(rows, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2700-2711")
    ap.add_argument("--states-per-seed", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--every", type=int, default=800)
    ap.add_argument("--warm", type=int, default=800)
    ap.add_argument("--out", default="oracle_sweep.json")
    a = ap.parse_args()

    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        elif part.strip():
            seeds.append(int(part))

    rows = []
    print(f"ORACLE SWEEP | {len(seeds)} seeds x {a.states_per_seed} states | {a.horizon}-tick branches "
          f"| 6 behaviours + controller control\n")
    for sd in seeds:
        run_seed(sd, a.states_per_seed, a.horizon, a.every, a.warm, rows, a.out)

    if not rows:
        print("no usable states")
        return
    n = len(rows)
    beats = sum(r["beats"] for r in rows)
    print(f"\n=== DISAGREEMENT: an alternative beat the controller in {beats}/{n} states "
          f"({100*beats/n:.0f}%) ===")

    per = {}
    for r in rows:
        for w in r["winners"]:
            per.setdefault(w, []).append(r)
    print(f"\n{'behaviour':>18} {'wins':>6} {'share of states':>16}   wins more often when...")
    for name, rs in sorted(per.items(), key=lambda kv: -len(kv[1])):
        lock = st.mean([1.0 if r["features"]["in_lockout"] else 0.0 for r in rs])
        pred = st.mean([r["features"]["pred_visible"] for r in rs])
        fr = st.mean([r["features"]["fruit_visible"] for r in rs])
        ef = st.mean([r["features"]["energy_frac"] for r in rs])
        print(f"{name:>18} {len(rs):>6} {100*len(rs)/n:>15.0f}%   focal e={ef:.2f} lockout={lock:.2f} "
              f"pred={pred:.2f} fruit={fr:.1f}")

    # how does disagreement depend on the state?
    print(f"\n{'state bucket':>26} {'states':>7} {'beats':>7} {'rate':>7}")
    buckets = {
        "focal in lockout": lambda r: r["features"]["in_lockout"],
        "predator visible": lambda r: r["features"]["pred_visible"] > 0,
        "no predator visible": lambda r: r["features"]["pred_visible"] == 0,
        "fruit visible": lambda r: r["features"]["fruit_visible"] > 0,
        "no fruit visible": lambda r: r["features"]["fruit_visible"] == 0,
        "early (t<3000)": lambda r: r["features"]["tick"] < 3000,
        "late (t>=3000)": lambda r: r["features"]["tick"] >= 3000,
    }
    for name, f in buckets.items():
        rs = [r for r in rows if f(r)]
        if rs:
            b = sum(r["beats"] for r in rs)
            print(f"{name:>26} {len(rs):>7} {b:>7} {100*b/len(rs):>6.0f}%")
    print(f"\nwrote {a.out}  ({n} states)")


if __name__ == "__main__":
    main()
