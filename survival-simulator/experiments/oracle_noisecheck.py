#!/usr/bin/env python3
"""oracle_noisecheck.py - MEASURE the oracle's noise floor before believing its disagreements.

Q2's sweep reported "an alternative beats the controller in 24% of states" (77/317). But 59 of those 77
states had >=3 winners AT ONCE, and 'still' (do nothing) won 60 times while 'toward_predator' (suicide)
won 53 - i.e. in a disagreeing state almost every alternative wins. That is not a lesson about
behaviour, it is the signature of a branch-level coin flip: if the CONTROLLER branch is not
reproducible from the same snapshot, then "an alternative beat it" is just noise.

So: restore the same snapshot twice (or N times) and run the very same branch - twice for the
controller, twice for each alternative. If the controller disagrees with ITSELF, every counterfactual
disagreement rate must be compared against that self-disagreement rate, and the sweep's 24% may be
entirely explained by it.

Usage: PYTHONHASHSEED=0 ./oracle_noisecheck.py --seeds 2700-2705 --states-per-seed 3 --horizon 400
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import oracle_probe as op          # noqa: E402
import oracle_sweep as osweep      # noqa: E402


def run_branch(P, snap, core, aid, act, horizon):
    """Restore the snapshot and roll one branch. act=None -> the controller's own policy."""
    op.restore(core, snap)
    f = op.bc.make_policy(P)
    op.bc.reset_memory()
    if act is None:
        op.roll(core, f, horizon)
    else:
        op.roll(core, f, horizon, override=(aid, act))
    return {"focal": osweep.score_focal(core, aid, 0),
            "fleet": len(core.env.agents),
            "score": round(float(core.env.score), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2700-2705")
    ap.add_argument("--states-per-seed", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--every", type=int, default=800)
    ap.add_argument("--warm", type=int, default=800)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="oracle_noise.json")
    a = ap.parse_args()

    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        elif part.strip():
            seeds.append(int(part))

    behaviours = {"still": osweep.behaviours()["still"],
                  "sprint_to_fruit": osweep.behaviours()["sprint_to_fruit"]}
    rows = []
    print(f"NOISE CHECK | hashseed={os.environ.get('PYTHONHASHSEED','<unset>')} | "
          f"{len(seeds)} seeds x {a.states_per_seed} states x {a.reps} reps x "
          f"(controller + {len(behaviours)} behaviours) | horizon {a.horizon}\n")
    for seed in seeds:
        P = op.load_params()
        core = op.build_core(seed)
        fn = op.bc.make_policy(P)
        op.bc.reset_memory()
        for i in range(a.states_per_seed):
            op.roll(core, fn, a.warm if i == 0 else a.every)
            if not core.env.agents:
                break
            snap = op.snapshot(core.env)
            states_list = [core.env.get_agent_state(x.agent_id) for x in core.env.agents]
            focal = sorted(states_list, key=lambda s: s["agent_id"])[i % len(states_list)]
            aid = focal["agent_id"]
            rec = {"seed": seed, "focal": aid, "tick": int(round(float(core.env.time) * 10)),
                   "arms": {}}
            for name, act in [("controller", None)] + list(behaviours.items()):
                outs = [run_branch(P, snap, core, aid, act, a.horizon) for _ in range(a.reps)]
                sigs = [(o["focal"]["alive"], round(o["focal"]["energy"], 6), o["fleet"]) for o in outs]
                rec["arms"][name] = {"outs": outs, "distinct": len(set(sigs)), "reps": a.reps}
            rows.append(rec)
            c = rec["arms"]["controller"]
            print(f"  seed {seed} t~{rec['tick']:>5} a{aid:<3} "
                  f"controller distinct={c['distinct']}/{a.reps} "
                  f"energies={[o['focal']['energy'] for o in c['outs']]} "
                  f"fleet={[o['fleet'] for o in c['outs']]}", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    n = len(rows)
    print(f"\n=== SELF-DISAGREEMENT (same snapshot, same arm, replicated) over {n} states ===")
    for name in ["controller"] + list(behaviours):
        unstable = sum(1 for r in rows if r["arms"][name]["distinct"] > 1)
        print(f"  {name:>16}: differs on {unstable}/{n} states ({100*unstable/max(n,1):.0f}%)")
    print("\nInterpretation: the counterfactual sweep must beat THIS number to mean anything.")


if __name__ == "__main__":
    main()
