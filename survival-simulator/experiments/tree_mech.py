#!/usr/bin/env python3
"""tree_mech.py - MECHANISM screen for the tree-attraction dose curve (travel per fruit).

WHY: deleting tree attraction (tree_weight 0.25 -> 0.0) improved paired survival in two
independent 40-seed runs:
    seeds 2660-2699 (c6minus stage 2): +2.7%  W22/L18
    seeds 2960-2999 (c6conf  stage 1): +7.6%  W26/L14  (fruit 1387 vs 1307)
That is counter-intuitive: comments in best_controller.py say income is ACCESS-limited and
the trees ARE the fruit source, so a search cue that points at trees should matter. Before
believing the survival number, measure the MECHANISM the arm claims to move: TRAVEL PER FRUIT
(lower = winners' signature, 0.34x in 16/16 traced episodes).

Dose curve, same seeds for every arm, read-only (nothing is deployed):
    tree_weight = -0.25 (wrong-direction control, should RAISE travel/fruit)
                  0.0   (the candidate)
                  0.1
                  0.25  (deployed C6 value = base)
                  0.5

Usage: SDL_VIDEODRIVER=dummy /opt/nacv/bin/python tree_mech.py --seeds 3040-3042 --outdir /tmp/tree_mech
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import mechanism_screen as ms  # noqa: E402  (reuse the verified screen, no edits to it)

DOSE = {
    "tw_neg025": {"tree_weight": -0.25},
    "tw_0": {"tree_weight": 0.0},
    "tw_01": {"tree_weight": 0.1},
    "tw_025_base": {},
    "tw_05": {"tree_weight": 0.5},
}
# NOTE: must be applied at MODULE level, not inside main(): Python 3.14 defaults the multiprocessing
# start method to forkserver on Linux, so the workers re-import this module and would otherwise see
# mechanism_screen.ARMS without the dose arms (KeyError).
ms.ARMS.update(DOSE)


def job(item):
    name, seeds, outdir = item
    try:
        return ms.screen_arm(name, seeds, outdir)
    except Exception as e:  # one bad arm must not kill the screen
        return {"arm": name, "error": repr(e), "seeds": len(seeds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3040-3042")
    ap.add_argument("--outdir", default="/tmp/tree_mech")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    ms.ARMS.update(DOSE)  # screen_arm() looks the arm up in mechanism_screen.ARMS
    seeds = ms.parse_seeds(a.seeds)
    os.makedirs(a.outdir, exist_ok=True)

    print(f"TREE DOSE MECHANISM SCREEN | seeds {a.seeds} ({len(seeds)}) x 18000 ticks | "
          f"outdir {a.outdir}", flush=True)
    with Pool(min(a.workers, len(DOSE))) as p:
        rows = p.map(job, [(n, seeds, a.outdir) for n in DOSE])
    json.dump(rows, open(os.path.join(a.outdir, "tree_dose.json"), "w"), indent=1)

    base = next((r["travel_per_fruit"] for r in rows if r["arm"] == "tw_025_base"
                 and r.get("travel_per_fruit")), None)
    print(f"\n{'arm':>14} {'travel/fruit':>13} {'vs base':>8} {'lockout':>8} {'fruit/1k':>9} "
          f"{'eMax med':>9} {'ticks med':>10} {'agents':>7}")
    for r in rows:
        if r.get("error"):
            print(f"{r['arm']:>14} ERROR {r['error']}")
            continue
        tpf = r.get("travel_per_fruit") or 0.0
        rel = f"{tpf / base:.2f}x" if base else ""
        print(f"{r['arm']:>14} {tpf:>13.1f} {rel:>8} {(r.get('lockout_frac') or 0):>8.2f} "
              f"{(r.get('fruits_per_1k') or 0):>9.1f} {(r.get('e_max_med') or 0):>9.0f} "
              f"{(r.get('ticks_median') or 0):>10.0f} {r.get('agents_measured', 0):>7}")
    print("\nrule: the candidate earns belief only if it LOWERS travel/fruit AND the "
          "wrong-direction control (-0.25) RAISES it.")


if __name__ == "__main__":
    main()
