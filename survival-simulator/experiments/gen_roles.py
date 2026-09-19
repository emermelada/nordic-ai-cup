#!/usr/bin/env python3
"""gen_roles.py - generate HETEROGENEOUS FLEET candidates for sched.py ("__policy__": "roles").

The untested hypothesis: a fleet of deliberately different specialists survives longer than a fleet of
identical generalists, because the score only requires ONE lineage to survive and a mixed fleet cannot
be ended by a single failure mode (famine, a predator pack, an age cliff).

Roles are expressed as parameter overrides only, so no change to the served controller is needed.
The live controller's own defaults stay as the base for every role; each role then tilts a few knobs:

  EXPLORER  - wide search, cheap movement, low commitment: high blind_explore_frac, lower fruit_weight
  BANKER    - slow, energy-hoarding, spawns rarely, long-lived
  BREEDER   - spawns aggressively (low repro thresholds) so the ratchet gets many draws
  FORAGER   - maximum fruit commitment, hunts the nearest fruit every tick
  COWARD    - wide danger radius, retreats early and far (the one thing E3 got partly right)

Compositions are enumerated over role menus of size 1..3, plus assignment modes (by index, by energy),
because "which agent gets which role" is itself part of the hypothesis.

Usage:
    ./gen_roles.py --out roles_180.json
"""
import argparse
import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOYED = os.path.join(os.path.dirname(HERE), "best_controller", "params.json")

ROLES = {
    "explorer": {"blind_explore_frac": 0.55, "explore_frac": 0.45, "fruit_weight": 0.85,
                 "wander_weight": 0.16, "forage_nearest": 0.0, "wall_weight": 0.05},
    "banker": {"walk_frac": 0.35, "fruit_weight": 1.35, "repro_frac_min": 0.40,
               "repro_global_target": 8.0, "low_energy_frac": 0.30},
    "breeder": {"repro_frac_min": 0.20, "repro_frac": 0.28, "repro_global_target": 16.0,
                "spawn_cooldown": 70, "repro_urgency": 1.3},
    "forager": {"forage_speed": 1.0, "forage_nearest": 1.0, "fruit_weight": 1.6,
                "explore_frac": 0.1, "blind_explore_frac": 0.35},
    "coward": {"danger_dist": 240.0, "flee_dist": 260.0, "escape_dist": 330.0,
               "flee_speed_frac": 1.0, "concern_cone": 2.6, "fruit_risk_penalty": 1.2},
    "generalist": {},                      # the deployed behaviour, as a control role
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="roles_cands.json")
    ap.add_argument("--max-roles", type=int, default=3)
    ap.add_argument("--jitter", type=int, default=0,
                    help="emit this many jittered variants of every role (widens the search)")
    ap.add_argument("--jitter-scale", type=float, default=0.25, help="relative jitter magnitude")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    deployed = json.load(open(DEPLOYED))
    cands = []

    def variants(role):
        """The role itself plus jittered copies - so a role family is searched, not just one point."""
        yield role
        for _ in range(args.jitter):
            v = {}
            for k, val in role.items():
                if isinstance(val, (int, float)):
                    v[k] = round(val * (1.0 + rng.gauss(0.0, args.jitter_scale)), 4)
                else:
                    v[k] = val
            yield v

    def add(tag, roles, assign):
        cands.append({"id": tag, "params": {"__policy__": "roles", "__roles__": roles,
                                            "__role_assign__": assign}})

    # homogeneous controls (same role repeated) and single-role fleets
    for rname in ROLES:
        for i, rv in enumerate(variants(ROLES[rname])):
            add(f"hom_{rname}_v{i}", [rv], "by_index")

    # heterogeneous pairs (ordered: index 0 gets the first role)
    names = [n for n in ROLES if n != "generalist"]
    for a, b in itertools.permutations(names, 2):
        add(f"pair_{a}_{b}", [ROLES[a], ROLES[b]], "by_index")
        for i, (ra, rb) in enumerate(zip(variants(ROLES[a]), variants(ROLES[b]))):
            if i:
                add(f"pair_{a}_{b}_v{i}", [ra, rb], "by_index")

    # a few triples, chosen for strategic contrast rather than exhaustive enumeration
    triples = [("explorer", "banker", "breeder"), ("forager", "banker", "coward"),
               ("explorer", "forager", "breeder"), ("coward", "banker", "breeder"),
               ("explorer", "generalist", "breeder"), ("forager", "explorer", "banker")]
    for t in triples[: args.max_roles + 3]:
        add("trip_" + "_".join(t), [ROLES[x] for x in t], "by_index")
        add("tripE_" + "_".join(t), [ROLES[x] for x in t], "by_energy")
        for i, rv in enumerate(zip(*[variants(ROLES[x]) for x in t])):
            if i:
                add("trip_" + "_".join(t) + f"_v{i}", list(rv), "by_index")
                add("tripE_" + "_".join(t) + f"_v{i}", list(rv), "by_energy")

    json.dump(cands, open(args.out, "w"))
    print(f"wrote {len(cands)} heterogeneous-fleet candidates -> {args.out} (jitter {args.jitter})")
    print(f"roles used: {sorted(ROLES)}  | base = deployed params ({len(deployed)} keys)")


if __name__ == "__main__":
    main()
