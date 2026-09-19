#!/usr/bin/env python3
"""gen_search.py - sample the FULL knob space, instead of testing one hypothesis at a time.

Why: roughly 30 arms were tested in this project, each a single-knob hypothesis. In a space of dozens
of interacting continuous knobs that is a walk, not a search, and it cannot find a distant optimum. This
generator samples broadly around the deployed controller (log-scaled jitter, so a knob at 0.05 and one
at 12.0 both get sensible variation), plus a minority of wide excursions for genuine exploration.

Everything is sampled from the DEPLOYED params, so every candidate is a perturbation of the artifact
that is actually serving, and sched.py reports each one paired against that same baseline.

Usage: ./gen_search.py --n 600 --jitter 0.6 --wide 0.15 --seed 7 --out wide1_cands.json
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEPLOYED = os.path.join(ROOT, "best_controller", "params.json")

# Knobs deliberately EXCLUDED from sampling:
#   * identity/structural switches whose effect is already measured to be neutral or harmful when
#     flipped (evade_* is the closed E3 line, genome_select is the closed V2 line)
#   * anything whose "off" value is load-bearing for the current behaviour (memory_search, face_predator)
# Excluding them keeps the search inside the region where the incumbent's behaviour is intact and only
# its continuous trade-offs move. Flipping structural switches is a different experiment, not a sample.
EXCLUDE_PREFIX = ("evade_", "gs_", "genome_", "thin_relay", "relay_", "wB", "sel_", "reserve_",
                  "repro_energy", "commit_speed")


def jitter_value(name, v, rng, jitter, wide):
    if v is None:
        return v
    # log-scaled multiplicative jitter, symmetric in ratio space; handles values near 0 by an offset
    scale = jitter if rng.random() > wide else jitter * 3.0
    if isinstance(v, bool):
        return v if rng.random() > 0.25 else (not v)
    if isinstance(v, int):
        f = math.exp(rng.uniform(-scale, scale))
        return max(0, int(round(v * f))) if v > 0 else int(round(v + rng.uniform(-scale, scale)))
    f = math.exp(rng.uniform(-scale, scale))
    return round(v * f, 6) if abs(v) > 1e-9 else round(rng.uniform(-scale, scale) * 0.05, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--jitter", type=float, default=0.6)
    ap.add_argument("--wide", type=float, default=0.15, help="fraction of knobs given a 3x wider draw")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="wide_cands.json")
    ap.add_argument("--per-cand-knobs", type=float, default=0.45,
                    help="fraction of eligible knobs perturbed per candidate (keeps most of the incumbent)")
    args = ap.parse_args()

    base = json.load(open(DEPLOYED))
    eligible = [k for k, v in base.items()
                if not k.startswith(EXCLUDE_PREFIX) and isinstance(v, (int, float, bool))]
    rng = random.Random(args.seed)
    out = []
    for i in range(args.n):
        p = dict(base)
        k = max(1, int(len(eligible) * args.per_cand_knobs))
        for name in rng.sample(eligible, k):
            p[name] = jitter_value(name, base[name], rng, args.jitter, args.wide)
        out.append({"tag": f"s{i:04d}", "params": p})
    json.dump(out, open(args.out, "w"))
    print(f"wrote {len(out)} candidates to {args.out}")
    print(f"eligible knobs: {len(eligible)} of {len(base)} (excluded families: {EXCLUDE_PREFIX})")
    print(f"per candidate: {max(1, int(len(eligible) * args.per_cand_knobs))} knobs perturbed, "
          f"jitter {args.jitter} (3x wider for {int(100*args.wide)}% of draws)")


if __name__ == "__main__":
    main()
