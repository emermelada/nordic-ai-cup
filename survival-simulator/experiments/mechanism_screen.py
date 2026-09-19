#!/usr/bin/env python3
"""mechanism_screen.py - screen arms on the MECHANISM first, score second.

Why: winner analysis (winner_signature.py, 16 episodes, 586v586) found winners travel 0.34x as far per
fruit and spend 0.36x as long in the sprint lockout, both consistent in 16/16 episodes. The energy
ledger independently shows movement is the dominant cost. So the pre-registered mechanism target is:

    TRAVEL PER FRUIT  (units travelled per fruit absorbed)

That is measurable in minutes; a survival difference of the same size needs ~40 paired seeds. This
script records a few traced episodes per arm and reports the mechanism FIRST:

    arm, seeds -> travel_per_fruit, lockout_frac, fruit absorbed, then survival (ticks) as secondary

Only an arm that MOVES the mechanism earns a paired survival test. An arm that raises survival without
moving the mechanism is not doing the thing that made winners win, and should not be believed.

Arms are overrides on the deployed controller (params_sha is recorded in every replay). Read-only:
nothing is deployed, the served artifact is untouched.

Usage: ./mechanism_screen.py --seeds 2100-2103 --arms hyst16,hyst28,blind,both --outdir /tmp/mech
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import replay as rp  # noqa: E402

DEPLOYED = os.path.join(ROOT, "best_controller", "params.json")

# Each arm targets travel-per-fruit via a different route, so a win tells us WHICH route matters.
ARMS = {
    # stickier fruit targeting: fewer target switches, less travel to a target already abandoned
    "hyst16": {"target_hyst": 1.6},
    "hyst28": {"target_hyst": 2.8},
    # shorter blind hops and less random wander: travel only with a purpose
    "blind": {"blind_explore_frac": 0.12, "wander_weight": 0.03},
    # both routes together
    "both": {"target_hyst": 1.6, "blind_explore_frac": 0.12, "wander_weight": 0.03},
    # a deliberately WRONG direction as a control: travel MORE per fruit (should worsen the mechanism)
    "wanderup": {"wander_weight": 0.20, "blind_explore_frac": 0.50},
}


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def arm_params(name):
    p = dict(json.load(open(DEPLOYED)))
    p.update(ARMS[name])
    return p


def screen_arm(name, seeds, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}_params.json")
    json.dump(arm_params(name), open(path, "w"))
    tpf, lock, fruits, ticks, per_agent, lock_fracs = [], [], [], [], [], []
    for sd in seeds:
        rec = rp.record_episode(sd, 18000, params_path=path, every=4, traces=True)
        ticks.append(rec["final"]["ticks"])
        for a in rec.get("agents", []):
            if a["ticks"] < 100 or not a["fruits"]:
                continue
            tpf.append(a["travel"] / a["fruits"])
            lock_fracs.append(a["lockout_frac"])
            per_agent.append(a["fruits"] / max(1, a["ticks"]) * 1000)
        fruits.append(sum(1 for e in rec["events"] if e["k"] == "die" and e["cause"] == "eaten"))
        lock.append(0)
    return {"arm": name, "seeds": len(seeds),
            "travel_per_fruit": st.median(tpf) if tpf else None,
            "lockout_frac": st.median(lock_fracs) if lock_fracs else None,
            "fruits_per_1k": st.median(per_agent) if per_agent else None,
            "ticks_mean": st.mean(ticks) if ticks else None,
            "ticks_median": st.median(ticks) if ticks else None,
            "agents_measured": len(tpf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2100-2103")
    ap.add_argument("--arms", default="base,hyst16,hyst28,blind,both,wanderup")
    ap.add_argument("--outdir", default="/tmp/mech")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    wanted = [] if args.arms == "base" else [a for a in args.arms.split(",") if a != "base"]
    names = (["base"] if "base" in args.arms.split(",") else []) + wanted

    rows = []
    for nm in names:
        if nm == "base":
            # the deployed controller itself, recorded through the same path so the comparison is exact
            os.makedirs(args.outdir, exist_ok=True)
            p = os.path.join(args.outdir, "base_params.json")
            json.dump(json.load(open(DEPLOYED)), open(p, "w"))
            tpf, lock_fracs, per_agent, ticks = [], [], [], []
            for sd in seeds:
                rec = rp.record_episode(sd, 18000, params_path=p, every=4, traces=True)
                ticks.append(rec["final"]["ticks"])
                for a in rec.get("agents", []):
                    if a["ticks"] < 100 or not a["fruits"]:
                        continue
                    tpf.append(a["travel"] / a["fruits"])
                    lock_fracs.append(a["lockout_frac"])
                    per_agent.append(a["fruits"] / max(1, a["ticks"]) * 1000)
            rows.append({"arm": "base (deployed)", "seeds": len(seeds),
                         "travel_per_fruit": st.median(tpf) if tpf else None,
                         "lockout_frac": st.median(lock_fracs) if lock_fracs else None,
                         "fruits_per_1k": st.median(per_agent) if per_agent else None,
                         "ticks_mean": st.mean(ticks) if ticks else None,
                         "ticks_median": st.median(ticks) if ticks else None,
                         "agents_measured": len(tpf)})
        else:
            rows.append(screen_arm(nm, seeds, args.outdir))
        print(f"  done {nm}", flush=True)

    base = rows[0]["travel_per_fruit"] if rows else None
    print(f"\nMECHANISM SCREEN | {len(seeds)} seeds x 18000 ticks | target = travel per fruit (LOWER better)")
    print(f"{'arm':>18} {'travel/fruit':>13} {'vs base':>8} {'lockout':>8} {'fruit/1k':>9} "
          f"{'ticks mean':>10} {'agents':>7}")
    for r in rows:
        rel = ""
        if base and r["travel_per_fruit"]:
            rel = f"{r['travel_per_fruit'] / base:.2f}x"
        print(f"{r['arm']:>18} {(r['travel_per_fruit'] or 0):>13.1f} {rel:>8} "
              f"{(r['lockout_frac'] or 0):>8.2f} {(r['fruits_per_1k'] or 0):>9.1f} "
              f"{(r['ticks_mean'] or 0):>10.0f} {r['agents_measured']:>7}")
    print("\nrule: an arm earns a paired survival test ONLY if it lowers travel/fruit meaningfully")
    print("(and the wrong-direction control must raise it, or the screen is not measuring what it claims)")


if __name__ == "__main__":
    main()
