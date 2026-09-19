#!/usr/bin/env python3
"""replay_survey.py - cross-episode analysis of recorded replays (Phase 3 groundwork).

The visualizer answers "what do the agents physically do"; this answers "what does that show ACROSS
episodes, and which parts did our old telemetry already know". It reads recordings only, needs no
simulator, and is deliberately platform-independent so the same numbers can be recomputed anywhere.

What it reports per episode: survival ticks, deaths by cause, the lockout share of PREDATION deaths,
how many predation deaths happened with NO predator inside the dying agent's observations (i.e. killed
from outside its own detection), and the derived counters for dead agents (wall-blocked,
turned-from-visible-fruit, oscillation, lockout ticks). Plus the final-state picture: population,
mean/min energy, and fruit still standing on the map - the access-limited signature.

Usage:  ./replay_survey.py <replay_dir_or_files...>
"""
import glob
import gzip
import json
import os
import statistics as st
import sys


def load(path):
    return json.load(gzip.open(path, "rt")) if path.endswith(".gz") else json.load(open(path))


def survey_one(path):
    rec = load(path)
    deaths = [e for e in rec["events"] if e["k"] == "die"]
    eaten = [e for e in deaths if e["cause"] == "eaten"]
    per_cause = {}
    for e in deaths:
        per_cause[e["cause"]] = per_cause.get(e["cause"], 0) + 1
    lock_eaten = [e for e in eaten if e.get("lockout")]
    blind_eaten = [e for e in eaten if e.get("pred_dist") is None]
    wall = [e.get("counters", {}).get("wall_block", 0) for e in deaths]
    turnf = [e.get("counters", {}).get("turned_from_fruit", 0) for e in deaths]
    osc = [e.get("counters", {}).get("oscillate", 0) for e in deaths]
    lockt = [e.get("counters", {}).get("lockout_ticks", 0) for e in deaths]
    ticks_alive = [e.get("counters", {}).get("ticks", 0) for e in deaths]
    pops = [f["n"] for f in rec["frames"]]
    last = rec["frames"][-1]
    collapse = next((f["t"] for f in rec["frames"] if f["n"] <= 2), None)
    return {
        "file": os.path.basename(path), "seed": rec["seed"], "ticks": rec["final"]["ticks"],
        "score": rec["final"]["score"], "causes": per_cause, "n_deaths": len(deaths),
        "eaten": len(eaten), "eaten_lockout_pct": (100 * len(lock_eaten) / len(eaten)) if eaten else None,
        "eaten_blind_pct": (100 * len(blind_eaten) / len(eaten)) if eaten else None,
        "mean_wall_block": st.mean(wall) if wall else 0, "mean_turned_from_fruit": st.mean(turnf) if turnf else 0,
        "mean_oscillate": st.mean(osc) if osc else 0,
        "mean_lockout_frac": (st.mean(lockt) / st.mean(ticks_alive)) if lockt and st.mean(ticks_alive) else 0,
        "pop_max": max(pops) if pops else 0, "collapse_tick": collapse,
        "final_pop": last["n"], "final_e_mean": last["e_mean"], "final_e_min": last["e_min"],
        "fruit_left_on_map": len(last["fruits"]), "preds_at_end": len(last["preds"]),
        "frames": len(rec["frames"]), "walls": len(rec.get("edges", [])),
    }


def main():
    args = sys.argv[1:] or ["."]
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "*.json.gz")) if os.path.isdir(a) else [a])
    if not files:
        print("no replays found"); return
    rows = [survey_one(f) for f in files]
    rows.sort(key=lambda r: r["ticks"])
    print(f"{len(rows)} replays surveyed\n")
    hdr = (f"{'seed':>6} {'ticks':>7} {'score':>7} {'dead':>5} {'eaten':>5} {'eat_lock%':>9} "
           f"{'eat_blind%':>10} {'wall':>5} {'turnFruit':>9} {'osc':>4} {'lockFrac':>8} "
           f"{'popMax':>6} {'collapse':>8} {'endPop':>6} {'endE':>6} {'fruitLeft':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['seed']:>6} {r['ticks']:>7} {r['score']:>7.0f} {r['n_deaths']:>5} {r['eaten']:>5} "
              f"{(r['eaten_lockout_pct'] if r['eaten_lockout_pct'] is not None else -1):>9.0f} "
              f"{(r['eaten_blind_pct'] if r['eaten_blind_pct'] is not None else -1):>10.0f} "
              f"{r['mean_wall_block']:>5.1f} {r['mean_turned_from_fruit']:>9.1f} {r['mean_oscillate']:>4.1f} "
              f"{r['mean_lockout_frac']:>8.2f} {r['pop_max']:>6} "
              f"{(r['collapse_tick'] if r['collapse_tick'] is not None else -1):>8} {r['final_pop']:>6} "
              f"{r['final_e_mean']:>6.0f} {r['fruit_left_on_map']:>9}")
    print("\nAGGREGATE (median across replays)")
    keys = ["ticks", "eaten_lockout_pct", "eaten_blind_pct", "mean_wall_block", "mean_turned_from_fruit",
            "mean_oscillate", "mean_lockout_frac", "pop_max", "collapse_tick", "final_pop",
            "final_e_mean", "fruit_left_on_map"]
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        if vals:
            print(f"  {k:24s} median {st.median(vals):10.2f}   min {min(vals):8.2f}   max {max(vals):8.2f}")
    # the access-limited signature: food standing while the fleet dies
    standing = [r for r in rows if r["fruit_left_on_map"] > 50 and r["final_pop"] <= 3]
    print(f"\nepisodes ending with <=3 agents but >50 fruit standing on the map: {len(standing)}/{len(rows)}")
    print("  (this is the access-limited signature: the world is not short of food, the fleet is)")


if __name__ == "__main__":
    main()
