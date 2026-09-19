#!/usr/bin/env python3
"""survival_autopsy.py - how do the LONG runs survive? The mirror of the death autopsy.

We know why runs die (starved/lockout mid-game). We have never asked how runs reach 12,000 ticks, and
that matters for two reasons:
  1. If long survival comes from a degenerate trick (a corner, a stall, an unreachable spot), then the
     perceived "variance" is really a LOOPHOLE, and the fix is to exploit it deliberately.
  2. If long survival comes from behaviour we can describe, that behaviour is the thing to encode -
     which is exactly the "minimal ruleset that keeps the 1,231 behaviour" the whole plan now rests on.

Reads recordings only (replay.py --traces). Compares the longest runs against the shortest on:
population trajectory, spawning, position on the map (centre distance / spread), travel, lockout share,
and fruit intake. Position comes from the recorded frames, so a "sits in a corner" strategy is visible.

Usage: ./survival_autopsy.py <replay_dir_or_files...>
"""
import glob
import gzip
import json
import math
import os
import statistics as st
import sys


def load(path):
    return json.load(gzip.open(path, "rt")) if path.endswith(".gz") else json.load(open(path))


def episode_stats(path):
    rec = load(path)
    frames = rec["frames"]
    W, H = rec.get("w", 1600), rec.get("h", 1200)
    cx, cy = W / 2.0, H / 2.0
    radius = math.hypot(cx, cy)

    # spatial behaviour: how central are agents, and how clustered are they?
    centr, spread, pops = [], [], []
    for fr in frames:
        a = fr["agents"]
        pops.append(len(a))
        if not a:
            continue
        # frame agent layout: [agent_id, x, y, energy, direction, vision_range, vision_angle, max_energy]
        xs = [ag[1] for ag in a]
        ys = [ag[2] for ag in a]
        centr.append(st.mean([math.hypot(x - cx, y - cy) / radius for x, y in zip(xs, ys)]))
        mx, my = st.mean(xs), st.mean(ys)
        spread.append(st.mean([math.hypot(x - mx, y - my) for x, y in zip(xs, ys)]))

    ag = rec.get("agents") or []
    travel = sum(a["travel"] for a in ag)
    fruits = sum(a["fruits"] for a in ag)
    lock = [a["lockout_frac"] for a in ag if a["ticks"] > 100]
    ticks = rec["final"]["ticks"]
    return {
        "seed": rec["seed"], "ticks": ticks, "score": rec["final"]["score"],
        "pop_max": max(pops) if pops else 0, "pop_mean": st.mean(pops) if pops else 0,
        "pop_end": pops[-1] if pops else 0,
        "centr_mean": st.mean(centr) if centr else None,   # 0 = dead centre, 1 = map edge
        "spread_mean": st.mean(spread) if spread else None,
        "travel_per_1k": 1000.0 * travel / max(1, ticks),
        "fruits_per_1k": 1000.0 * fruits / max(1, ticks),
        "lockout_frac": st.median(lock) if lock else None,
        "spawns": sum(a["spawns"] for a in ag),
        "agents": len(ag),
    }


def main():
    args = sys.argv[1:] or ["."]
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "*.json.gz")) if os.path.isdir(a) else [a])
    rows = sorted((episode_stats(f) for f in files), key=lambda r: r["ticks"])
    if not rows:
        print("no replays found")
        return
    print(f"{len(rows)} episodes, shortest first\n")
    hdr = (f"{'seed':>6} {'ticks':>7} {'popMax':>7} {'popMean':>8} {'popEnd':>7} {'centr':>7} "
           f"{'spread':>7} {'travel/1k':>10} {'fruit/1k':>9} {'lockout':>8} {'spawns':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['seed']:>6} {r['ticks']:>7} {r['pop_max']:>7} {r['pop_mean']:>8.1f} {r['pop_end']:>7} "
              f"{(r['centr_mean'] or 0):>7.2f} {(r['spread_mean'] or 0):>7.1f} {r['travel_per_1k']:>10.0f} "
              f"{r['fruits_per_1k']:>9.1f} {(r['lockout_frac'] or 0):>8.2f} {r['spawns']:>7}")

    k = max(2, len(rows) // 3)
    short, long_ = rows[:k], rows[-k:]
    print(f"\nSHORTEST {k} vs LONGEST {k} (medians)")
    for key in ("ticks", "pop_max", "pop_mean", "centr_mean", "spread_mean", "travel_per_1k",
                "fruits_per_1k", "lockout_frac", "spawns"):
        s = [r[key] for r in short if r[key] is not None]
        l = [r[key] for r in long_ if r[key] is not None]
        if s and l:
            print(f"  {key:>14}  short {st.median(s):>9.2f}   long {st.median(l):>9.2f}   "
                  f"ratio {(st.median(l) / st.median(s) if st.median(s) else float('inf')):>6.2f}")
    print("\nread: 'centr' near 0 = agents sit at the map centre; near 1 = at the edges. A big drop in")
    print("centr for long runs means survival is spatial (a safe place), not behavioural (a skill).")


if __name__ == "__main__":
    main()
