#!/usr/bin/env python3
"""ledger_analysis.py - does income per agent collapse BEFORE the population does?

This is the decisive cheap test of the overshoot hypothesis (H-B) and the banking hypothesis (H-C),
using the per-tick energy ledger that replay.py records. No controller change, no new simulation - just
reading recordings.

THE QUESTION. The fleet peaks near 20 agents and is at <=2 by ~6,700 ticks. Two very different stories
produce that:
  (1) OVERSHOOT: income per agent falls (too many mouths for a fruit flux that decays as 0.5^(t/3000)),
      so agents starve, and the collapse is a correction. Fix = smaller, better-fed fleet.
  (2) SOMETHING ELSE: income per agent stays flat while the fleet dies - then the cause is not food
      supply, and the overshoot story is false.

METHOD. Income is exact from the score (the fruit term is score-visible: each fruit adds energy/1000).
Costs are exact from the simulator's own formulas (metabolism 0.1/tick + age 0.01*age + movement
0.05/unit or speed*0.05+extra*0.5 + 100/birth). Both are normalised PER AGENT, which is the whole
point: a flat total income with a growing population is already falling income per agent.

Usage:  ./ledger_analysis.py <replay.json.gz ...>   (or a directory)
"""
import glob
import gzip
import json
import os
import statistics as st
import sys

WINDOW = 1000          # ticks per reporting window
METAB_PER_TICK = 0.1   # dt * biome_energy_modifier (biome modifier is 1.0 in the common biomes)


def load(path):
    return json.load(gzip.open(path, "rt")) if path.endswith(".gz") else json.load(open(path))


def analyse(path):
    rec = load(path)
    led = rec.get("ledger") or []
    if not led:
        return None
    rows = []
    for lo in range(0, int(led[-1][0]) + 1, WINDOW):
        win = [r for r in led if lo <= r[0] < lo + WINDOW]
        if len(win) < WINDOW // 4:
            continue
        pops = [r[1] for r in win]
        mean_pop = st.mean(pops) if pops else 0
        # income from the score: the fruit term is energy/1000, the constant dt term is 0.1/tick
        income = 0.0
        prev = None
        for r in win:
            if prev is not None:
                d = r[3] - prev
                if d > 0.1 + 1e-9:
                    income += (d - 0.1) * 1000.0
            prev = r[3]
        cmd = sum(r[4] for r in win)
        births = sum(r[5] for r in win)
        deaths = sum(r[6] for r in win)
        lock = sum(r[7] for r in win) / max(1.0, sum(r[1] for r in win))   # lock COUNT / agents = fraction
        e_start, e_end = win[0][2], win[-1][2]
        # costs the fleet actually paid, reconstructed from the balance: absorbed - delta = costs
        costs = income - (e_end - e_start)
        if mean_pop > 0:
            rows.append({"lo": lo, "pop": mean_pop, "income_pa": income / mean_pop,
                         "cmd_pa": cmd / mean_pop, "cost_pa": costs / mean_pop,
                         "need_pa": METAB_PER_TICK * WINDOW + cmd / mean_pop,
                         "births": births, "deaths": deaths, "lock": lock,
                         "e_sum": e_end, "income_total": income})
    return {"seed": rec["seed"], "ticks": rec["final"]["ticks"], "rows": rows}


def main():
    args = sys.argv[1:] or ["."]
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "*.json.gz")) if os.path.isdir(a) else [a])
    analyses = [a for a in (analyse(f) for f in files) if a]
    if not analyses:
        print("no replays with an energy ledger found - re-record with the current replay.py")
        return
    for a in sorted(analyses, key=lambda x: x["ticks"]):
        rows = a["rows"]
        if not rows:
            continue
        peak = max(rows, key=lambda r: r["pop"])
        collapse = next((r for r in rows if r["pop"] <= max(1.0, peak["pop"] * 0.5)), None)
        per_agent = [r["income_pa"] for r in rows]
        # first window where per-agent income drops >15% below the episode's own early level
        early = st.mean(per_agent[:2]) if len(per_agent) >= 2 else per_agent[0]
        drop = next((r for r in rows if r["income_pa"] < 0.85 * early), None)
        print(f"\n=== seed {a['seed']} | {a['ticks']} ticks | peak pop {peak['pop']:.0f} @ t~{peak['lo']} "
              f"| collapse(<=50% peak) t~{collapse['lo'] if collapse else -1} ===")
        print(f"{'window':>8} {'pop':>5} {'inc/agent':>10} {'cost/agent':>11} {'need/agent':>11} "
              f"{'births':>7} {'deaths':>7} {'lock%':>6} {'fleetE':>8}")
        step = max(1, len(rows) // 14)
        for r in rows[::step]:
            print(f"{r['lo']:>8} {r['pop']:>5.1f} {r['income_pa']:>10.0f} {r['cost_pa']:>11.0f} "
                  f"{r['need_pa']:>11.0f} {r['births']:>7} {r['deaths']:>7} {100*r['lock']:>5.0f}% {r['e_sum']:>8.0f}")
        v = (f"per-agent income fell below its early level at t~{drop['lo']}" if drop else
             "per-agent income never fell >15% below its early level")
        print(f"  VERDICT: {v}; population collapsed at t~{collapse['lo'] if collapse else 'never'}")
        if drop and collapse:
            print(f"  -> income/agent led the collapse by {collapse['lo'] - drop['lo']} ticks "
                  f"({'OVERSHOOT SUPPORTED' if drop['lo'] < collapse['lo'] else 'income fell AFTER the collapse'})")
        print(f"  early income/agent {early:.0f} vs metabolic need alone {METAB_PER_TICK*WINDOW:.0f} per 1000t "
              f"-> surplus ratio {early / (METAB_PER_TICK*WINDOW):.2f}x")


if __name__ == "__main__":
    main()
