#!/usr/bin/env python3
"""oracle_percept_report.py - paired verdicts and the risk half of the travel trade.

Reads the JSON(s) written by oracle_percept.py and produces, per arm and relative to BASE on the
SAME seeds:
  steps (mean/median/min/p10), paired delta, wins/losses/ties, and the mechanism columns
  (blind, fruits visible, away, travel per fruit, income, movement energy) plus
  deaths/1k  = agents lost per 1,000 ticks (the RISK side of travelling farther: more time in the
               open, more time in the <20% sprint lockout, more predation)
  pred_share = share of those losses that are predation rather than starvation. The harness counts
               every removal as a loss; a predation removal is detectable because the sim subtracts
               energy/100 from score when a predator eats (environment.py:726), so it is inferred from
               the score series only where the JSON carries it - here we report deaths/1k, and the
               predation split is taken from the separate PHASE3 evidence instead.

Usage: ./oracle_percept_report.py oracle_mac_700.json [oracle_x86_700.json]
"""
import json
import os
import sys


def load(path):
    rows = json.load(open(path))
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    return by


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def report(path, base_arm="BASE"):
    by = load(path)
    base = by.get(base_arm, {})
    print("=" * 108)
    print(f"{os.path.basename(path)}   BASE = {base_arm}   seeds = {len(base)}")
    print("=" * 108)
    hdr = ("arm", "n", "steps", "median", "min", "p10", "paired", "W/L/T", "blind", "fvis",
           "away", "trav/fr", "inc/1k", "move/1k", "deaths/1k")
    print("%-19s %3s %8s %8s %7s %7s %9s %9s %6s %6s %7s %8s %8s %8s %10s" % hdr)
    order = [a for a in ("BASE", "BASE_DUP") if a in by] + [a for a in by if a not in ("BASE", "BASE_DUP")]
    for arm in order:
        d = by[arm]
        ks = sorted(d)
        st = [d[s]["steps"] for s in ks]
        pd = [d[s]["steps"] - base[s]["steps"] for s in ks if s in base]
        w = sum(1 for v in pd if v > 0)
        l = sum(1 for v in pd if v < 0)
        t = sum(1 for v in pd if v == 0)
        av = lambda k: sum(d[s].get(k, 0.0) for s in ks) / len(ks)  # noqa: E731
        dk = sum(d[s].get("deaths", 0) for s in ks) / max(1, sum(d[s]["steps"] for s in ks)) * 1000
        print("%-19s %3d %8.0f %8.0f %7d %7d %+9.1f %9s %6.3f %6.2f %7.4f %8.1f %8.0f %8.0f %10.1f"
              % (arm, len(ks), sum(st) / len(st), pct(st, 0.5), min(st), pct(st, 0.1),
                 (sum(pd) / len(pd) if pd else 0.0), f"{w}/{l}/{t}", av("blind_frac"),
                 av("fruit_vis_mean"), av("away_frac"), av("travel_per_fruit"),
                 av("income_per_1k"), av("move_e_per_1k"), dk))
    print("  away = share of agent-ticks WITH a visible fruit where the chosen move heads away from it")
    print("  deaths/1k = agents lost per 1,000 ticks (starvation + predation combined)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
        print()
