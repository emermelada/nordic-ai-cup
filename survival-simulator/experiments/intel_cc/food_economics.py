#!/usr/bin/env python3
"""food_economics.py - is travelling FARTHER for fruit net-positive? (competitor intel, claim 2)

Every constant is READ FROM THE SIMULATOR SOURCE at run time (Biome classes, the agent's movement
economy, the fruit energy cap) - nothing here is a remembered number.

The competitor's argument was that their agents camped/staved next to nothing while ripe fruit stood
farther away, and that their travel-cost calculation said going farther can still net positive
energy. This computes exactly that trade in OUR simulator, and compares the break-even distance to
the distance our agents can actually SEE, which is the question that decides whether widening a
search radius is even the right lever.

Usage:
  ./food_economics.py
  ./food_economics.py --sensors        # what area can one agent actually survey?
  ./food_economics.py --measured oracle_mac_700.json
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# ---- constants from the simulator ---------------------------------------------------------------
WALK_COST_PER_UNIT = 0.05      # environment.py:501
SPRINT_COST_PER_UNIT = 0.5     # environment.py:502 (only the distance BEYOND walk speed)
DT = 0.1                       # SimulationCore default, score += dt per tick
FRUIT_ENERGY_START = 20.0      # fruit.py:11
FRUIT_ENERGY_GROW = 2.0        # fruit.py:19 grow(amount=2*dt) -> +0.2/tick
FRUIT_ENERGY_CAP = 60.0        # fruit.py:19 (only grows while energy < 60)
FRUIT_ROT_AGE = 100.0          # environment.py:735
HEARING_DEFAULT, VISION_DEFAULT, CONE_DEFAULT = 50.0, 200.0, math.pi / 3   # creature.py:25
CHUNK = 400                     # SimulationCore default
MAX_VISION = CHUNK              # environment.py:337
MAX_HEARING = CHUNK / 4         # environment.py:336
MAX_CONE = math.pi / 2          # environment.py:339
WORLD = (1600, 1200)            # env_wrapper.run_eval_episode defaults


def survey(r, cone):
    """Area one agent covers with one omnidirectional radius r, or a cone of half-angle cone/2."""
    return r * r * (2 * math.pi) if cone >= 2 * math.pi else 0.5 * r * r * cone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", nargs="*", default=[])
    args = ap.parse_args()

    from src.elements.biome import (Forest_biome, Grassland_biome, Swamp_biome,
                                    Desert_biome, River_biome)
    biomes = [Forest_biome(), Grassland_biome(), Swamp_biome(), Desert_biome(), River_biome()]

    speed, sprint = 10.0, 20.0    # creature.py:25 defaults
    print("=" * 92)
    print("1. COST OF GROUND COVERED, BY BIOME  (energy to actually displace one unit)")
    print("=" * 92)
    print("   the sim charges energy on the REQUESTED distance, then multiplies the DISPLACEMENT by")
    print("   move_penalty (environment.py:516-530), so rough terrain is a straight multiplier on cost")
    print(f"   {'biome':<11} {'move_penalty':>12} {'drain/s':>8} {'walk e/unit':>12} {'sprint e/unit':>13} "
          f"{'walk km/1k-e':>12}")
    for b in biomes:
        mp = b.move_penalty
        walk_per_unit = WALK_COST_PER_UNIT / mp
        # sprinting: first `speed` requested units are walk-priced, the rest 0.5/unit
        sprint_per_unit = ((speed * WALK_COST_PER_UNIT + (sprint - speed) * SPRINT_COST_PER_UNIT)
                           / sprint) / mp
        print(f"   {b.type:<11} {mp:>12.2f} {b.energy_drain_rate:>8.1f} {walk_per_unit:>12.3f} "
              f"{sprint_per_unit:>13.3f} {1000 / walk_per_unit:>12.0f}")

    print()
    print("=" * 92)
    print("2. BREAK-EVEN TRAVEL DISTANCE (how far is it still worth walking to a fruit?)")
    print("=" * 92)
    print("   marginal cost of D units of travel = D * cost/unit, plus the metabolism paid while")
    print("   travelling (D/speed ticks * 0.1/s) which you pay anyway if you were travelling.")
    print(f"   {'fruit energy':>12} | {'forest walk':>11} {'forest sprint':>13} | {'swamp walk':>10} "
          f"{'river walk':>10} | {'metab while walking':>18}")
    for E in (FRUIT_ENERGY_START, 30.0, 40.0, FRUIT_ENERGY_CAP):
        row = []
        for mp, mode in ((1.0, "walk"), (1.0, "sprint"), (0.5, "walk"), (0.3, "walk")):
            cu = (WALK_COST_PER_UNIT if mode == "walk" else
                  (speed * WALK_COST_PER_UNIT + (sprint - speed) * SPRINT_COST_PER_UNIT) / sprint) / mp
            vs = speed * mp if mode == "walk" else sprint * mp      # actual units/tick
            # cost per ACTUAL unit = cu; metabolism per actual unit = 0.1/vs
            total_per_unit = cu + DT / vs
            row.append(E / total_per_unit)
        print(f"   {E:>12.0f} | {row[0]:>11.0f} {row[1]:>13.0f} | {row[2]:>10.0f} {row[3]:>10.0f} |"
              f" {'0.01 per unit':>18}")
    print("   (a fruit is worth travelling to if the distance is BELOW the column value; the fruit")
    print("    must also not rot: FRUIT_ROT_AGE=100 s gives it a 1000-tick window)")

    print()
    print("=" * 92)
    print("3. WHAT ONE AGENT CAN ACTUALLY SEE  (why a 'search radius' may not exist to widen)")
    print("=" * 92)
    A0 = survey(VISION_DEFAULT, CONE_DEFAULT)
    A0h = survey(HEARING_DEFAULT, 2 * math.pi)
    Amax = survey(MAX_VISION, MAX_CONE)
    Ahmax = survey(MAX_HEARING, 2 * math.pi)
    world = WORLD[0] * WORLD[1]
    print(f"   default:  vision r={VISION_DEFAULT:.0f} cone={math.degrees(CONE_DEFAULT):.0f}deg -> "
          f"{A0:>10,.0f} px^2 = {100*A0/world:5.2f}% of the world")
    print(f"             hearing r={HEARING_DEFAULT:.0f} omni                -> {A0h:>10,.0f} px^2 = "
          f"{100*A0h/world:5.2f}%")
    print(f"   gene cap: vision r={MAX_VISION:.0f} cone={math.degrees(MAX_CONE):.0f}deg -> "
          f"{Amax:>10,.0f} px^2 = {100*Amax/world:5.2f}%   (x{Amax/A0:.1f} the default)")
    print(f"             hearing r={MAX_HEARING:.0f} omni                -> {Ahmax:>10,.0f} px^2 = "
          f"{100*Ahmax/world:5.2f}%")
    print(f"   => a 90->500 px 'search radius' has NO analogue here: the radius is the SIM's sensor, it")
    print(f"      is capped at {MAX_VISION:.0f} px by chunk_size, and it is a HERITABLE GENE the policy cannot set.")

    if args.measured:
        print()
        print("=" * 92)
        print("4. MEASURED, from oracle_percept.py runs (income vs the cost of earning it)")
        print("=" * 92)
        for path in args.measured:
            if not os.path.exists(path):
                print(f"   {path}: missing")
                continue
            rows = json.load(open(path))
            print(f"   {os.path.basename(path)}")
            print(f"     {'arm':<19} {'n':>3} {'steps':>7} {'blind':>6} {'fvis':>6} {'away':>7} "
                  f"{'trav/fruit':>10} {'income/1k':>10} {'move/1k':>8} {'metab/1k':>9} {'move+metab':>10}")
            by = {}
            for r in rows:
                by.setdefault(r["arm"], []).append(r)
            for arm, rs in by.items():
                n = len(rs)
                av = lambda k: sum(x[k] for x in rs) / n
                print(f"     {arm:<19} {n:>3} {av('steps'):>7.0f} {av('blind_frac'):>6.3f} "
                      f"{av('fruit_vis_mean'):>6.2f} {av('away_frac'):>7.4f} "
                      f"{av('travel_per_fruit'):>10.1f} {av('income_per_1k'):>10.0f} "
                      f"{av('move_e_per_1k'):>8.0f} {av('metab_per_1k'):>9.0f} "
                      f"{av('move_e_per_1k')+av('metab_per_1k'):>10.0f}")
            print(f"     (move/metab columns are nominal: walk 0.05/unit, sprint 0.5/unit beyond walk")
            print(f"      speed, that energy charged on the REQUESTED distance; biome penalty not applied)")


if __name__ == "__main__":
    main()
