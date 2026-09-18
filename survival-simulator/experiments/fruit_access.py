"""SUPPLY or ACCESS? Decompose the fleet's food income directly from the live world.

Ledger finding that motivates this: income (fruit energy) collapses after t~3000 while burn stays
roughly constant, and every run dies at 6.3-8.4k ticks. Two very different worlds produce that, and
they imply OPPOSITE fixes:

  SUPPLY-limited  the world simply has no food left (production decays as 0.5^(t/300), and fruit
                  ROTS at age>100 sim-seconds ~= 1000 ticks) -> accept the ceiling; optimise the
                  last few hundred ticks.
  ACCESS-limited  food EXISTS but the fleet never reaches it (wasted movement, bad search, too few
                  foragers for the area) -> raising capture rate is a large, real lever.

So measure the world's standing food energy next to the fleet's capture rate, tick by tick:

  supply_E   sum(fruit.energy for fruit in env.fruits)   -- energy physically available NOW
  income_E   energy actually captured (= (score_delta - dt) * 1000, net of predation)
  vis_fruit  fruits visible to ANY agent
  near_min   min distance to a visible fruit, over agents
  fleet_E    sum/max agent energy

Signature of ACCESS limitation: supply_E stays in the hundreds/thousands while income_E is near
zero and agents die with fruit on the map. Signature of SUPPLY limitation: supply_E collapses to
~0 before the fleet does.

Own episode loop: env_wrapper.run_eval_episode does not expose env.fruits/env.predators.

Usage: python fruit_access.py <horizon> <seed> [bucket]
"""
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np

import env_wrapper  # sets up src.* on sys.path
from env_wrapper import make_action
from src.core import SimulationCore

import best_controller as bc
from best_controller import DEFAULT_PARAMS

DT = 0.1


def load_params():
    import json
    P = dict(DEFAULT_PARAMS)
    try:
        with open(os.path.join(REPO, "best_controller", "params.json")) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def run(seed, horizon, bucket):
    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    fn = bc.make_policy(load_params())

    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=5, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    states = core.env.get_agent_state
    ledger = []
    prev_score = 0.0

    for i in range(horizon):
        livestates = [states(a.agent_id) for a in core.env.agents]
        acts = [(s["agent_id"], make_action(s, fn(s))) for s in livestates]
        out = core.step(acts)
        delta = out["score"] - prev_score
        prev_score = out["score"]

        supply = sum(getattr(f, "energy", 0.0) for f in core.env.fruits)
        vis = 0
        near = []
        for s in livestates:
            fobs = [o for o in s.get("observations", []) if isinstance(o, dict) and o.get("type") == "Fruit"]
            vis += len(fobs)
            if fobs:
                near.append(min(o["distance"] for o in fobs))
        en = [s["energy"] for s in livestates]
        ledger.append({
            "t": i, "n": out["num_agents"], "pred": len(core.env.predators),
            "fruits_world": len(core.env.fruits), "supply": supply,
            "income": (delta - DT) * 1000.0,
            "vis": vis, "near_min": min(near) if near else None,
            "e_sum": sum(en), "e_max": max(en) if en else 0.0,
        })
        if out["num_agents"] == 0:
            break

    ticks = len(ledger)
    print("=" * 104)
    print("seed=%d  ticks=%d  final_score=%.1f" % (seed, ticks, prev_score))
    print("bucket     n  pred | fruits  supply_E |  income_E | vis  near_min | fleet_E  e_max")
    for b0 in range(0, ticks, bucket):
        seg = [x for x in ledger if b0 <= x["t"] < b0 + bucket]
        if not seg:
            continue
        nm = [x for x in seg if x["near_min"] is not None]
        print("%5d-%5d %4.1f %5.1f | %6.1f %9.0f | %+9.0f | %3.1f %8s | %7.0f %6.0f"
              % (b0, b0 + bucket, st.mean(x["n"] for x in seg), st.mean(x["pred"] for x in seg),
                 st.mean(x["fruits_world"] for x in seg), st.mean(x["supply"] for x in seg),
                 sum(x["income"] for x in seg), st.mean(x["vis"] for x in seg),
                 ("%.0f" % st.mean(x["near_min"] for x in nm)) if nm else "none",
                 st.mean(x["e_sum"] for x in seg), st.mean(x["e_max"] for x in seg)))
    # the decisive ratios
    tail = [x for x in ledger if x["t"] > ticks - 2000]
    if tail:
        print("-" * 104)
        print("FINAL 2000 ticks: mean supply_E=%.0f  mean fruits_world=%.1f  mean agents=%.2f  "
              "mean fleet_E=%.0f  income_E=%.0f"
              % (st.mean(x["supply"] for x in tail), st.mean(x["fruits_world"] for x in tail),
                 st.mean(x["n"] for x in tail), st.mean(x["e_sum"] for x in tail),
                 sum(x["income"] for x in tail)))
        ag = [x for x in tail if x["n"] > 0]
        visr = [x["vis"] / max(x["n"], 1) for x in ag]
        if visr:
            print("           fruits visible per agent: mean=%.1f  (0 => nothing to steer at)" % st.mean(visr))
        # how often is there food but nobody near it?
        starve = [x for x in ag if x["near_min"] is not None and x["e_max"] < 60]
        if starve:
            print("           ticks with a starving agent (e_max<60) that CAN see fruit: %d (mean nearest %.0f)"
                  % (len(starve), st.mean(x["near_min"] for x in starve)))
        blind = [x for x in ag if x["near_min"] is None]
        print("           ticks where NO agent could see any fruit: %d / %d" % (len(blind), len(ag)))
    return ledger


if __name__ == "__main__":
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
    seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "100").split(",")]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    print("FRUIT-ACCESS controller=%s" % bc.__file__)
    for s in seeds:
        run(s, H, B)
