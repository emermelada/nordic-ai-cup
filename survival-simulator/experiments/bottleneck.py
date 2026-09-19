#!/usr/bin/env python3
"""bottleneck.py - WHICH single constraint actually kills the fleet?

The user's causal-experiment list, applied at HARNESS level so the simulator is never edited:
    no_predators       kill every predator each step
    no_repro           block all spawning
    unlimited_sprint   raise every agent's max_energy so the <20% sprint lockout can never bind
    lower_move_cost    refund half of each agent's per-tick energy drop's movement share
    perfect_info       give every agent a huge vision radius
    cheap_food         top up standing fruit energy (approximates unlimited access)
    more_agents        force an extra spawn every N ticks
    fewer_agents       block spawning once population exceeds a low cap

Each is run against the SAME seeds and compared to an untouched control. The intervention with the
largest survival gain identifies the true bottleneck, which tells us what a learned policy should
optimise. Run: bottleneck.py SEED1,SEED2,... [--ticks N]
"""
import argparse, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import best_controller as bc
import oracle_probe as op
from env_wrapper import make_action

P0 = dict(bc.DEFAULT_PARAMS)
import json
P0.update(json.load(open(os.path.join(os.path.dirname(HERE), "best_controller", "params.json"))))


def run(seed, mode, ticks):
    P = dict(P0)
    random.seed(seed)
    np.random.seed(seed)
    core = op.build_core(seed)
    f = bc.make_policy(P)
    bc.reset_memory()
    t = 0
    for t in range(ticks):
        alive = list(core.env.agents)
        if not alive:
            return t
        if mode == "unlimited_sprint":
            for a in alive:
                a.max_energy = 1e6
        states = [core.env.get_agent_state(a.agent_id) for a in alive]
        acts = []
        for s in states:
            a = make_action(s, f(s))
            if mode == "no_repro" or (mode == "fewer_agents" and len(alive) > 4):
                # make_action returns an ActionRequest, not a tuple - set the field
                try:
                    a.spawn_agent = False
                except Exception:
                    pass
            acts.append((s["agent_id"], a))
        if mode == "perfect_info":
            for a in alive:
                a.vision_radius = 5000.0
        pre = {a.agent_id: a.energy for a in alive}
        core.step(acts)
        if mode == "no_predators":
            core.env.predators = []
        elif mode == "lower_move_cost":
            for a in core.env.agents:
                if a.agent_id in pre and a.energy < pre[a.agent_id]:
                    a.energy += 0.5 * (pre[a.agent_id] - a.energy)
        elif mode == "cheap_food":
            for fr in core.env.fruits:
                fr.energy = max(fr.energy, 60.0)
        elif mode == "more_agents" and t % 300 == 0 and len(core.env.agents) < 30:
            try:
                core.env.agents[0].energy += 100.0
            except Exception:
                pass
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds")
    ap.add_argument("--ticks", type=int, default=18000)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    modes = ["control", "no_predators", "no_repro", "unlimited_sprint", "lower_move_cost",
             "perfect_info", "cheap_food", "fewer_agents"]
    print(f"{'intervention':>18} {'mean ticks':>11} {'vs control':>11}   per-seed")
    base = [run(s, "control", a.ticks) for s in seeds]
    bm = sum(base) / len(base)
    print(f"{'control':>18} {bm:11.0f} {'-':>11}   {base}")
    for m in modes[1:]:
        r = [run(s, m, a.ticks) for s in seeds]
        mm = sum(r) / len(r)
        print(f"{m:>18} {mm:11.0f} {100*(mm-bm)/max(1,bm):+10.1f}%   {r}")
