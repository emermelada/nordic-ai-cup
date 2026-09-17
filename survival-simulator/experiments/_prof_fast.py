"""Profile the shipped controller's per-request cost in isolation from the sim.

Collects real per-agent states from a shippy episode with a recorder, then
cProfile's ONLY the policy call over those states. Reports top functions and
a per-section breakdown of observation shape (n_edges, n_fruit, n_pred).

Usage: python _prof_fast.py [seed] [n_states]
"""
import cProfile
import json
import os
import pstats
import sys
import time
from io import StringIO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc


def collect_states(seed=100, horizon=3000, n_agents=5):
    states = []

    def rec(i, livestates, acts, out):
        states.extend(livestates)

    run_eval_episode(bc.make_policy(bc._load_params()), n_agents=n_agents, seed=seed,
                     horizon=horizon, stop_on_death=True, recorder=rec,
                     reset_fn=bc.reset_memory)
    return states


def shape(states):
    from collections import Counter
    c = Counter()
    tot = Counter()
    for s in states:
        o = s.get("observations") or []
        k = Counter(x.get("type") for x in o)
        for t in ("Fruit", "Predator", "Agent", "Tree", "Edge"):
            tot[t] += k[t]
            c[t] += 1 if k[t] else 0
        tot["len"] += len(o)
        c["len"] += 1
    n = max(1, len(states))
    return {t: round(tot[t] / n, 2) for t in ("Fruit", "Predator", "Agent", "Tree", "Edge", "len")} , {t: c[t] for t in ("Fruit","Predator","Agent","Tree","Edge")}


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    states = collect_states(seed=seed)
    print("collected %d states" % len(states))
    mean, seen = shape(states)
    print("mean obs per state:", json.dumps(mean))
    print("states with >=1 of type:", json.dumps(seen))

    P = bc._load_params()
    fn = bc.make_policy(P)
    # warm cache (module memory) exactly like a real episode does
    bc.reset_memory()
    for s in states[:50]:
        fn(s)
    t0 = time.perf_counter()
    for s in states:
        fn(s)
    dt = time.perf_counter() - t0
    print("pure-controller ms/call: %.4f  (%d calls)" % (1000.0 * dt / len(states), len(states)))

    bc.reset_memory()
    for s in states[:50]:
        fn(s)
    pr = cProfile.Profile()
    pr.enable()
    for s in states:
        fn(s)
    pr.disable()
    sio = StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats("tottime")
    ps.print_stats(18)
    print(sio.getvalue())


if __name__ == "__main__":
    main()
