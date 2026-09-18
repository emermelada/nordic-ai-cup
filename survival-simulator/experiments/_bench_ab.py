"""Interleaved A/B micro-benchmark: isolates policy CPU cost from sim noise.

Collects real per-agent states once per seed, then times shipped and fast policies in alternating
rounds over the same state sequence (warm module memory), reporting per-round ms/call so run-to-run
drift rather than policy cost is visible.

Usage: python _bench_ab.py [rounds] [horizon]
"""
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
import best_controller_fast as bcf
from env_wrapper import run_eval_episode

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
H = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
SEEDS = [100, 200, 300]


def collect(seed):
    states = []
    run_eval_episode(bc.make_policy(bc._load_params()), n_agents=5, seed=seed, horizon=H,
                     stop_on_death=True,
                     recorder=lambda i, s, a, o: states.extend(s), reset_fn=bc.reset_memory)
    return states


def timeit(module, states, reps=1):
    fn = module.make_policy(module._load_params())
    t0 = time.perf_counter()
    for _ in range(reps):
        module.reset_memory()
        for s in states:
            fn(s)
    return 1000.0 * (time.perf_counter() - t0) / (len(states) * reps)


def time_served(module, states, reps=1):
    served = module.best_controller
    module.reset_memory()
    t0 = time.perf_counter()
    for s in states:
        served(s)
    return 1000.0 * (time.perf_counter() - t0) / len(states)


rows = []
for seed in SEEDS:
    states = collect(seed)
    fn_a, fn_f = [], []
    for _ in range(ROUNDS):          # interleave: drift affects both equally
        fn_a.append(timeit(bc, states))
        fn_f.append(timeit(bcf, states))
    sa = time_served(bc, states, 2)
    sf = time_served(bcf, states, 2)
    rows.append((seed, states, fn_a, fn_f, sa, sf))
    print("seed=%-4d states=%-7d  policy ms/call shipped median=%.4f min=%.4f | fast median=%.4f min=%.4f | speedup(median)=%.2fx"
          % (seed, len(states), st.median(fn_a), min(fn_a), st.median(fn_f), min(fn_f),
             st.median(fn_a) / st.median(fn_f)))
    print("        served entry ms/call shipped=%.4f fast=%.4f -> %.2fx"
          % (sa, sf, sa / sf))

ma = st.mean([st.median(r[2]) for r in rows]); mf = st.mean([st.median(r[3]) for r in rows])
na = st.mean([r[4] for r in rows]); nf = st.mean([r[5] for r in rows])
print("\nMEAN over %d seeds: policy %.4f -> %.4f ms/call  (%.2fx, -%.1f%%) | served %.4f -> %.4f ms/call (%.2fx, -%.1f%%)"
      % (len(rows), ma, mf, ma / mf, 100 * (1 - mf / ma), na, nf, na / nf, 100 * (1 - nf / na)))
