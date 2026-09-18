"""Timing + end-to-end identity: shipped vs lean controller, horizon 16000, train seeds 100/200/300.

For each seed both policies run the SAME episode (same seed, same reset_fn) with a recorder that
captures every action. We then compare, exactly:
  * every per-agent action (float equality + repr equality -> catches -0.0/0.0 too)
  * survival ticks, score, spawns, fruits eaten, predated, final agents
and report ms/tick for each policy (same definition as experiments/bench_std.py:
1000 * wall_seconds / total_ticks, stop_on_death=True, n_agents=5).

Also measures the isolated per-call CPU cost of the policy and of the served entry point
`best_controller(state)`, which is what a grader's accumulated response wait actually pays for.

Usage: python _time_fast.py [horizon] [seeds]
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
from env_wrapper import make_action, run_eval_episode

H = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [100, 200, 300])]
N_AGENTS = 5


def run(module, seed, horizon):
    acts = []
    states = []

    def rec(i, livestates, actions, out):
        for aid, a in actions:
            acts.append((aid, repr(a.move_distance), repr(a.move_direction), repr(a.turn_angle),
                         bool(a.spawn_agent)))
        states.extend(livestates)

    fn = module.make_policy(module._load_params())
    t0 = time.perf_counter()
    r = run_eval_episode(fn, n_agents=N_AGENTS, seed=seed, horizon=horizon, stop_on_death=True,
                         recorder=rec, reset_fn=module.reset_memory)
    wall = time.perf_counter() - t0
    return r, wall, acts, states


def isolate(module, states):
    """Isolated per-call cost: same call sequence, warm module memory, no profiler."""
    fn = module.make_policy(module._load_params())
    served = module.best_controller
    module.reset_memory()
    for s in states[:100]:
        fn(s)
    t0 = time.perf_counter()
    for s in states:
        fn(s)
    d1 = time.perf_counter() - t0
    module.reset_memory()
    t0 = time.perf_counter()
    for s in states:
        served(s)
    d2 = time.perf_counter() - t0
    n = max(1, len(states))
    return 1000.0 * d1 / n, 1000.0 * d2 / n


def main():
    print("=== TIMING + IDENTITY  horizon=%d seeds=%s n_agents=%d ===" % (H, SEEDS, N_AGENTS))
    rows = []
    all_ok = True
    for seed in SEEDS:
        ra, wa, acts_a, states_a = run(bc, seed, H)
        rf_, wf, acts_f, _ = run(bcf, seed, H)
        msa, msf = isolate(bc, states_a), isolate(bcf, states_a)
        same_acts = acts_a == acts_f
        same_res = all(ra[k] == rf_[k] for k in
                       ("steps", "score", "spawns", "fruits_eaten", "predated", "final_agents"))
        all_ok = all_ok and same_acts and same_res
        rows.append((seed, ra, rf_, wa, wf, msa, msf, same_acts, same_res))
        print("seed=%-4d shipped ticks=%-6d score=%-9.2f spawns=%-4d fruits=%-5d pred=%-5d final=%-3d"
              % (seed, ra["steps"], ra["score"], ra["spawns"], ra["fruits_eaten"],
                 ra["predated"], ra["final_agents"]))
        print("        fast    ticks=%-6d score=%-9.2f spawns=%-4d fruits=%-5d pred=%-5d final=%-3d"
              % (rf_["steps"], rf_["score"], rf_["spawns"], rf_["fruits_eaten"],
                 rf_["predated"], rf_["final_agents"]))
        print("        identity: actions=%s outcomes=%s   ms/tick shipped=%.3f fast=%.3f  "
              "(policy ms/call %.4f->%.4f, served %.4f->%.4f)"
              % ("EXACT" if same_acts else "DIFFER", "EXACT" if same_res else "DIFFER",
                 1000 * wa / ra["steps"], 1000 * wf / rf_["steps"], msa[0], msf[0], msa[1], msf[1]))

    tt_a = sum(r[1]["steps"] for r in rows)
    tt_f = sum(r[2]["steps"] for r in rows)
    wa_ = sum(r[3] for r in rows)
    wf_ = sum(r[4] for r in rows)
    print("\n--- aggregate over %d seeds ---" % len(SEEDS))
    print("shipped: total_ticks=%d wall=%.2fs  ms/tick=%.3f  | score mean=%.1f ticks mean=%.1f"
          % (tt_a, wa_, 1000 * wa_ / tt_a, st.mean([r[1]["score"] for r in rows]),
             st.mean([r[1]["steps"] for r in rows])))
    print("fast   : total_ticks=%d wall=%.2fs  ms/tick=%.3f  | score mean=%.1f ticks mean=%.1f"
          % (tt_f, wf_, 1000 * wf_ / tt_f, st.mean([r[2]["score"] for r in rows]),
             st.mean([r[2]["steps"] for r in rows])))
    print("ms/tick delta: %+.3f ms  (%.1f%% of the shipped tick)"
          % (1000 * wf_ / tt_f - 1000 * wa_ / tt_a,
             100 * (1000 * wf_ / tt_f - 1000 * wa_ / tt_a) / (1000 * wa_ / tt_a)))
    pa = st.mean([r[5][0] for r in rows]); fa = st.mean([r[6][0] for r in rows])
    ps = st.mean([r[5][1] for r in rows]); fs = st.mean([r[6][1] for r in rows])
    print("isolated policy ms/call: %.4f -> %.4f  (%.2fx faster)" % (pa, fa, pa / fa))
    print("served entry ms/call  : %.4f -> %.4f  (%.2fx faster)" % (ps, fs, ps / fs))
    print("\nIDENTITY (all seeds): %s" % ("EXACT - every action + every outcome matches" if all_ok
                                          else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
