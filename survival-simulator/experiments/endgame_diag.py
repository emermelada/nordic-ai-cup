"""End-game diagnostic: what actually kills the LAST agent?

Logs per-tick world telemetry (population, energy min/mean, predators visible, fruit visible,
mean age) and prints the final phase leading to the wipe, bucketed. Answers "why did it die",
not just "did it die" (protocol: diagnose, don't just measure).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

P = dict(DEFAULT_PARAMS)
with open(os.path.join(HERE, "best_controller", "params.json")) as f:
    P.update(json.load(f))
if len(sys.argv) > 1 and sys.argv[1] != "-":
    P.update(json.loads(sys.argv[1]))
fn = bc.make_policy(P)

H = int(os.environ.get("HORIZON", "16000"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "100,200,300").split(",")]

print("END-GAME DIAG horizon=%d seeds=%s params_override=%s" % (H, SEEDS, sys.argv[1:2]))
for s in SEEDS:
    world = []

    def recorder(i, livestates, acts, out, _w=world):
        es = [(st.get("energy") or 0.0) for st in livestates]
        ages = [(st.get("age") or 0.0) for st in livestates]
        sp = (out.get("num_predators") if isinstance(out, dict) else None)
        nf = 0
        npred = 0
        for st in livestates:
            for o in (st.get("observations") or []):
                t = o.get("type")
                if t == "Fruit":
                    nf += 1
                elif t == "Predator":
                    npred += 1
        _w.append((i, len(livestates), min(es) if es else 0.0,
                   sum(es) / len(es) if es else 0.0,
                   max(ages) if ages else 0.0, nf, npred,
                   1 if any(a[1].spawn_agent for a in acts) else 0))

    r = run_eval_episode(fn, n_agents=5, seed=s, horizon=H, stop_on_death=True,
                         reset_fn=bc.reset_memory, recorder=recorder)
    n = len(world)
    print("\n=== seed %d: died at tick %d (score %.1f) ===" % (s, r["steps"], r["score"]))
    print("   tick     n   emin   emean  maxage  fruit_vis  pred_vis  spawns")
    tail_start = max(0, n - 2000)
    step = max(1, (n - tail_start) // 20)
    for j in range(tail_start, n, step):
        t, na, emin, emean, mxage, nf, npred, sp = world[j]
        print("  %6d  %3d  %6.1f %6.1f  %6.1f  %8d  %8d  %5d"
              % (t, na, emin, emean, mxage, nf // max(1, na), npred // max(1, na), sp))
    print("   last tick: %s" % (world[-1],))