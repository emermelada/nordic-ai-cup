"""Why do harnesses disagree on the SAME seed?

Observed (seed 100, canonical controller, identical params, horizon 16000):
    bench_std.py      -> 5368      (run_eval_episode with trace=True)
    ab_copies.py      -> 4674      (run_eval_episode, no trace)
    fruit_access.py   -> 4674      (own loop, no trace)
    energy_budget.py  -> 5363      (run_eval_episode, no trace)
Two processes of the same harness agree exactly, so this is NOT process-level nondeterminism --
something about the harness changes the episode. This isolates it in ONE process: same policy
object, same seed, run repeatedly with and without `trace`.

If notrace != trace  -> the trace path perturbs the simulation (a real bug worth knowing, because
                        every tuning sweep uses trace=True while the deployed server does not).
If run #1 != run #2  -> there is order/state dependence inside a single process.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

P = dict(DEFAULT_PARAMS)
with open(os.path.join(REPO, "best_controller", "params.json")) as f:
    P.update(json.load(f))

SEED = 100
H = 16000
fn = bc.make_policy(P)

cases = [
    ("notrace #1", dict()),
    ("trace   #1", dict(trace=True)),
    ("notrace #2", dict()),
    ("trace   #2", dict(trace=True)),
    ("notrace #3 all-agents-traced", dict()),
]
print("probe: seed=%d horizon=%d controller=%s" % (SEED, H, bc.__file__))
print("%-28s %8s %10s %8s" % ("case", "ticks", "score", "spawns"))
for label, kw in cases:
    r = run_eval_episode(fn, n_agents=5, seed=SEED, horizon=H, stop_on_death=True,
                         reset_fn=bc.reset_memory, **kw)
    print("%-28s %8d %10.1f %8d" % (label, r["steps"], r["score"], r["spawns"]))
