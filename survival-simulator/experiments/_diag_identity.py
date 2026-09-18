"""Find the first action divergence in a full-length episode (controller-isolated replay).

Q1: is the shipped controller reproducible for the same seed in one process?
Q2: driven over the SAME full-length state sequence, where do shipped and fast first differ?
Q3: dump the offending state + both action tuples.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
import best_controller_fast as bcf
from env_wrapper import run_eval_episode

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 100
H = int(sys.argv[2]) if len(sys.argv) > 2 else 8000


def episode(module, seed, horizon):
    acts = []
    states = []

    def rec(i, livestates, actions, out):
        for aid, a in actions:
            acts.append((aid, repr(a.move_distance), repr(a.move_direction), repr(a.turn_angle),
                         bool(a.spawn_agent)))
        states.extend(list(livestates))

    r = run_eval_episode(module.make_policy(module._load_params()), n_agents=5, seed=seed,
                         horizon=horizon, stop_on_death=True, recorder=rec,
                         reset_fn=module.reset_memory)
    return r, acts, states


def replay(module, states):
    fn = module.make_policy(module._load_params())
    module.reset_memory()
    out = [fn(s) for s in states]
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y or repr(x) != repr(y):
            return i
    return None


print("seed=%d horizon=%d" % (SEED, H))
t0 = time.perf_counter()
r1, acts1, states1 = episode(bc, SEED, H)
print("shipped episode: ticks=%d score=%.2f spawns=%d (%.1fs)"
      % (r1["steps"], r1["score"], r1["spawns"], time.perf_counter() - t0))

# Q1 reproducibility of the shipped policy in this process
r2, acts2, _ = episode(bc, SEED, H)
print("Q1 shipped run x2 identical: actions=%s ticks=%s" % (acts1 == acts2, r1["steps"] == r2["steps"]))

# Q2 replay over the full shipped state sequence
a = replay(bc, states1)
b = replay(bcf, states1)
i = first_diff(a, b)
print("Q2 replay over %d calls: %s" % (len(states1), "IDENTICAL" if i is None else "first diff at call %d" % i))
if i is not None:
    st = states1[i]
    print("  state: agent_id=%s age=%.3f energy=%.3f max_e=%s ef=%.4f speed=%s"
          % (st.get("agent_id"), st.get("age", 0), st.get("energy", 0), st.get("max_energy"),
             (st.get("energy", 0) / max(st.get("max_energy", 1) or 1, 1)), st.get("speed")))
    from collections import Counter
    print("  obs types:", dict(Counter(x.get("type") for x in (st.get("observations") or []))))
    print("  shipped=%r\n  fast   =%r" % (tuple(a[i]), tuple(b[i])))

# Q3 sim-level: where do the two episodes diverge?
r3, acts3, _ = episode(bcf, SEED, H)
j = first_diff(acts1, acts3)
print("Q3 episode-level: shipped ticks=%d fast ticks=%d; first action diff at index %s"
      % (r1["steps"], r3["steps"], j if j is not None else "none"))
