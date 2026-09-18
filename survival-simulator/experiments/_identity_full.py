"""Decisive identity test, controller-isolated and sim-noise-free.

PART A (paired, exact): a fresh process runs the SHIPPED policy over a full-length episode and
  records the exact per-call state sequence + the exact actions it chose. A second fresh process
  replays `best_controller_fast` over that same state sequence and every action is compared with
  repr equality (so -0.0 vs 0.0 counts as a difference). No sim non-determinism can leak in.
PART B (noise control): per-tick action digests of repeated fresh-process episodes of the SAME
  policy, to show whether the harness itself reproduces an identical action stream.

Usage: python _identity_full.py [horizon] [seeds]
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
import best_controller_fast as bcf
from env_wrapper import run_eval_episode

PY = sys.executable
H = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "100,200,300".split(","))]

MODE = os.environ.get("IF_MODE", "all")

# ------------------------------------------------------------------ PART A
if MODE in ("all", "capture"):

    def capture(seed):
        states = []
        acts = []

        def rec(i, livestates, actions, out):
            # actions and livestates are 1:1 and in the same order (see run_eval_episode)
            for aid, a in actions:
                acts.append((aid, repr(a.move_distance), repr(a.move_direction),
                             repr(a.turn_angle), bool(a.spawn_agent)))
            states.extend(livestates)

        run_eval_episode(bc.make_policy(bc._load_params()), n_agents=5, seed=seed, horizon=H,
                         stop_on_death=True, recorder=rec, reset_fn=bc.reset_memory)
        with open(os.path.join(HERE, "_cap_%d.pkl" % seed), "wb") as f:
            pickle.dump({"states": states, "acts": acts}, f)
        return len(states)

    for seed in SEEDS:
        n = capture(seed)
        print("PART A capture: seed=%d calls=%d (fresh process)" % (seed, n), flush=True)

if MODE == "capture":
    sys.exit(0)

# ------------------------------------------------------------------ PART A replay (this process)
if MODE == "all":
    print("\n--- PART A: fast replayed over the SHIPPED episode's exact state sequence ---")
    a_ok = True
    for seed in SEEDS:
        with open(os.path.join(HERE, "_cap_%d.pkl" % seed), "rb") as f:
            d = pickle.load(f)
        states, acts = d["states"], d["acts"]
        fn = bcf.make_policy(bcf._load_params())
        bcf.reset_memory()
        bad = 0
        first = None
        for k, s in enumerate(states):
            a = fn(s)
            ref = acts[k]
            if (repr(a[0]) != ref[1] or repr(a[1]) != ref[2] or repr(a[2]) != ref[3]
                    or bool(a[3]) != ref[4]):
                bad += 1
                if first is None:
                    first = (k, ref, tuple(a))
        a_ok = a_ok and bad == 0
        print("seed=%-4d calls_compared=%-7d mismatches=%d %s"
              % (seed, len(states), bad, "EXACT" if bad == 0 else "first=%r" % (first,)), flush=True)
    print("PART A RESULT: %s" % ("every action bit-identical over 3 full-length episodes"
                                 if a_ok else "MISMATCH FOUND"))

# ------------------------------------------------------------------ PART B
if MODE in ("all", "noise"):
    print("\n--- PART B: fresh-process action digests of the SAME policy (harness noise control) ---")
    script = os.path.join(HERE, "_ab_episode.py")
    for seed in SEEDS:
        for pol in ("shipped", "fast"):
            digs = []
            for rep in range(3):
                r = subprocess.run([PY, script, pol, str(seed), str(H)], capture_output=True,
                                   text=True, cwd=HERE)
                out = (r.stdout or "").strip().splitlines()
                if r.returncode != 0 or not out:
                    print("  run failed", pol, seed, rep, r.stderr[-400:])
                    continue
                j = json.loads(out[-1])
                digs.append((j["action_sha"], j["ticks"], j["score"], j["spawns"]))
            uniq = sorted(set(d[0] for d in digs))
            print("  seed=%-4d %-8s digests=%s  ticks=%s  -> %s"
                  % (seed, pol, [d[0][:8] for d in digs], [d[1] for d in digs],
                     "ONE digest (reproducible)" if len(uniq) == 1
                     else "%d DIFFERENT digests -> harness is not action-reproducible" % len(uniq)),
                  flush=True)
