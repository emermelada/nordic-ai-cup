"""Isolate the residual non-determinism: expert vs expert (same seed) and learned vs learned in one
process. If the EXPERT also wobbles, it is a simulator-level property, not a policy property."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi

for tag, mk in (("expert", lambda: (im.expert_policy(), im.bc.reset_memory)),
                ("learned", lambda: (lambda p: (p, p.reset))(pi.ImitationPolicy(im.MODEL, device="cpu")))):
    out = []
    for rep in (1, 2, 3):
        fn, rf = mk()
        rows = im.eval_policy(fn, [1000, 1100], horizon=3000, reset_fn=rf, label="%s%d" % (tag, rep))
        out.append([(r["seed"], r["ticks"], r["spawns"], r["predated"], r["score"]) for r in rows])
    same = out[0] == out[1] == out[2]
    print("%-8s identical across 3 runs: %s" % (tag, same))
    for i, o in enumerate(out):
        print("   run%d %s" % (i + 1, o))
