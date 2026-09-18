"""Behaviour-identity verification: shipped best_controller vs best_controller_fast.

Phase A  replay equality: both policies are driven over the SAME state sequence (in the same
         order, so module memory/epoch/population estimate evolve identically) and every action
         is compared exactly (== on floats AND repr equality, so even -0.0 vs 0.0 is caught).
Phase B  same for the served entry point `best_controller(state)` (params.json memoisation path).
Phase C  the param tables the two modules actually use are identical.

Usage: python _verify_fast.py [replay_horizon] [seeds]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
import best_controller_fast as bcf
from env_wrapper import run_eval_episode

H = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                          "100,200,300,700".split(","))]


def collect(seed, horizon):
    seq = []

    def rec(i, livestates, acts, out):
        seq.extend(livestates)

    run_eval_episode(bc.make_policy(bc._load_params()), n_agents=5, seed=seed, horizon=horizon,
                     stop_on_death=True, recorder=rec, reset_fn=bc.reset_memory)
    return seq


def replay(module, seq, served):
    if served:
        fn = module.best_controller
    else:
        fn = module.make_policy(module._load_params())
    module.reset_memory()
    out = []
    for s in seq:
        a = fn(s)
        out.append((a[0], a[1], a[2], a[3]))
    return out


def diff(a, b, label, seed):
    if len(a) != len(b):
        print("  %-28s MISMATCH length %d vs %d" % (label, len(a), len(b)))
        return 1
    bad = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y or repr(x) != repr(y):
            bad += 1
            if bad <= 3:
                print("  %-28s MISMATCH seed=%d call=%d\n     shipped=%r\n     fast   =%r"
                      % (label, seed, i, x, y))
    if bad:
        print("  %-28s %d/%d actions differ (seed=%d)" % (label, bad, len(a), seed))
        return bad
    print("  %-28s OK  %d/%d actions bit-identical (seed=%d)" % (label, len(a), len(a), seed))
    return 0


def main():
    print("PHASE C: param tables")
    pa, pb = bc._load_params(), bcf._load_params()
    same_keys = set(pa) == set(pb)
    diffs = [k for k in pa if pa.get(k) != pb.get(k)]
    print("  shipped P == fast P: %s (keys_equal=%s, value_diffs=%s)"
          % (pa == pb, same_keys, diffs or "none"))
    print("  DEFAULT_PARAMS identical: %s" % (bc.DEFAULT_PARAMS == bcf.DEFAULT_PARAMS))
    assert pa == pb and bc.DEFAULT_PARAMS == bcf.DEFAULT_PARAMS

    fails = 0
    print("PHASE A/B: replay equality, horizon=%d seeds=%s" % (H, SEEDS))
    for seed in SEEDS:
        seq = collect(seed, H)
        print(" seed=%d  %d policy calls" % (seed, len(seq)))
        fails += diff(replay(bc, seq, False), replay(bcf, seq, False), "potential_controller", seed)
        fails += diff(replay(bc, seq, True), replay(bcf, seq, True), "served best_controller()", seed)

    print("\nRESULT: %s" % ("IDENTICAL on all sampled states" if fails == 0
                            else "FAILED (%d differing actions)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
