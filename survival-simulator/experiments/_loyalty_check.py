#!/usr/bin/env python3
"""_loyalty_check.py - PROVE that a recorded replay is the real simulation, not a look-alike.

Why this exists: the observability work is only worth anything if what you watch is what actually ran.
`replay.py` drives SimulationCore itself (it cannot use env_wrapper.run_eval_episode, because it needs
per-tick state that the wrapper does not expose), so the recorder is a second implementation of the
same loop. A second implementation is exactly where a silent divergence ("diff synth") would hide.

This compares, on identical (seed, horizon, policy, params):
    reference: env_wrapper.run_eval_episode  (the harness every other result in the project used)
    recorder:  replay.record_episode
and requires EXACT agreement on survival ticks and final score. Score is the strict test: it accrues
0.1/tick plus a fruit term, so any divergence in behaviour moves it.

Run this after ANY change to replay.py. A failed check invalidates every replay recorded since.

    ./_loyalty_check.py 1500,1501,1502 4000
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import best_controller as bc                       # noqa: E402
import replay as rp                                # noqa: E402
from env_wrapper import run_eval_episode           # noqa: E402


def reference(seed, horizon, params):
    P = dict(bc.DEFAULT_PARAMS)
    P.update(params)
    bc.reset_memory()
    fn = bc.make_policy(P)
    r = run_eval_episode(fn, n_agents=5, seed=seed, horizon=horizon, stop_on_death=True,
                         reset_fn=bc.reset_memory)
    return r


def main():
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1500,1501,1502").split(",")]
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    params = rp.load_params()
    print(f"loyalty check | horizon {horizon} | seeds {seeds} | params = deployed "
          f"({os.path.basename(rp.DEPLOYED)})", flush=True)
    ok = True
    for sd in seeds:
        ref = reference(sd, horizon, params)
        rec = rp.record_episode(sd, horizon, every=100)        # `every` must not affect the run
        r_ticks, r_score = rec["final"]["ticks"], rec["final"]["score"]
        # tolerance 1e-3: the recording rounds score to 4 decimals for compactness. Exact steps and a
        # score match to 1e-3 IS behavioural fidelity; 1e-6 tests float formatting instead.
        match = (ref["steps"] == r_ticks) and (abs(ref["score"] - r_score) < 1e-3)
        ok &= match
        f = rec["final"]
        print(f"  seed {sd}: reference steps={ref['steps']:6d} score={ref['score']:9.3f} "
              f"fruits={ref['fruits_eaten']:4d} losses={ref['predated']:3d} | "
              f"recorder ticks={r_ticks:6d} score={r_score:9.3f} deaths(eaten/starved/aged)="
              f"{f['eaten']}/{f['starved']}/{f['aged']} -> "
              f"{'IDENTICAL' if match else '*** DIVERGENCE ***'}", flush=True)
    print("\nVERDICT:", "recorder IS the real simulation on these seeds" if ok else
          "DIVERGENCE - replays are NOT trustworthy until this passes")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
