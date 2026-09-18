"""HELD-OUT CONFIRM: evolved candidate vs the shipped incumbent, on seeds the search never saw.

Every local sweep this session (tree-seeking, blind movement, population 16/20/26, phase policy)
gained on the search's own seeds and then FAILED on held-out seeds -- textbook overfitting with
~1.6 SE train effects. So the +25% the search now reports (10,552 vs 8,433) is worth nothing until
it reproduces on unseen seeds at a horizon LONGER than the search used (12,000), because the
search's own horizon truncates its best candidates (a seed pinned at exactly 12,000).

Usage:  python w_confirm.py <candidate.json> <seed,seed,...> <horizon>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from env_wrapper import run_eval_episode          # noqa: E402
import best_controller as bc                      # noqa: E402


def ev(params, label, seeds, horizon):
    fn = bc.make_policy(params)
    ticks, fr = [], []
    for sd in seeds:
        r = run_eval_episode(fn, n_agents=5, seed=sd, horizon=horizon,
                             stop_on_death=True, reset_fn=bc.reset_memory)
        ticks.append(r["steps"])
        fr.append(r["fruits_eaten"])
        print(f"  {label:9s} seed {sd:5d}: {r['steps']:6d} ticks  {r['fruits_eaten']:5d} fruits", flush=True)
    print(f"{label}: MEAN {np.mean(ticks):8.1f} ticks {ticks} | fruits {np.mean(fr):7.1f} {fr}", flush=True)
    return float(np.mean(ticks)), ticks


def main():
    cand_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/nac-compute/experiments/evolve_det_best.json"
    seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1000,1001,1002,1003,1004").split(",")]
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    base_path = sys.argv[4] if len(sys.argv) > 4 else ""

    with open(cand_path) as fh:
        blob = json.load(fh)
    cand = blob.get("params", blob)

    # BASELINE MATTERS: the first run compared against bc.DEFAULT_PARAMS, which is NOT what the VPS
    # serves. The deployed controller loads best_controller/params.json, so a deploy decision made
    # against DEFAULT_PARAMS would be against the wrong reference. Pass the live params file here.
    base = dict(bc.DEFAULT_PARAMS)
    base_label = "DEFAULT_PARAMS"
    if base_path:
        try:
            base.update(json.load(open(base_path)))
            base_label = os.path.basename(base_path)
        except Exception as exc:
            print(f"!! could not load baseline {base_path}: {exc}", flush=True)

    print(f"candidate from {cand_path} ({len(cand)} params), horizon {horizon}, seeds {seeds}", flush=True)
    print(f"BASELINE = {base_label} ({baseline_desc(base, base_label)})", flush=True)

    m_c, t_c = ev(cand, "CANDIDATE", seeds, horizon)
    m_i, t_i = ev(base, "BASELINE", seeds, horizon)

    deltas = [c - i for c, i in zip(t_c, t_i)]
    print(f"PER-SEED DELTA: {deltas}", flush=True)
    print(f"RESULT: candidate {m_c:.0f} vs {base_label} {m_i:.0f}  -> {m_c - m_i:+.0f} ticks "
          f"({(m_c / m_i - 1) * 100:+.1f}%)  wins {sum(1 for d in deltas if d > 0)}/{len(deltas)} seeds", flush=True)
    print(f"VERDICT: {'REAL GAIN (deploy candidate)' if m_c - m_i > 500 else 'NOT CONFIRMED (within noise)'}",
          flush=True)


def baseline_desc(base, label):
    keys = ("walk_frac", "repro_frac", "spawn_cooldown", "repro_global_target", "predator_weight")
    return ", ".join(f"{k}={base.get(k)}" for k in keys)


if __name__ == "__main__":
    main()
