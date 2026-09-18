"""PAIRED multi-candidate comparison on held-out seeds: which gene group actually helps?

The session's finding: the search winner earns ~30% more fruit than the deployed controller but
survives no longer -- it spends the surplus on children that starve (spawn_cooldown 30 vs 120,
fleet target 17.6 vs 12, absolute spawn gate 34 energy). So decompose the winners into gene groups
and test the cross:
  H1 = winner's EARNING + live's SPENDING      -> is the winner's foraging genuinely better?
  H2 = live's EARNING + winner's SPENDING      -> or is the edge purely breeding?
  H3 = winner's earning + middle breeding      -> is the trade-off smooth, with an interior optimum?
All measured in ONE process on the SAME seeds, so the comparison is paired.

Usage: python w_hybrid.py "LABEL:path,LABEL:path" seeds horizon live_params.json
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


def load(path):
    """Load a params file, tolerating BOTH shapes: flat {k: v} and the search's {"params": {...}}.

    Not tolerating both silently produced a wrong control: without unwrapping, the winner loaded as
    DEFAULT_PARAMS and the test "showed" it scoring 1,948 ticks (the defaults' number) instead of
    ~8,000. A silently-degraded control is worse than no control.
    """
    blob = json.load(open(path))
    p = dict(bc.DEFAULT_PARAMS)
    p.update(blob.get("params", blob) if isinstance(blob, dict) else {})
    return p


def ev(params, label, seeds, horizon):
    fn = bc.make_policy(params)
    tk, fr = [], []
    for sd in seeds:
        r = run_eval_episode(fn, n_agents=5, seed=sd, horizon=horizon,
                             stop_on_death=True, reset_fn=bc.reset_memory)
        tk.append(r["steps"])
        fr.append(r["fruits_eaten"])
    print(f"{label:24s} MEAN {np.mean(tk):8.1f} ticks | fruits {np.mean(fr):7.1f} | seeds {tk}", flush=True)
    return float(np.mean(tk)), float(np.mean(fr)), tk


def main():
    cands = sys.argv[1].split(",")
    seeds = [int(x) for x in sys.argv[2].split(",")]
    horizon = int(sys.argv[3])
    live_path = sys.argv[4]
    print(f"PAIRED TEST: {len(seeds)} held-out seeds, horizon {horizon}", flush=True)

    res = {}
    for spec in cands:
        label, path = spec.split(":", 1)
        res[label] = ev(load(path), label, seeds, horizon)
    res["LIVE"] = ev(load(live_path), "LIVE (deployed)", seeds, horizon)

    print("\n=== SUMMARY (paired on identical seeds) ===", flush=True)
    base = res["LIVE"][0]
    for label, (m, f, tk) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        print(f"  {label:24s} {m:8.1f} ticks  {f:7.1f} fruits   vs LIVE {m - base:+8.1f} ({(m / base - 1) * 100:+5.1f}%)",
              flush=True)


if __name__ == "__main__":
    main()