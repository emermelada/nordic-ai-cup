"""Survival-length tuning harness: evaluate param overrides over seeds at a reduced horizon,
reporting median survival ticks (tick-to-collapse), survivorship score, fruit, births.

Usage:
  python tune_surv.py                       # baseline + presets
  python tune_surv.py <H> <seed,seed,...>   # custom
Prints a table. Reuses env_wrapper.run_eval_episode (no harness rewrite).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS, potential_controller


def evaluate(overrides, seeds, horizon, tag="", verbose=True):
    P = dict(DEFAULT_PARAMS)
    # start from the deployed params.json (merge cleanly)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "best_controller", "params.json")) as f:
            P.update(json.load(f))
    except Exception:
        pass
    P.update(overrides)
    fn = bc.make_policy(P)
    steps, scores, fruits, preds, births = [], [], [], [], []
    for s in seeds:
        r = run_eval_episode(fn, n_agents=5, seed=s, horizon=horizon, stop_on_death=True)
        steps.append(r["steps"]); scores.append(r["score"])
        fruits.append(r["fruits_eaten"]); preds.append(r["predated"])
        births.append(r["steps"])  # placeholder
    d = {
        "tag": tag, "steps": steps, "scores": scores, "fruits": fruits, "preds": preds,
        "med_steps": float(np.median(steps)), "mean_steps": float(np.mean(steps)),
        "med_score": float(np.median(scores)), "mean_score": float(np.mean(scores)),
        "mean_fruit": float(np.mean(fruits)), "n_alive_full": int(sum(1 for x in steps if x >= horizon)),
    }
    if verbose:
        print(f"{tag:28s} med_ticks={d['med_steps']:7.0f} mean_ticks={d['mean_steps']:7.0f} "
              f"med_score={d['med_score']:7.1f} fruit={d['mean_fruit']:6.0f} "
              f"ticks={[int(x) for x in steps]}", flush=True)
    return d


if __name__ == "__main__":
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [100, 200, 300, 400, 500]
    print(f"horizon={H} seeds={seeds}")
    t0 = time.time()
    evaluate({}, seeds, H, tag="CURRENT (deployed)")
    print(f"elapsed {time.time()-t0:.0f}s")