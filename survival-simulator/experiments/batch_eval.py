"""Batch evaluate named param configs over seeds. Configs from a JSON file:
   [{"tag": "...", "params": {...}}, ...]
Usage: python batch_eval.py <configs.json> <horizon> <seed,seed,...>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS


def deployed():
    P = dict(DEFAULT_PARAMS)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "best_controller", "params.json")) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def evaluate(overrides, seeds, horizon, tag=""):
    P = deployed(); P.update(overrides)
    fn = bc.make_policy(P)
    steps, scores, fruits, preds = [], [], [], []
    for s in seeds:
        r = run_eval_episode(fn, n_agents=5, seed=s, horizon=horizon, stop_on_death=True,
                             reset_fn=bc.reset_memory)
        steps.append(r["steps"]); scores.append(r["score"])
        fruits.append(r["fruits_eaten"]); preds.append(r["predated"])
    print(f"{tag:34s} med_ticks={np.median(steps):7.0f} mean_ticks={np.mean(steps):7.0f} "
          f"med_score={np.median(scores):7.1f} fruit={np.mean(fruits):6.0f} "
          f"pred={np.mean(preds):5.1f} ticks={[int(x) for x in steps]}", flush=True)
    return {"tag": tag, "steps": steps, "med_steps": float(np.median(steps)),
            "mean_steps": float(np.mean(steps)), "med_score": float(np.median(scores)),
            "fruit": float(np.mean(fruits)), "pred": float(np.mean(preds))}


if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1]))
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [100, 200, 300, 400, 500]
    print(f"horizon={H} seeds={seeds} configs={len(cfgs)}")
    res = [evaluate(c.get("params", {}), seeds, H, c["tag"]) for c in cfgs]
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "batch_last.json"), "w"), indent=1)