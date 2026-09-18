"""Rollout probe (CPU): WHY does the learned policy stop spawning on collapsing seeds?

Records, along the policy's own trajectory AND along the expert's on the same seed:
mean energy fraction, mean predicted spawn probability, spawn-request rate, nearest-predator
distance, and the live-population estimate. Eval seeds are NOT used here (train seeds only).
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi
import imitation_obs as io
import env_wrapper as ew
import best_controller as bc
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else im.MODEL
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "200").split(",")]
H = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
pol = pi.ImitationPolicy(MODEL, device="cpu")
expert = im.expert_policy()

def probe_run(policy_fn, tag, seed, learn_probe=False):
    log = {"e_frac": [], "p_spawn": [], "spawn": [], "pred_min": [], "pop_est": []}
    aug = io.AugObs()
    def fn(state):
        o = ew.build_obs(state)
        if learn_probe:
            with torch.no_grad():
                x = torch.as_tensor((np.concatenate([o, aug.features(state)]) - pol.obs_mean)
                                    / pol.obs_std, dtype=torch.float32)
                cont, lg = pol.net(x[None, :])
                log["p_spawn"].append(float(torch.sigmoid(lg[0])))
            log["pop_est"].append(float(aug.features(state)[2] * 20))
        a = policy_fn(state)
        log["e_frac"].append(float(o[0]))
        log["spawn"].append(float(a[3]))
        log["pred_min"].append(float(o[19] * 300) if o[19] < 1 else 300.0)
        return a
    r = ew.run_eval_episode(fn, n_agents=5, seed=seed, horizon=H, stop_on_death=True,
                            reset_fn=pol.reset)
    d = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in log.items()}
    d.update({"ticks": r["steps"], "spawns": r["spawns"], "spawn_rate": round(r["spawns"] / max(r["steps"], 1), 5),
              "predated": r["predated"], "pop_end": r["final_agents"], "score": round(r["score"], 1)})
    print("[%s seed=%d] %s" % (tag, seed, json.dumps(d)), flush=True)
    return d

for s in SEEDS:
    probe_run(pol, "learned", s, learn_probe=True)
    probe_run(expert, "expert ", s)
