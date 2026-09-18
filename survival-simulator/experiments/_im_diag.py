"""Diagnose WHERE imitation breaks: dist vs dir vs spawn, plus covariate shift.

Compares, on TRAIN seeds only (eval seeds untouched):
  expert / learned / learned-dist+expert-dir / expert-dist+learned-dir
and measures per-step expert-vs-policy action disagreement along the policy's own trajectory.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi
import best_controller as bc

MODEL = sys.argv[1] if len(sys.argv) > 1 else im.MODEL
pol = pi.ImitationPolicy(MODEL)
P = im.baseline_params()
expert = bc.make_policy(P)
seeds = [100, 200]
H = 3000

# ---- 1) offline: steering error split by whether a fruit is visible / regime ----
d = np.load(im.DATA, allow_pickle=True)
sd = d["seed"].astype(np.int64)
oid = d["variant_id"].astype(np.int64)
names = [str(x) for x in d["variant_names"]]
obs = d["obs"].astype(np.float32)
act = d["act"].astype(np.float32)
m = np.isin(sd, im.VAL_SEEDS)
for vi, nm in enumerate(names):
    sel = m & (oid == vi)
    if sel.sum() == 0:
        continue
    X = (obs[sel] - pol.obs_mean) / pol.obs_std
    import torch
    with torch.no_grad():
        cont, _ = pol.net(torch.tensor(X, dtype=torch.float32, device=pol.device))
    pred_dir = pi.DIR_SCALE * cont[:, 1].cpu().numpy()
    tgt = act[sel, 1]
    err = np.abs(np.arctan2(np.sin(pred_dir - tgt), np.cos(pred_dir - tgt)))
    vis = obs[sel, 16] < 0.999                       # nearest-fruit distance feature < 1 => fruit visible
    # reference predictor: copy the nearest visible fruit's local angle (expert's direct-forage rule)
    fang = np.arctan2(obs[sel, 18], obs[sel, 17])
    e_f = np.abs(np.arctan2(np.sin(fang - tgt), np.cos(fang - tgt)))
    print("%-18s n=%7d fruit_vis=%.2f  dirMAE=%.3f (fruit-step %.3f n=%d) | copy-fruit-angle MAE=%.3f"
          % (nm, sel.sum(), float(vis.mean()), float(err.mean()),
             float(err[vis].mean()) if vis.any() else -1, int(vis.sum()), float(e_f.mean())))

# ---- 2) rollout decomposition: which output head actually breaks survival? ----
def hybrid(state, mode, _l=pol):
    l = _l(state)
    e = expert(state)
    if mode == "learned":
        return l
    if mode == "learned_dist":
        return [l[0], e[1], 0.0, e[3]]
    if mode == "learned_dir":
        return [e[0], l[1], 0.0, e[3]]
    if mode == "learned_spawn":
        return [e[0], e[1], 0.0, l[3]]
    return e

for mode in ("expert", "learned", "learned_dist", "learned_dir", "learned_spawn"):
    rows = im.eval_policy(lambda s, mm=mode: hybrid(s, mm), seeds, horizon=H,
                          reset_fn=pi.reset_memory, label=mode)
    print("  %-14s ticks %s" % (mode, [r["ticks"] for r in rows]))

# ---- 3) covariate shift: expert-vs-policy disagreement along the POLICY's trajectory ----
log = {"dir_err": [], "dist_err": [], "expert_dist": [], "pol_dist": [], "spawn_pol": [],
       "spawn_expert": [], "energy": [], "fruit_vis": [], "dev_dist": [], "dev_dir": []}
import env_wrapper as ew

def probe(state):
    l = pol(state)
    e = expert(state)
    o = ew.build_obs(state)
    log["dir_err"].append(abs(np.arctan2(np.sin(l[1] - e[1]), np.cos(l[1] - e[1]))))
    log["dist_err"].append(abs(l[0] - e[0]))
    log["expert_dist"].append(e[0]); log["pol_dist"].append(l[0])
    log["spawn_pol"].append(l[3]); log["spawn_expert"].append(e[3])
    log["energy"].append(state.get("energy", 0.0))
    log["fruit_vis"].append(float(o[16] < 0.999))
    return l

r = ew.run_eval_episode(probe, n_agents=5, seed=100, horizon=H, stop_on_death=True, reset_fn=pi.reset_memory)
print(json.dumps({k: round(float(np.mean(v)), 3) for k, v in log.items()}, indent=1))
print("episode ticks", r["steps"], "spawns", r["spawns"], "predated", r["predated"])
