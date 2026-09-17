"""Phase-2 IMITATION LEARNING for the Survival Simulator: collect -> BC train -> eval vs expert.

    python imitation.py train                       # BC on traj_train_im.npz -> imitation_model*.pt
    python imitation.py eval                        # expert vs imitation on HELD-OUT seeds 1000..1700
    python imitation.py tune                        # quick model/knob selection on TRAIN seeds only
    python imitation.py ppo                         # imitation-then-PPO warm start on real multi-agent world
    python imitation.py all

Data: experiments/collect_trajectories.py (expert + 4 deliberate failure regimes, per-step per-agent).
Eval harness: experiments/env_wrapper.py :: run_eval_episode  — the SAME metrics bench_std.py reports,
with reset_fn so episodes are order-invariant.

Seed discipline: TRAIN 100..800 (data), VAL 700/800 (BC model selection), HELD-OUT EVAL 1000..1700
(never trained or tuned on).
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root first (mirrors bench_std.py import order)
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode, build_obs, OBS_DIM, make_action
import best_controller as bc
from best_controller import DEFAULT_PARAMS

if not hasattr(bc, "reset_memory"):
    raise SystemExit("FATAL: wrong best_controller imported (%s)" % getattr(bc, "__file__", "?"))

import policy_imitation as pi
from policy_imitation import MLPPolicy, ImitationPolicy, DIST_SCALE, DIR_SCALE

TRAIN_SEEDS = list(range(100, 900, 100))     # 100..800
VAL_SEEDS = [700, 800]                       # held out of BC training (model selection only)
EVAL_SEEDS = list(range(1000, 1800, 100))    # 1000..1700 HELD OUT
HORIZON = 16000
N_AGENTS = 5
DATA = os.path.join(HERE, "traj_train_im.npz")
MODEL = os.path.join(HERE, "imitation_model.pt")
MODEL_EXPERT_ONLY = os.path.join(HERE, "imitation_model_expert.pt")
RESULTS = os.path.join(HERE, "imitation_results.json")
PPO_MODEL = os.path.join(HERE, "imitation_ppo.pt")


# ------------------------------------------------------------------ expert reference policy
def baseline_params():
    P = dict(DEFAULT_PARAMS)
    try:
        with open(os.path.join(HERE, "best_controller", "params.json")) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def expert_policy():
    return bc.make_policy(baseline_params())


# ------------------------------------------------------------------ BC training
def load_dataset(path=DATA, exclude_seeds=(), variant_weights=None, subsample=1):
    d = np.load(path, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    act = d["act"].astype(np.float32)
    seed = d["seed"].astype(np.int64)
    vid = d["variant_id"].astype(np.int64)
    names = [str(x) for x in d["variant_names"]]
    keep = np.ones(len(seed), bool)
    for s in exclude_seeds:
        keep &= seed != int(s)
    idx = np.where(keep)[0]
    if subsample > 1:
        idx = idx[::subsample]
    w = np.ones(len(idx), np.float32)
    if variant_weights:
        for vi, nm in enumerate(names):
            w[vid[idx] == vi] = float(variant_weights.get(nm, 1.0))
    return obs[idx], act[idx], seed[idx], vid[idx], names, w


def train_bc(path=DATA, out=MODEL, variant_filter=None, variant_weights=None, hidden=(256, 128),
             epochs=14, batch=8192, lr=1e-3, spawn_loss_weight=1.0, seed=0, subsample=1, verbose=True):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, act, sd, vid, names, _ = load_dataset(path, exclude_seeds=VAL_SEEDS, subsample=subsample)
    if variant_filter:  # e.g. {"expert"} -> train only on the deployed expert's own trajectories
        m = np.isin(vid, [names.index(v) for v in variant_filter])
        obs, act, sd, vid = obs[m], act[m], sd[m], vid[m]
    # sample weights (variant mixture control)
    w = np.ones(len(obs), np.float32)
    if variant_weights:
        for vi, nm in enumerate(names):
            w[vid == vi] = float(variant_weights.get(nm, 1.0))

    mu, sg = obs.mean(0), np.maximum(obs.std(0), 1e-3)
    X = (obs - mu) / sg
    y_dist = np.clip(act[:, 0] / DIST_SCALE, 0.0, 1.0)
    y_dir = act[:, 1] / DIR_SCALE
    y_spawn = act[:, 3]
    pos = float(y_spawn.mean())
    pos_weight = float(np.clip((1.0 - pos) / max(pos, 1e-6) * 0.15, 1.0, 300.0))
    if verbose:
        print("BC data: %d samples (val seeds %s excluded) | variants %s" % (len(X), VAL_SEEDS, names))
        print("  spawn positive rate %.4f -> pos_weight %.1f | dist mean %.2f dir|mean| %.2f"
              % (pos, pos_weight, float(act[:, 0].mean()), float(np.abs(act[:, 1]).mean())))

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = MLPPolicy(OBS_DIM, hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.tensor(y_dist, dtype=torch.float32),
        torch.tensor(y_dir, dtype=torch.float32), torch.tensor(y_spawn, dtype=torch.float32),
        torch.tensor(w))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=dev))
    hist = []
    for ep in range(epochs):
        net.train()
        tot = 0.0
        nb = 0
        for xb, db, rb, sb, wb in dl:
            xb, db, rb, sb, wb = [t.to(dev) for t in (xb, db, rb, sb, wb)]
            cont, logit = net(xb)
            ld = ((torch.sigmoid(cont[:, 0]) - db) ** 2) * wb
            lr_ = ((cont[:, 1] - rb) ** 2) * wb
            ls = bce(logit, sb) * wb * spawn_loss_weight
            loss = (ld.mean() + lr_.mean() + ls.mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        hist.append(tot / max(nb, 1))
        if verbose:
            print("  epoch %2d  loss %.5f" % (ep + 1, hist[-1]), flush=True)

    torch.save({"state_dict": net.state_dict(), "obs_mean": mu, "obs_std": sg,
                "config": {"obs_dim": OBS_DIM, "hidden": list(hidden), "lr": lr, "epochs": epochs,
                           "batch": batch, "spawn_loss_weight": spawn_loss_weight,
                           "variant_filter": variant_filter, "variant_weights": variant_weights,
                           "subsample": subsample, "pos_weight": pos_weight, "loss_hist": hist,
                           "train_samples": int(len(X)),
                           "train_seeds": sorted(set(sd.tolist()))},
                "expert_params": baseline_params()}, out)
    if verbose:
        print("saved %s" % out)
    return out


# ------------------------------------------------------------------ action-imitation report
def bc_metrics(model_path, path=DATA, seed_filter=VAL_SEEDS):
    d = np.load(path, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    act = d["act"].astype(np.float32)
    sd = d["seed"].astype(np.int64)
    m = np.isin(sd, seed_filter)
    obs, act = obs[m], act[m]
    p = ImitationPolicy(model_path, cooldown_enabled=False)
    X = (obs - p.obs_mean) / p.obs_std
    import torch
    with torch.no_grad():
        cont, logit = p.net(torch.tensor(X, dtype=torch.float32, device=p.device))
    dist = DIST_SCALE * torch.sigmoid(cont[:, 0]).cpu().numpy()
    drc = DIR_SCALE * cont[:, 1].cpu().numpy()
    prob = torch.sigmoid(logit).cpu().numpy()
    err_dist = np.abs(dist - act[:, 0])
    # angle error on the wrapped difference (direction is a relative heading -> wrap at +-pi)
    dd = np.arctan2(np.sin(drc - act[:, 1]), np.cos(drc - act[:, 1]))
    err_dir = np.abs(dd)
    nz = act[:, 1] != 0
    thr = 0.5
    pred = prob > thr
    gt = act[:, 3] > 0.5
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "val_seeds": list(seed_filter), "n": int(len(act)),
        "dist_mae": round(float(err_dist.mean()), 3),
        "dist_mae_rel": round(float(err_dist.mean() / max(act[:, 0].mean(), 1e-6)), 3),
        "dir_mae_rad": round(float(err_dir.mean()), 3),
        "dir_mae_on_moving": round(float(err_dir[nz].mean()), 3) if nz.any() else None,
        "spawn_expert_rate": round(float(gt.mean()), 5),
        "spawn_pred_rate": round(float(pred.mean()), 5),
        "spawn_precision": round(prec, 3), "spawn_recall": round(rec, 3),
        "spawn_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
        "dist_expert_mean": round(float(act[:, 0].mean()), 3),
        "dist_pred_mean": round(float(dist.mean()), 3),
    }


# ------------------------------------------------------------------ evaluation (own loop, shared metrics)
def eval_policy(policy_fn, seeds, horizon=HORIZON, reset_fn=None, n_agents=N_AGENTS, label=""):
    rows = []
    for s in seeds:
        r = run_eval_episode(policy_fn, n_agents=n_agents, seed=s, horizon=horizon,
                             stop_on_death=True, trace=True, trace_every=100, reset_fn=reset_fn)
        pop_peak = max([t["n"] for t in r["traces"]] or [0])
        row = {"seed": s, "ticks": int(r["steps"]), "score": round(float(r["score"]), 1),
               "spawns": int(r["spawns"]), "predated": int(r["predated"]),
               "fruits_eaten": int(r["fruits_eaten"]), "pop_peak": int(pop_peak),
               "final_agents": int(r["final_agents"])}
        rows.append(row)
        print("    %-14s seed=%-5d ticks=%6d score=%8.1f spawns=%-4d pred=%-4d pop=%d"
              % (label, s, row["ticks"], row["score"], row["spawns"], row["predated"], row["pop_peak"]),
              flush=True)
    return rows


def agg(rows, key):
    xs = [r[key] for r in rows]
    return {"mean": round(st.mean(xs), 1), "median": round(st.median(xs), 1),
            "std": round(st.pstdev(xs), 1), "min": min(xs), "max": max(xs)}


def summarize(rows):
    out = {"episodes": rows}
    for k in ("ticks", "score", "spawns", "predated", "fruits_eaten", "pop_peak"):
        out[k] = agg(rows, k)
    return out


def run_eval(seeds=None, horizon=HORIZON, models=None, tag="eval"):
    seeds = seeds or EVAL_SEEDS
    models = models or [("expert", None), ("imitation", MODEL)]
    res = {"tag": tag, "horizon": horizon, "seeds": list(seeds), "n_agents": N_AGENTS, "policies": {}}
    for name, path in models:
        print("  == policy: %s (horizon=%d, seeds=%s)" % (name, horizon, list(seeds)), flush=True)
        t0 = time.time()
        if path is None:
            rows = eval_policy(expert_policy(), seeds, horizon, reset_fn=bc.reset_memory, label="expert")
        else:
            pol = ImitationPolicy(path)
            rows = eval_policy(pol, seeds, horizon, reset_fn=pi.reset_memory, label=name)
        res["policies"][name] = summarize(rows)
        res["policies"][name]["model_path"] = path
        res["policies"][name]["seconds"] = round(time.time() - t0, 1)
        print("    -> mean %.0f median %.0f std %.0f (%.0fs)"
              % (res["policies"][name]["ticks"]["mean"], res["policies"][name]["ticks"]["median"],
                 res["policies"][name]["ticks"]["std"], res["policies"][name]["seconds"]), flush=True)
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote %s" % RESULTS)
    return res


# ------------------------------------------------------------------ PPO warm start (raw/unshaped reward)
class ActorCritic(nn.Module):
    """BC-initialised actor (same MLP trunk/heads -> contract-compatible) + value head.

    Action parametrisation (normalized space z):
        dist_z ~ N(mu0, s0)   -> dist = 20 * sigmoid(dist_z)     (0..20)
        dir_z  ~ N(mu1, s1)   -> dir  = pi * dir_z               (relative steer)
        spawn  ~ Bernoulli(sigmoid(logit))
    """

    def __init__(self, obs_dim=OBS_DIM, hidden=(256, 128), bc_path=None):
        super().__init__()
        self.actor_backbone = MLPPolicy(obs_dim, hidden)
        if bc_path and os.path.exists(bc_path):
            sd = torch.load(bc_path, map_location="cpu", weights_only=False)
            self.actor_backbone.load_state_dict(sd["state_dict"])
        self.critic = nn.Sequential(nn.Linear(hidden[-1], 128), nn.ReLU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.tensor([-1.0, -1.0]))  # dist_z, dir_z

    def heads(self, x):
        h = self.actor_backbone.trunk(x)
        return self.actor_backbone.cont(h), self.actor_backbone.spawn(h).squeeze(-1), self.critic(h).squeeze(-1)

    def dist_logp(self, dist_z, mu, s):
        return -0.5 * (((dist_z - mu) / s).pow(2) + 2 * torch.log(s) + math.log(2 * math.pi))


class PpoPolicy:
    """Deployable deterministic wrapper for a PPO-trained (or BC-initialised) ActorCritic."""

    def __init__(self, path=PPO_MODEL, device=None, spawn_cooldown=120, spawn_threshold=0.5,
                 stochastic=False):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        self.net = ActorCritic(cfg.get("obs_dim", OBS_DIM), tuple(cfg.get("hidden", (256, 128))))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.net.to(self.device)
        self.obs_mean = np.asarray(ck["obs_mean"], np.float32)
        self.obs_std = np.asarray(ck["obs_std"], np.float32)
        self.spawn_cooldown = int(spawn_cooldown)
        self.spawn_threshold = float(spawn_threshold)
        self.stochastic = bool(stochastic)
        self._clock = {}
        self.cfg = cfg
        self.path = path

    def reset(self):
        self._clock = {}

    def __call__(self, state):
        z = build_obs(state)
        with torch.no_grad():
            x = torch.as_tensor((z - self.obs_mean) / self.obs_std, dtype=torch.float32,
                                device=self.device)[None, :]
            cont, sp, _ = self.net.heads(x)
            if self.stochastic:
                s = torch.exp(self.net.log_std.clamp(-3.5, 0.0))
                dist_z = cont[0, 0] + s[0] * torch.randn((), device=self.device)
                dir_z = cont[0, 1] + s[1] * torch.randn((), device=self.device)
                p = torch.sigmoid(sp[0])
            else:
                dist_z, dir_z, p = cont[0, 0], cont[0, 1], torch.sigmoid(sp[0])
            dist = DIST_SCALE * float(torch.sigmoid(dist_z))
            direction = DIR_SCALE * float(dir_z)
        spawn = 0.0
        aid = int(state.get("agent_id", 0))
        cd = self._clock.get(aid, 0)
        if cd > 0:
            self._clock[aid] = cd - 1
        elif float(p) > self.spawn_threshold:
            spawn = 1.0
            self._clock[aid] = self.spawn_cooldown
        return [float(dist), float(direction), 0.0, spawn]


def ppo_warmstart(bc_path=MODEL, out=PPO_MODEL, iterations=20, ticks_per_iter=2000,
                  train_seeds=(100, 200, 300), lr=3e-4, gamma=0.995, lam=0.95, clip=0.2,
                  epochs=4, minibatch=2048, print_every=2):
    """PPO on the REAL multi-agent world with the RAW grader payoff (no -dt shaping).

    reward_t = st["score"] - last_score == dt + fruit.energy/1000 - victim.energy/100
    Episode ends when num_agents == 0 (grader semantics). All agents share one policy.
    """
    import torch
    from src.core import SimulationCore
    from env_wrapper import make_action

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = ActorCritic(OBS_DIM, (256, 128), bc_path).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    d = np.load(DATA, allow_pickle=True)
    mu, sg = d["obs"].astype(np.float32).mean(0), np.maximum(d["obs"].astype(np.float32).std(0), 1e-3)

    log = []
    for it in range(iterations):
        sd = train_seeds[it % len(train_seeds)]
        core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=N_AGENTS,
                              starting_predators=0, starting_fruits=32, starting_trees=50, seed=sd)
        last_score = 0.0
        traj = {}   # aid -> dict of lists
        ep_reward = 0.0
        steps = 0
        ep_scores = []
        for i in range(ticks_per_iter):
            steps = i + 1
            live = [(a.agent_id, core.env.get_agent_state(a.agent_id)) for a in core.env.agents]
            live = [(aid, stt) for aid, stt in live if stt is not None]
            if live:
                Z = np.stack([(build_obs(stt) - mu) / sg for _, stt in live])
                with torch.no_grad():
                    x = torch.as_tensor(Z, dtype=torch.float32, device=dev)
                    cont, sp, val = net.heads(x)
                    s = torch.exp(net.log_std.clamp(-3.5, 0.0))
                    dist_z = cont[:, 0] + s[0] * torch.randn(len(live), device=dev)
                    dir_z = cont[:, 1] + s[1] * torch.randn(len(live), device=dev)
                    p = torch.sigmoid(sp)
                    spawn = (torch.rand(len(live), device=dev) < p).float()
                    logp = (net.dist_logp(dist_z, cont[:, 0], s[0])
                            + net.dist_logp(dir_z, cont[:, 1], s[1])
                            + torch.distributions.Bernoulli(p).log_prob(spawn))
                acts = {}
                for j, (aid, _) in enumerate(live):
                    acts[aid] = float(DIST_SCALE * torch.sigmoid(dist_z[j]))
                    acts[aid] = (acts[aid], DIR_SCALE * float(dir_z[j]), 0.0, bool(spawn[j] > 0.5))
                reqs = [(aid, make_action(core.env.get_agent_state(aid), acts[aid])) for aid, _ in live]
                out = core.step(reqs)
                r = float(out["score"] - last_score)
                last_score = float(out["score"])
                ep_reward += r
                for j, (aid, _) in enumerate(live):
                    tr = traj.setdefault(aid, {"obs": [], "dz": [], "az": [], "sp": [], "logp": [],
                                               "val": [], "rew": [], "done": []})
                    tr["obs"].append(Z[j]); tr["dz"].append(float(dist_z[j])); tr["az"].append(float(dir_z[j]))
                    tr["sp"].append(float(spawn[j])); tr["logp"].append(float(logp[j]))
                    tr["val"].append(float(val[j])); tr["rew"].append(r); tr["done"].append(0.0)
                if out["num_agents"] == 0:
                    break
            else:
                break
        ep_scores.append(last_score)

        # GAE per agent trajectory (terminal at death/horizon)
        O, DZ, AZ, SP, LP, VA, AD, RT = [], [], [], [], [], [], [], []
        for aid, tr in traj.items():
            adv = 0.0
            n = len(tr["rew"])
            advs = np.zeros(n, np.float32)
            for t in range(n - 1, -1, -1):
                nextv = tr["val"][t + 1] if t + 1 < n else 0.0
                nonterm = 0.0 if t == n - 1 else 1.0
                delta = tr["rew"][t] + gamma * nextv * nonterm - tr["val"][t]
                adv = delta + gamma * lam * nonterm * adv
                advs[t] = adv
            ret = advs + np.asarray(tr["val"], np.float32)
            O.append(np.stack(tr["obs"])); DZ += tr["dz"]; AZ += tr["az"]; SP += tr["sp"]
            LP.append(np.asarray(tr["logp"], np.float32)); VA.append(np.asarray(tr["val"], np.float32))
            AD.append(advs); RT.append(ret)
        if not O:
            continue
        O = np.concatenate(O).astype(np.float32); DZ = np.asarray(DZ, np.float32)
        AZ = np.asarray(AZ, np.float32); SP = np.asarray(SP, np.float32)
        LP = np.concatenate(LP); VA = np.concatenate(VA)
        AD = np.concatenate(AD); RT = np.concatenate(RT)
        AD = (AD - AD.mean()) / max(AD.std(), 1e-6)
        n = len(O)
        idx = np.arange(n)
        stats = {}
        for ep in range(epochs):
            np.random.shuffle(idx)
            for b0 in range(0, n, minibatch):
                b = idx[b0:b0 + minibatch]
                x = torch.as_tensor(O[b], dtype=torch.float32, device=dev)
                cont, sp, val = net.heads(x)
                s = torch.exp(net.log_std.clamp(-3.5, 0.0))
                dz = torch.as_tensor(DZ[b], device=dev); az = torch.as_tensor(AZ[b], device=dev)
                spl = torch.as_tensor(SP[b], device=dev)
                p = torch.sigmoid(sp)
                logp = (net.dist_logp(dz, cont[:, 0], s[0]) + net.dist_logp(az, cont[:, 1], s[1])
                        + torch.distributions.Bernoulli(p).log_prob(spl))
                old = torch.as_tensor(LP[b], device=dev)
                ratio = torch.exp(logp - old)
                a = torch.as_tensor(AD[b], device=dev); rt = torch.as_tensor(RT[b], device=dev)
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                vloss = ((val - rt) ** 2).mean()
                ent = torch.distributions.Bernoulli(p).entropy().mean()
                loss = pg + 0.5 * vloss - 0.01 * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats = {"pg": float(pg), "v": float(vloss), "ent": float(ent)}
        log.append({"iter": it, "seed": sd, "ticks": steps, "raw_score": round(last_score, 1),
                    "samples": int(n), **{k: round(v, 4) for k, v in stats.items()}})
        if it % print_every == 0 or it == iterations - 1:
            print("  ppo it %2d seed=%-4d ticks=%5d raw_score=%7.1f n=%d pg=%.3f v=%.3f ent=%.3f"
                  % (it, sd, steps, last_score, n, stats.get("pg", 0), stats.get("v", 0),
                     stats.get("ent", 0)), flush=True)

    torch.save({"state_dict": net.state_dict(), "obs_mean": mu, "obs_std": sg,
                "config": {"obs_dim": OBS_DIM, "hidden": [256, 128], "bc_init": bc_path,
                           "iterations": iterations, "ticks_per_iter": ticks_per_iter,
                           "reward": "raw score delta (unshaped)", "log": log}}, out)
    print("saved %s" % out)
    return out


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["train", "tune", "eval", "ppo", "all"])
    ap.add_argument("--seeds", default=None, help="comma list of eval seeds")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--no-expert-only", action="store_true")
    ap.add_argument("--skip-ppo", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    if a.stage in ("train", "all"):
        print("== BC training (all 5 expert regimes, expert regime up-weighted 3x) ==")
        train_bc(path=DATA, out=MODEL, variant_weights=None, epochs=14)
        if not a.no_expert_only:
            print("== BC training (expert regime only) ==")
            train_bc(path=DATA, out=MODEL_EXPERT_ONLY, variant_filter=["expert"], epochs=14)
        for p in (MODEL, MODEL_EXPERT_ONLY):
            if os.path.exists(p):
                print("  action-imitation on VAL seeds", p)
                print("   ", json.dumps(bc_metrics(p)))

    if a.stage == "tune":
        # model selection on TRAIN seeds only (never eval seeds)
        tr = [200, 400, 600]
        for name, path in [("imitation", MODEL), ("expert_only", MODEL_EXPERT_ONLY)]:
            if not os.path.exists(path):
                continue
            pol = ImitationPolicy(path)
            rows = eval_policy(pol, tr, horizon=6000, reset_fn=pi.reset_memory, label=name)
            print("  %s train-seed median %.0f mean %.0f" % (name, agg(rows, "ticks")["median"],
                                                             agg(rows, "ticks")["mean"]))

    if a.stage in ("eval", "all"):
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else EVAL_SEEDS
        models = [("expert", None), ("imitation", MODEL)]
        if os.path.exists(MODEL_EXPERT_ONLY):
            models.append(("imitation_expert_only", MODEL_EXPERT_ONLY))
        if os.path.exists(PPO_MODEL):
            print("  == policy: imitation_ppo (BC -> PPO warm start)")
            res = run_eval(seeds=seeds, horizon=a.horizon, models=models)
            t1 = time.time()
            pol = PpoPolicy(PPO_MODEL)
            rows = eval_policy(pol, seeds, a.horizon, reset_fn=pol.reset, label="imitation_ppo")
            res["policies"]["imitation_ppo"] = summarize(rows)
            res["policies"]["imitation_ppo"]["model_path"] = PPO_MODEL
            res["policies"]["imitation_ppo"]["seconds"] = round(time.time() - t1, 1)
            with open(RESULTS, "w") as f:
                json.dump(res, f, indent=1)
        else:
            run_eval(seeds=seeds, horizon=a.horizon, models=models)

    if a.stage in ("ppo", "all") and not a.skip_ppo:
        print("== PPO warm start from BC (raw/unshaped reward, real multi-agent world) ==")
        ppo_warmstart(bc_path=MODEL if os.path.exists(MODEL) else MODEL_EXPERT_ONLY, out=PPO_MODEL)

    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

