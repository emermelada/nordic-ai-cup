#!/usr/bin/env python3
"""ppo_hive.py - recurrent RESIDUAL PPO on the HIVE base (survival-v2, sha256 829e4147).\n\nIdentical learner to ppo_rec.py; the only difference is which controller the residual sits on. hive is\nmeasured +39.4% over the deployed heuristic on 119 fresh paired seeds (t=8.30), so a gain found here\nis worth more than the same percentage on the heuristic. hive decides breeding/genome logic itself and\nthe residual touches ONLY movement direction and magnitude.\n\nOriginal header follows.\n\nppo_rec.py - recurrent RESIDUAL PPO on the deployed controller (the V2 incumbent).

WHY THIS SHAPE (every choice traceable to a measurement; nothing here "because it might help")

  residual, zero-init   best_controller.py IS the policy at initialisation: the actor head's mu is
                        zero-initialised, so at init the corrections are exactly 0 and the policy
                        reproduces V2 bit-for-bit. PPO can only ADD to the incumbent, never replace it.
                        (A BC-cloned recurrent net measured -16.8% paired on 40 unseen seeds, which is
                        why the learned component must be a residual and never a replacement.)
  recurrent (GRU)       probe_enc: one 40-float frame explains ~3% of the incumbent's heading in the
                        blind regime; the same frame plus the incumbent's own memory scalars explains
                        57.5%. 82.4% of agent-ticks are blind, and there the incumbent's action is a
                        function of its own history. The missing capability is STATE, not width.
  actor obs = LEGAL     build_v2() over the fields the grader sends (energy, age, biome, vision,
                        observations). No fruit age, no hidden coordinates, no simulator internals.
                        The critic is the same GRU trunk on the SAME legal tensor - no privileged
                        information exists anywhere in this file.
  2 outputs             heading offset (+-0.5 rad) and speed offset (+-0.3*sprint). Reproduction is
                        left EXACTLY as deployed: every alternative to the deployed gate measured
                        -4.3% to -11.5% over 160 paired seeds, so it is not something to hand a fresh
                        learner. (The action space can be extended later if a gain appears.)

REWARD ALIGNMENT (algebraic, not by feel)
  Competition score = 0.1*ticks + fruit_energy/1000 - sum(victim_energy)/100. The per-tick accrual is
  therefore 0.1. Training reward is r_t = +0.1 for every tick the fleet is alive, undiscounted
  (gamma = 1), and the episode ends at extinction, so
        sum_t r_t = 0.1 * T
  which is IDENTICALLY the dominant term of the competition score. Survival cannot cancel out, and
  there is no shaping (the project's one shaping attempt was gameable by dying). The food term is
  deliberately omitted: it is a +-5% modulation, and a wrong bonus is worse than a 5% omission.

PARALLELISM  N independent simulator processes, one torch thread each, synchronous PPO: collect
  episodes -> compute advantages -> minibatch update -> checkpoint. Workers get episodes in chunks so
  the weights are pickled once per chunk.

    python3 ppo_rec.py --workers 28 --out /opt/nac_hm/ppo --updates 100000
"""
import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                       # noqa: E402
from probe_enc import build_v2, V2_DIM                                   # noqa: E402

# Values below forced by the first run's diagnosis (update 19): the critic fitted well (value MSE
# 3.0 against a target variance of ~676) but the RESIDUAL advantage was only ~0.29 return units =
# ~2.9 ticks, and the actor head had moved to |W|=0.114 (corrections ~0.01 rad). The exploration
# scale was too small to make two states differ in outcome, so there was nothing to learn from.
# Authority doubled; exploration raised to a scale that actually changes behaviour.
CORR = np.array([1.00, 0.50], np.float32)      # heading (rad), speed (fraction of sprint_speed)
HID = 48
LOGSTD_INIT = math.log(0.25)
LOG2PI = math.log(2 * math.pi)


# --------------------------------------------------------------------------------------- policy
def make_net(dim=V2_DIM, hid=HID):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(dim, hid, batch_first=True)
            self.mu = nn.Linear(hid, 2)
            nn.init.zeros_(self.mu.weight)          # <- THE ZERO-CHANGE PATH
            nn.init.zeros_(self.mu.bias)
            self.value = nn.Linear(hid, 1)          # critic: any init, it cannot affect the actor
            self.logstd = nn.Parameter(torch.tensor([LOGSTD_INIT, LOGSTD_INIT]))

        def forward(self, x, h=None):
            o, h = self.gru(x, h)
            return self.mu(o), h

        def value_of(self, x):
            o, _ = self.gru(x)
            return self.value(o).squeeze(-1)
    return Net()


def net_weights(net):
    return {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}


def load_weights(net, w):
    import torch
    net.load_state_dict({k: torch.tensor(v) for k, v in w.items()})


# --------------------------------------------------------------------------------------- rollout
def _mem_update(s, m, bc, n_agents, tick=None):
    """The incumbent's own memory scalars, maintained exactly as build_v2 expects."""
    energy = float(s.get("energy", 0.0) or 0.0)
    pe = m.get("e")
    inc = 0.0 if pe is None else max(0.0, energy - pe) + 0.1
    m["inc_f"] = 0.9 * m.get("inc_f", 0.0) + 0.1 * inc
    m["inc_s"] = 0.99 * m.get("inc_s", 0.0) + 0.01 * inc
    me = max(float(s.get("max_energy", 500.0) or 500.0), 1.0)
    ef = energy / me
    m["ef_s"] = 0.99 * m.get("ef_s", ef) + 0.01 * ef
    m["lock"] = 0.99 * m.get("lock", 0.0) + 0.01 * (1.0 if energy < me / 5.0 else 0.0)
    nf = sum(1 for e in (s.get("observations") or []) if e.get("type") == "Fruit")
    if nf > 0:
        m["tf"], m["df"] = 0.0, 0.0
    else:
        m["tf"] = m.get("tf", 0.0) + 1.0
        m["df"] = m.get("df", 0.0) + float(s.get("speed", 10.0) or 10.0) * 0.12
    m["tick"] = float(tick) if tick is not None else float(bc._SIM_TICK)
    m["n_agents"] = float(n_agents)
    if bc is not None and bc._GS_LAST is not None:   # hive path passes bc=None
        m["rank"] = float(bc._GS_LAST[1])
    m["e"] = energy
    return m


def rollout(job):
    """Same as ppo_rec.rollout but the BASE POLICY IS HIVE (the survival-v2 controller, sha256
    829e4147) instead of the deployed heuristic. hive decides the fleet's actions (its own genome /
    colony / breeding logic is untouched, exactly as the deployed gate is untouched in ppo_rec), and
    the residual only ADDS bounded corrections to movement direction and magnitude. Zero-ish init =>
    the policy starts at hive.

    mode 'base' returns pure hive (the frozen reference for evaluation).
    """
    mode, weights, seeds, horizon, params_path, deterministic = job
    import torch
    torch.set_num_threads(1)
    from hive_v2 import Hive
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    net = None
    if mode == "ppo":
        net = make_net()
        load_weights(net, weights)
        net.eval()

    out = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        hive = Hive(seed=seed)
        core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                              starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
        memo, hid, aidmap = {}, {}, {}
        OBS, U, LGP, AG, TK = [], [], [], [], []
        REW = []
        csum = cmax = 0.0
        cn = 0
        i = 0
        terminated = False
        for i in range(horizon):
            live = list(core.env.agents)
            if not live:
                terminated = True
                break
            states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
            if not states:
                terminated = True
                break
            REW.append(0.1)                      # competition score accrues exactly 0.1 per sim tick
            acts_out = hive.decide({"agent_status": states, "sim_time": i / 10.0, "n_agents": len(states)})
            byid = {s["agent_id"]: s for s in states}
            acted = []
            for d in (acts_out or []):
                aid = d.get("agent_id")
                s = byid.get(aid)
                if s is None:
                    continue
                dr = float(d.get("move_direction", 0.0) or 0.0)
                dist = float(d.get("move_distance", 0.0) or 0.0)
                turn = float(d.get("turn_angle", 0.0) or 0.0)
                spawn = bool(d.get("spawn_agent", False))
                if net is not None:
                    m = memo.setdefault(aid, {})
                    _mem_update(s, m, None, len(states), tick=i / 10.0)
                    x = np.asarray(build_v2(s, m), np.float32)
                    with torch.no_grad():
                        mu, hnew = net(torch.tensor(x)[None, None, :], hid.get(aid))
                        hnew = torch.clamp(hnew, -10.0, 10.0)
                        mu = mu[0, 0]
                        ls = net.logstd.exp()
                        u = mu if deterministic else mu + ls * torch.randn(2)
                        lp = (-0.5 * ((u - mu) / ls) ** 2 - net.logstd - 0.5 * LOG2PI).sum()
                        a2 = (torch.tanh(u) * torch.tensor(CORR)).numpy()
                    hid[aid] = hnew
                    csum += float(np.abs(a2).mean())
                    cmax = max(cmax, float(np.abs(a2).max()) / float(CORR.max()))
                    cn += 1
                    OBS.append(x)
                    U.append(u.numpy().astype(np.float32))
                    LGP.append(float(lp))
                    gid = aidmap.setdefault(aid, len(aidmap))
                    AG.append(gid)
                    TK.append(i)
                    sprint = max(1.0, float(s.get("sprint_speed", 20.0) or 20.0))
                    if a2[0] != 0.0:
                        dr = dr + float(a2[0])
                    if a2[1] != 0.0:
                        dist = max(0.0, dist + float(a2[1]) * sprint)
                acted.append((aid, ActionRequest(agent_id=aid, move_distance=dist,
                                                 move_direction=dr, turn_angle=turn,
                                                 spawn_agent=spawn)))
            core.step(acted)
            keep = {a.agent_id for a in core.env.agents}
            hid = {k: v for k, v in hid.items() if k in keep}
        T = len(REW) if mode == "ppo" else i + 1
        if mode == "ppo":
            out.append({"seed": seed, "T": T, "terminated": bool(terminated),
                        "obs": np.stack(OBS) if OBS else np.zeros((0, V2_DIM), np.float32),
                        "u": np.stack(U) if U else np.zeros((0, 2), np.float32),
                        "logp": np.array(LGP, np.float32),
                        "agent": np.array(AG, np.int32), "tick": np.array(TK, np.int32),
                        "rew": np.array(REW, np.float32),
                        "corr_mean_frac": (csum / max(1, cn)) / float(CORR.mean()),
                        "corr_max_frac": cmax})
        else:
            out.append({"seed": seed, "T": T, "terminated": bool(terminated)})
    return out


# --------------------------------------------------------------------------------------- learner
def build_sequences(net, eps, lam=0.95):
    """Per-episode: per-agent recurrent pass for values, tick-level GAE on the shared reward, then
    per-agent sequences carrying obs/u/logp/adv/ret. Correct recurrent state = per agent."""
    import torch
    seqs = []
    with torch.no_grad():
        for e in eps:
            n = len(e["obs"])
            if n < 4:
                continue
            V = np.zeros(n, np.float32)
            for aid in np.unique(e["agent"]):
                sel = np.where(e["agent"] == aid)[0]
                v = net.value_of(torch.tensor(e["obs"][sel])[None, :, :])[0]
                V[sel] = v.numpy()
            T = e["T"]
            tickV = np.zeros(T, np.float32)
            tickN = np.zeros(T, np.float32)
            np.add.at(tickV, e["tick"], V)
            np.add.at(tickN, e["tick"], 1.0)
            tickV = tickV / np.maximum(tickN, 1.0)
            rew = e["rew"]
            adv = np.zeros(T, np.float32)
            last = 0.0 if e["terminated"] else float(tickV[min(T - 1, len(tickV) - 1)])
            for t in range(T - 1, -1, -1):
                nv = last if t == T - 1 else float(tickV[t + 1])
                d = float(rew[t]) + nv - float(tickV[t])
                adv[t] = d + lam * (adv[t + 1] if t < T - 1 else 0.0)
            ret = adv + tickV
            adv_i = adv[e["tick"]]
            ret_i = ret[e["tick"]]
            for aid in np.unique(e["agent"]):
                sel = np.where(e["agent"] == aid)[0]
                if len(sel) < 2:
                    continue
                seqs.append({"obs": e["obs"][sel], "u": e["u"][sel], "logp": e["logp"][sel],
                             "adv": adv_i[sel], "ret": ret_i[sel]})
    return seqs


def pad_batch(batch):
    L = max(len(s["obs"]) for s in batch)
    D = batch[0]["obs"].shape[1]
    B = len(batch)
    obs = np.zeros((B, L, D), np.float32)
    u = np.zeros((B, L, 2), np.float32)
    lp = np.zeros((B, L), np.float32)
    adv = np.zeros((B, L), np.float32)
    ret = np.zeros((B, L), np.float32)
    mask = np.zeros((B, L), np.float32)
    for i, s in enumerate(batch):
        k = len(s["obs"])
        obs[i, :k] = s["obs"]
        u[i, :k] = s["u"]
        lp[i, :k] = s["logp"]
        adv[i, :k] = s["adv"]
        ret[i, :k] = s["ret"]
        mask[i, :k] = 1.0
    return obs, u, lp, adv, ret, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "ppo"))
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--episodes-per-update", type=int, default=28)
    ap.add_argument("--episodes-per-job", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=900)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=24, help="sequences per minibatch")
    ap.add_argument("--entropy", type=float, default=0.003)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--anchor", type=float, default=0.02, help="L2 pull of the correction toward 0")
    ap.add_argument("--anchor-decay", type=float, default=2000.0)
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-seeds", default="350000-350011")
    ap.add_argument("--eval-horizon", type=int, default=8000)
    ap.add_argument("--train-seed-base", type=int, default=400000)
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--logstd-init", type=float, default=math.exp(LOGSTD_INIT),
                    help="initial pre-tanh exploration sigma: corrections ~ sigma*CORR")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings, one update, then exit")
    a = ap.parse_args()
    if a.smoke:
        a.updates, a.workers, a.episodes_per_update = 2, 2, 2
        a.horizon, a.eval_every, a.eval_seeds, a.eval_horizon = 400, 1, "350000-350001", 600

    import torch
    import torch.nn as nn
    from multiprocessing import get_context

    os.makedirs(a.out, exist_ok=True)
    CKPT = os.path.join(a.out, "ckpt.pt")
    LOG = os.path.join(a.out, "train_log.jsonl")
    EVAL = os.path.join(a.out, "eval_history.jsonl")
    ctx = get_context("spawn")

    net = make_net()
    with torch.no_grad():
        net.logstd.data.fill_(math.log(a.logstd_init))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    start = 0
    if a.resume and os.path.exists(CKPT):
        ck = torch.load(CKPT, weights_only=False)
        net.load_state_dict(ck["sd"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["update"]) + 1
        print(f"  RESUMED at update {start}", flush=True)

    es = []
    for part in a.eval_seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            es += list(range(int(lo), int(hi) + 1))
        elif part:
            es.append(int(part))

    # ZERO-CHANGE CONTRACT, checked before anything trains
    with torch.no_grad():
        zx = torch.zeros(1, 4, V2_DIM)
        mu0, _ = net(zx)
        zok = float(mu0.abs().max()) == 0.0
    print(f"ppo_rec | workers {a.workers} | episodes/update {a.episodes_per_update} | horizon {a.horizon}"
          f" | reward +0.1/tick, gamma 1.0 | corr {CORR.tolist()} | obs dim {V2_DIM}", flush=True)
    print(f"  ZERO-CHANGE CONTRACT: actor output is exactly 0 at init = {zok}  (policy == V2)", flush=True)

    base_eval = None
    for upd in range(start, a.updates):
        t0 = time.time()
        w = net_weights(net)
        s0 = a.train_seed_base + upd * 1000
        seeds = list(range(s0, s0 + a.episodes_per_update))
        jobs = [("ppo", w, seeds[i:i + a.episodes_per_job], a.horizon, a.params, False)
                for i in range(0, len(seeds), a.episodes_per_job)]
        with ctx.Pool(a.workers, maxtasksperchild=None) as pool:
            res = pool.map(rollout, jobs)
        eps = [e for chunk in res for e in chunk]
        t_collect = time.time() - t0

        seqs = build_sequences(net, eps)
        if not seqs:
            print("  no sequences, skipping", flush=True)
            continue
        advs = np.concatenate([s["adv"] for s in seqs])
        mu_a, sd_a = float(advs.mean()), float(advs.std() + 1e-8)
        for s in seqs:
            s["adv"] = (s["adv"] - mu_a) / sd_a

        anchor = a.anchor * max(0.0, 1.0 - upd / max(1.0, a.anchor_decay))
        net.train()
        rng = np.random.default_rng(upd + 7)
        pl = vl = ent = 0.0
        nseq = 0
        for ep in range(a.epochs):
            order = rng.permutation(len(seqs))
            for k in range(0, len(order), a.minibatch):
                batch = [seqs[j] for j in order[k:k + a.minibatch]]
                obs, u, lp_old, adv, ret, mask = pad_batch(batch)
                x = torch.tensor(obs)
                mu, _ = net(x)
                ls = net.logstd.exp()
                lp = (-0.5 * ((torch.tensor(u) - mu) / ls) ** 2 - net.logstd - 0.5 * LOG2PI).sum(-1)
                ratio = torch.exp(lp - torch.tensor(lp_old))
                m = torch.tensor(mask)
                advt = torch.tensor(adv)
                surr = torch.min(ratio * advt, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * advt)
                o, _ = net.gru(x)          # critic: the same trunk on the same legal tensor
                vpred = net.value(o).squeeze(-1)
                vloss = (((vpred - torch.tensor(ret)) ** 2) * m).sum() / m.sum()
                ent_ = (-net.logstd - 0.5 * LOG2PI).sum()
                pol_loss = -(surr * m).sum() / m.sum()
                corr_pen = ((mu ** 2) * m.unsqueeze(-1)).sum() / m.sum()
                loss = pol_loss + a.vf * vloss - a.entropy * ent_ + anchor * corr_pen
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    net.logstd.clamp_(math.log(0.01), math.log(0.6))
                pl += float(pol_loss)
                vl += float(vloss)
                ent += float(ent_)
                nseq += 1
        net.eval()
        toc = time.time() - t0
        rec = {"update": upd, "episodes": len(eps), "seqs": len(seqs),
               "transitions": int(sum(len(s["obs"]) for s in seqs)),
               "mean_T": float(np.mean([e["T"] for e in eps])),
               "extinct_frac": float(np.mean([e["terminated"] for e in eps])),
               "corr_mean_frac": float(np.mean([e["corr_mean_frac"] for e in eps])),
               "corr_max_frac": float(np.max([e["corr_max_frac"] for e in eps])),
               "logstd": float(net.logstd.detach().mean()),
               "head_L1": float(net.mu.weight.abs().sum()),   # is the actor moving at all?
               "adv_mean": mu_a, "adv_sd": sd_a,
               "pol_loss": pl / max(1, nseq), "val_loss": vl / max(1, nseq),
               "entropy": ent / max(1, nseq), "anchor": anchor,
               "t_collect_s": round(t_collect, 2), "t_update_s": round(toc - t_collect, 2),
               "t_total_s": round(toc, 2)}
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        torch.save({"sd": net.state_dict(), "opt": opt.state_dict(), "update": upd, "cfg": vars(a)},
                   CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)
        print(f"  upd {upd}: eps {len(eps)} seqs {len(seqs)} mean_T {rec['mean_T']:.0f} "
              f"extinct {rec['extinct_frac']:.2f} corr {rec['corr_mean_frac']*100:.1f}% "
              f"(max {rec['corr_max_frac']*100:.0f}%) logstd {rec['logstd']:.2f} vloss {rec['val_loss']:.1f} "
              f"| collect {t_collect:.1f}s upd {toc-t_collect:.1f}s", flush=True)

        if a.eval_every and (upd + 1) % a.eval_every == 0:
            if base_eval is None:
                bj = [("base", None, [s_], a.eval_horizon, a.params, True) for s_ in es]
                with ctx.Pool(min(a.workers, len(bj)), maxtasksperchild=None) as pool:
                    br = pool.map(rollout, bj)
                base_eval = {e["seed"]: e["T"] for chunk in br for e in chunk}
            w2 = net_weights(net)
            pj = [("ppo", w2, [s_], a.eval_horizon, a.params, True) for s_ in es]
            with ctx.Pool(min(a.workers, len(pj)), maxtasksperchild=None) as pool:
                pr = pool.map(rollout, pj)
            pol = {e["seed"]: e["T"] for chunk in pr for e in chunk}
            common = [s for s in es if s in base_eval and s in pol]
            diffs = [pol[s] - base_eval[s] for s in common]
            er = {"update": upd, "n": len(common),
                  "base_mean": float(np.mean([base_eval[s] for s in common])),
                  "pol_mean": float(np.mean([pol[s] for s in common])),
                  "paired": float(np.mean(diffs)) if diffs else None,
                  "W": int(sum(1 for d in diffs if d > 0)), "L": int(sum(1 for d in diffs if d < 0))}
            with open(EVAL, "a") as f:
                f.write(json.dumps(er) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"    EVAL @{upd}: base {er['base_mean']:.0f} pol {er['pol_mean']:.0f} "
                  f"paired {er['paired']:+.0f} W/L {er['W']}/{er['L']} n={er['n']}", flush=True)
    print("ppo_rec done", flush=True)


if __name__ == "__main__":
    main()
