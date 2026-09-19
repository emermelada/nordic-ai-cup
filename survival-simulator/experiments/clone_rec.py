#!/usr/bin/env python3
"""clone_rec.py - stage 1 of the ML track: can a RECURRENT net reproduce the incumbent at all?

WHY THIS IS THE RIGHT FIRST ML EXPERIMENT (probe_enc.py settled the architecture question)
    Held-out probe, predicting the incumbent's OWN action from the observation frame:
        V1  31 floats (current)   heading R^2 0.086   blind-regime R^2 0.029
        V2  61 floats (richer)    heading R^2 0.083   blind-regime R^2 0.024
        V3  V2 + 5 MEMORY dims    heading R^2 0.570   blind-regime R^2 0.575
    Fifty-nine extra world-state features bought nothing; five scalars of the incumbent's OWN
    internal state (held wander heading, last steer, flee latch) bought a 24x jump. 82.4% of
    agent-ticks are blind, and in that regime the incumbent's heading is a function of its own
    history. So the missing capability is STATE, not width - a recurrent policy, not a wider frame.

    Before optimising anything, the load-bearing question is whether a recurrent policy can even
    STAND IN for the incumbent. If a GRU over the same frame cannot reproduce its survival, the
    formulation is wrong and training is wasted. If it can, we hold a parameterisation that
    CONTAINS the incumbent and gradient methods have something to improve.

STAGING (each stage is independently falsifiable)
    collect : roll the incumbent out, record per-agent frame sequences + its own (heading, speed)
    train   : GRU(61 -> 48) -> 3, behaviour cloning, windows of 64, seed-split train/val
    eval    : run the CLONE on unseen seeds vs the INCUMBENT on the same seeds, paired

    The clone replaces the incumbent's heading and speed ONLY. Turn (always 0.0 in production,
    face_predator=0) and the spawn decision stay with the incumbent, so this isolates the question.

    python3 clone_rec.py --stage collect --seeds 7000-7005 --out /opt/nac_h2h/diag/clone.npz
    python3 clone_rec.py --stage train   --data /opt/nac_h2h/diag/clone.npz --out /opt/nac_h2h/diag/clone.pt
    python3 clone_rec.py --stage eval    --model /opt/nac_h2h/diag/clone.pt --seeds 7200-7239
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from probe_enc import build_v2, V2_DIM  # noqa: E402

WIN = 64
HID = 48


def make_net(dim=V2_DIM, hid=HID):
    """Single definition shared by training and evaluation - a second, locally-defined class in the
    eval path is exactly how a 'module missing forward' bug gets in."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(dim, hid, batch_first=True)
            self.head = nn.Linear(hid, 3)

        def forward(self, x, h=None):
            o, h = self.gru(x, h)
            return self.head(o), h
    return Net()


def load_net(path):
    import torch
    ck = torch.load(path, weights_only=True)
    net = make_net(ck["dim"], ck["hid"])
    net.load_state_dict(ck["sd"])
    net.eval()
    return net


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_params(path):
    import best_controller as bc
    P = dict(bc.DEFAULT_PARAMS)
    with open(path) as f:
        P.update(json.load(f))
    return P


# --------------------------------------------------------------------------- collect
def collect_episode(args):
    seed, horizon, params_path = args
    import random
    import best_controller as bc
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    P = load_params(params_path)
    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    policy = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    memo, seqs, cur = {}, {}, {}
    for i in range(horizon):
        live = list(core.env.agents)
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
        acts = []
        for s in states:
            aid = s["agent_id"]
            m = memo.setdefault(aid, {})
            dist, dr, turn, spawn = policy(s)
            energy = float(s.get("energy", 0.0) or 0.0)
            prev_e = m.get("e")
            inc = 0.0 if prev_e is None else max(0.0, energy - prev_e) + 0.1
            m["inc_f"] = 0.9 * m.get("inc_f", 0.0) + 0.1 * inc
            m["inc_s"] = 0.99 * m.get("inc_s", 0.0) + 0.01 * inc
            max_e = max(float(s.get("max_energy", 500.0) or 500.0), 1.0)
            ef = energy / max_e
            m["ef_s"] = 0.99 * m.get("ef_s", ef) + 0.01 * ef
            m["lock"] = 0.99 * m.get("lock", 0.0) + 0.01 * (1.0 if energy < max_e / 5.0 else 0.0)
            nf = sum(1 for e in (s.get("observations") or []) if e.get("type") == "Fruit")
            if nf > 0:
                m["tf"], m["df"] = 0.0, 0.0
            else:
                m["tf"] = m.get("tf", 0.0) + 1.0
                m["df"] = m.get("df", 0.0) + float(s.get("speed", 10.0) or 10.0) * 0.12
            m["tick"] = float(bc._SIM_TICK)
            m["n_agents"] = float(len(states))
            if bc._GS_LAST is not None:
                m["rank"] = float(bc._GS_LAST[1])
            m["since_birth"] = float(i)
            seqs.setdefault(aid, [[], []])
            seqs[aid][0].append(build_v2(s, m))
            sprint = max(1e-6, float(s.get("sprint_speed", 20.0) or 20.0))
            seqs[aid][1].append([math.cos(dr), math.sin(dr), min(dist / sprint, 1.0)])
            m["e"] = energy
            cur[aid] = i
            acts.append((aid, ActionRequest(agent_id=aid, move_distance=dist, move_direction=dr,
                                            turn_angle=turn, spawn_agent=bool(spawn))))
        core.step(acts)
    X = [np.asarray(v[0], np.float32) for v in seqs.values() if len(v[0]) >= WIN]
    Y = [np.asarray(v[1], np.float32) for v in seqs.values() if len(v[0]) >= WIN]
    return seed, X, Y


def stage_collect(a):
    from multiprocessing import get_context
    seeds = parse_seeds(a.seeds)
    print(f"collect | {len(seeds)} seeds | horizon {a.horizon}", flush=True)
    ctx = get_context("spawn")
    jobs = [(s, a.horizon, a.params) for s in seeds]
    Xs, Ys, meta = [], [], []
    with ctx.Pool(a.workers) as pool:
        for seed, X, Y in pool.imap_unordered(collect_episode, jobs, chunksize=1):
            Xs += X
            Ys += Y
            meta.append((seed, len(X)))
            print(f"  seed {seed}: {len(X)} agent-sequences", flush=True)
    np.savez_compressed(a.data, X=np.concatenate(Xs), Y=np.concatenate(Ys),
                        n=np.array([len(x) for x in Xs]))
    print(f"saved {sum(len(x) for x in Xs)} frames from {len(Xs)} agent-lives -> {a.data}")


# --------------------------------------------------------------------------- train
def _train_core(data, out, steps, tseed, holdout_frac=0.1, verbose=True):
    """One net, one thread. Called in parallel across the population - that is how 64 cores get
    used here. The SIMULATOR is the compute bottleneck, not the net: a 48-unit GRU over a 128x64
    batch is ~13 MFLOP/step, so spreading ONE net over 64 threads spends all its time on barrier
    sync (measured: 3828% CPU and one step per ~2 min, vs 99.9% CPU and ~50 steps/s pinned to one
    core). Parallelism belongs at the level of episodes and nets, never tensors."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(tseed)
    d = np.load(data)
    X, Y, n = d["X"], d["Y"], d["n"]
    starts = np.concatenate([[0], np.cumsum(n)])
    rng = np.random.default_rng(tseed)
    lives = rng.permutation(len(n))
    ntr = int((1.0 - holdout_frac) * len(lives))
    tr_lives, va_lives = lives[:ntr], lives[ntr:]

    def windows(pool, k):
        """EXACTLY k windows total. The first version drew k windows from EVERY life, building an
        88,064-sample / 1.4 GB batch per step - that, not the CPU, was the slowness."""
        idx = []
        for li in rng.choice(pool, size=k, replace=True):
            s, e = starts[li], starts[li + 1]
            if e - s < WIN:
                continue
            o = int(rng.integers(0, e - s - WIN + 1))
            idx.append((s + o, s + o + WIN))
        return idx or [(0, WIN)]

    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.MSELoss()
    B = 128
    val = None
    for step in range(1, steps + 1):
        idx = windows(tr_lives, B)
        xb = torch.tensor(np.stack([X[s:e] for s, e in idx]))
        yb = torch.tensor(np.stack([Y[s:e] for s, e in idx]))
        pred, _ = net(xb)
        loss = lossf(pred, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 300 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                vi = windows(va_lives, 64)
                xv = torch.tensor(np.stack([X[s:e] for s, e in vi]))
                yv = torch.tensor(np.stack([Y[s:e] for s, e in vi]))
                pv, _ = net(xv)
                vl = lossf(pv, yv).item()
                ang = torch.atan2(yv[..., 1], yv[..., 0]) - torch.atan2(pv[..., 1], pv[..., 0])
                ang = torch.atan2(torch.sin(ang), torch.cos(ang)).abs().mean().item()
                spd = (yv[..., 2] - pv[..., 2]).abs().mean().item()
                val = (vl, ang, spd)
                if verbose:
                    print(f"  step {step:5d} train {loss.item():.5f} val {vl:.5f} "
                          f"angMAE {ang:.3f} rad spdMAE {spd:.3f}", flush=True)
            net.train()
    torch.save({"sd": net.state_dict(), "dim": V2_DIM, "hid": HID}, out)
    return {"seed": tseed, "path": out, "val_loss": val[0] if val else None,
            "val_ang": val[1] if val else None, "val_spd": val[2] if val else None}


def train(a):
    r = _train_core(a.data, a.out, a.steps, 0)
    print(f"saved -> {r['path']}  val_ang {r['val_ang']:.3f} val_spd {r['val_spd']:.3f}")


def _train_star(job):
    """module-level unpacker: with the spawn start method the pool pickles the callable, so it must
    be a top-level function taking ONE argument."""
    return _train_core(*job)


def stage_pop(a):
    """A POPULATION of nets, one thread each, all 64 cores busy. Same wall clock as one net and it
    yields a distribution rather than a single lucky init - which is the point."""
    from multiprocessing import get_context
    ctx = get_context("spawn")
    jobs = [(a.data, f"{a.out}.{i}.pt", a.steps, i + 1) for i in range(a.nets)]
    print(f"population | {a.nets} nets x {a.steps} steps, 1 thread each", flush=True)
    out = []
    with ctx.Pool(a.nets) as pool:
        for r in pool.imap_unordered(_train_star, jobs, chunksize=1):
            out.append(r)
            print(f"  net tseed={r['seed']:3d} val_loss {r['val_loss']:.5f} "
                  f"angMAE {r['val_ang']:.3f} spdMAE {r['val_spd']:.3f}", flush=True)
    out.sort(key=lambda r: r["val_ang"])
    json.dump(out, open(a.out + "_pop.json", "w"), indent=1)
    print(f"\nbest clone: {out[0]['path']} (val angMAE {out[0]['val_ang']:.3f} rad)")
    print(f"population val angMAE: min {out[0]['val_ang']:.3f} median "
          f"{out[len(out)//2]['val_ang']:.3f} max {out[-1]['val_ang']:.3f}")


# --------------------------------------------------------------------------- eval
def eval_episode(args):
    seed, horizon, params_path, model_path, arm = args
    import random
    import torch
    import torch.nn as nn
    import best_controller as bc
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    P = load_params(params_path)
    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    base = bc.make_policy(P)

    net = load_net(model_path) if arm == "clone" else None

    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    memo, hid, prev = {}, {}, {}
    for i in range(horizon):
        live = list(core.env.agents)
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
        acts = []
        for s in states:
            aid = s["agent_id"]
            dist_b, dr_b, turn_b, spawn_b = base(s)      # turn/spawn always stay with the base
            if arm == "clone":
                m = memo.setdefault(aid, {})
                energy = float(s.get("energy", 0.0) or 0.0)
                prev_e = m.get("e")
                inc = 0.0 if prev_e is None else max(0.0, energy - prev_e) + 0.1
                m["inc_f"] = 0.9 * m.get("inc_f", 0.0) + 0.1 * inc
                m["inc_s"] = 0.99 * m.get("inc_s", 0.0) + 0.01 * inc
                max_e = max(float(s.get("max_energy", 500.0) or 500.0), 1.0)
                ef = energy / max_e
                m["ef_s"] = 0.99 * m.get("ef_s", ef) + 0.01 * ef
                m["lock"] = 0.99 * m.get("lock", 0.0) + 0.01 * (1.0 if energy < max_e / 5.0 else 0.0)
                nf = sum(1 for e in (s.get("observations") or []) if e.get("type") == "Fruit")
                if nf > 0:
                    m["tf"], m["df"] = 0.0, 0.0
                else:
                    m["tf"] = m.get("tf", 0.0) + 1.0
                    m["df"] = m.get("df", 0.0) + float(s.get("speed", 10.0) or 10.0) * 0.12
                m["tick"] = float(bc._SIM_TICK)
                m["n_agents"] = float(len(states))
                if bc._GS_LAST is not None:
                    m["rank"] = float(bc._GS_LAST[1])
                m["since_birth"] = float(i)
                x = torch.tensor(build_v2(s, m))[None, None, :]
                with torch.no_grad():
                    o, h = net(x, hid.get(aid))
                hid[aid] = h
                o = o[0, 0].numpy()
                dr = math.atan2(float(o[1]), float(o[0]))
                sprint = max(1.0, float(s.get("sprint_speed", 20.0) or 20.0))
                dist = float(np.clip(o[2], 0.0, 1.0)) * sprint
                m["e"] = energy
            else:
                dist, dr = dist_b, dr_b
            acts.append((aid, ActionRequest(agent_id=aid, move_distance=float(dist),
                                            move_direction=float(dr), turn_angle=float(turn_b),
                                            spawn_agent=bool(spawn_b))))
        core.step(acts)
        keep = {a.agent_id for a in core.env.agents}
        hid = {k: v for k, v in hid.items() if k in keep}
    return arm, seed, i + 1


def stage_eval(a):
    from multiprocessing import get_context
    seeds = parse_seeds(a.seeds)
    print(f"eval | {len(seeds)} unseen seeds x 2 arms (clone vs incumbent) | horizon {a.horizon}",
          flush=True)
    ctx = get_context("spawn")
    jobs = []
    for s in seeds:
        jobs.append((s, a.horizon, a.params, a.model, "clone"))
        jobs.append((s, a.horizon, a.params, a.model, "base"))
    res = {"clone": {}, "base": {}}
    with ctx.Pool(a.workers) as pool:
        for arm, seed, ticks in pool.imap_unordered(eval_episode, jobs, chunksize=1):
            res[arm][seed] = ticks
    ks = sorted(set(res["clone"]) & set(res["base"]))
    c = [res["clone"][k] for k in ks]
    b = [res["base"][k] for k in ks]
    d = [x - y for x, y in zip(c, b)]
    w = sum(1 for v in d if v > 0)
    l = sum(1 for v in d if v < 0)
    print()
    print(f"CLONE vs INCUMBENT on {len(ks)} unseen paired seeds, horizon {a.horizon}")
    print(f"  clone     mean {np.mean(c):8.0f}  median {np.median(c):8.0f}  min {min(c):6d}  max {max(c):6d}")
    print(f"  incumbent mean {np.mean(b):8.0f}  median {np.median(b):8.0f}  min {min(b):6d}  max {max(b):6d}")
    print(f"  paired    {np.mean(d):+8.0f}  ({100*np.mean(d)/max(1,np.mean(b)):+.1f}%)  "
          f"W{w}/L{l}  win {100*w/max(1,w+l):.1f}%")
    json.dump({"clone": c, "base": b, "seeds": ks}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["collect", "train", "pop", "eval"])
    ap.add_argument("--nets", type=int, default=60)
    ap.add_argument("--seeds", default="7000-7005")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--data", default=os.path.join(HERE, "clone.npz"))
    ap.add_argument("--model", default=os.path.join(HERE, "clone.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "clone_eval.json"))
    a = ap.parse_args()
    {"collect": stage_collect, "train": train, "pop": stage_pop,
     "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()