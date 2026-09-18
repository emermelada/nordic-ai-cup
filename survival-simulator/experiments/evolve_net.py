"""evolve_net.py - BET A: evolve a policy NETWORK from scratch, with a low-variance fitness proxy.

WHY EVOLUTION AND NOT PPO HERE
--------------------------------
The simulator itself is the fitness function, so no reward engineering is required - and reward
engineering is exactly what defeated the residual-RL attempt tonight (the learned correction converged
to 0.004 rad = nothing, because the score's constant +0.1/tick term leaves a very weak local gradient
spread over a 12,000-tick episode). Evolution scores whole episodes, so it never needs that gradient.

WHY A PROXY FITNESS (the part every "just use neuroevolution" plan gets wrong for THIS simulator)
------------------------------------------------------------------------------------------------
Measured in this project: a candidate that gained +25% on its own 3-seed fitness pool measured -2.8% and
won only 8/20 seeds on held-out data. Few-seed selection on a chaotic objective selects noise. And ticks
ARE chaotic here: the same policy on 20 seeds spans 640 to 12,785 ticks. So fitness cannot be ticks.
Income is stable (fruit eaten across 20 seeds: 1312/1481/1873/1594/1426 = +-12%), so fitness is
    income per agent-tick   (the smooth proxy, primary)
with ticks used only as a tie-break and for the final confirmation on 20 paired seeds.
That is what makes a small population affordable at ~756 fleet-ticks/s/core.

WHAT IS EVOLVED (from scratch - no heuristic anywhere in the action path)
------------------------------------------------------------------------
    34 inputs = the served 31-dim observation + the agent's own previous action (Markov completion; the
    served observation alone is not Markov, which is why behaviour cloning failed before)
    -> 1 hidden layer (tanh) -> (move_distance 0..20, move_direction +-pi, turn_angle +-pi, spawn logit)
Spawn IS evolved (unlike the residual track, where the heuristic kept it) because this is the from-scratch
bet, and reproduction is the relay that keeps a fleet alive under age mortality.

USAGE
  python evolve_net.py --seeds 1300,1301,1302 --pop 16 --gens 6 --horizon 8000 --minutes 60 --workers 3
"""
import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.core import SimulationCore                      # noqa: E402
from src.utils.DTOs import ActionRequest                 # noqa: E402
from env_wrapper import build_obs, OBS_DIM               # noqa: E402

NET_IN = OBS_DIM + 3        # served obs + previous action (dist/20, dir/pi, spawn)
NET_HID = 24                # deliberately tiny: a few hundred parameters, evolvable with a small pop
SPAWN_CD = 400.0
OUT_PATH = os.path.join(HERE, "evolve_net_best.json")


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- numpy net (workers use this)
def unpack(theta):
    """theta is a flat vector -> weight matrices. Layout: W1(HID,IN) b1(HID) W2(4,HID) b2(4)."""
    i = 0
    w1 = theta[i:i + NET_HID * NET_IN].reshape(NET_HID, NET_IN); i += NET_HID * NET_IN
    b1 = theta[i:i + NET_HID]; i += NET_HID
    w2 = theta[i:i + 4 * NET_HID].reshape(4, NET_HID); i += 4 * NET_HID
    b2 = theta[i:i + 4]; i += 4
    return w1, b1, w2, b2


def n_params():
    return NET_HID * NET_IN + NET_HID + 4 * NET_HID + 4


def act(theta, obs, rng, greedy=False):
    w1, b1, w2, b2 = unpack(theta)
    h = np.tanh(obs @ w1.T + b1)
    out = h @ w2.T + b2
    dist = 20.0 * (1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(out[0]))))))     # sigmoid -> [0,20]
    dirr = math.pi * math.tanh(float(out[1]))
    turn = math.pi * math.tanh(float(out[2]))
    pspawn = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(out[3])))))
    spawn = pspawn > 0.5 if greedy else (rng.random() < pspawn)
    return dist, dirr, turn, bool(spawn)


def random_theta(rng):
    return np.array([rng.gauss(0.0, 0.3) for _ in range(n_params())], dtype=np.float64)


# --------------------------------------------------------------------------- episode
def run_episode(theta, seed, horizon, rng, collect=False):
    """One fleet episode with the evolved net controlling every agent (from scratch)."""
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    prev = {}
    since = {}
    last_score = 0.0
    income = 0.0
    steps = 0
    agent_ticks = 0
    for t in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a) for a in live) if s]
        if not states:
            break
        acts = []
        for s in states:
            aid = s["agent_id"]
            pd, ps, pk = prev.get(aid, (0.0, 0.0, 0.0))
            o = np.concatenate([build_obs(s), np.array([pd / 20.0, ps / math.pi, pk], np.float32)])
            d, dr, tu, sp = act(theta, o, rng)
            prev[aid] = (d, dr, 1.0 if sp else 0.0)
            acts.append((aid, ActionRequest(agent_id=aid, move_distance=d, move_direction=dr,
                                            turn_angle=tu, spawn_agent=bool(sp))))
        agent_ticks += len(states)
        out = core.step(acts)
        dsc = out["score"] - last_score
        last_score = out["score"]
        if dsc > 0.1 + 1e-9:
            income += max(0.0, (dsc - 0.1)) * 1000.0
        steps = t + 1
    return {"steps": steps, "income": income, "agent_ticks": max(1, agent_ticks),
            "fruit_rate": income / max(1, agent_ticks), "score": core.env.score}


def fitness(theta, seeds, horizon, rng):
    """PROXY fitness: fruit energy eaten per AVAILABLE agent-slot over the whole horizon.

    NOT income/agent-tick. That version was gameable by dying: the first smoke test gave its best proxy
    (0.0615 fruit/agent-tick) to a policy that survived 127 ticks, because a rate over a tiny denominator
    is huge. Dividing by the HORIZON instead of by the realised ticks makes a dead fleet score ~0 and a
    long-lived productive fleet score highest, while still being far smoother than raw ticks (income is
    stable across seeds at +-12%, ticks span 640..12,785). Ticks are returned for reporting/tie-breaks only.
    """
    rs = [run_episode(theta, sd, horizon, rng) for sd in seeds]
    denom = max(1.0, float(horizon) * 5.0)          # horizon x starting agents = available agent-slots
    return (float(np.mean([r["income"] / denom for r in rs])),
            float(np.mean([r["steps"] for r in rs])),
            rs)


def worker_main(conn):
    rng = random.Random(12345)
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            conn.send(None); return
        _, thetas, seeds, horizon = msg
        out = []
        for th in thetas:
            f, tks, _ = fitness(np.asarray(th), seeds, horizon, rng)
            out.append((f, tks))
        conn.send(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="1300,1301,1302")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=8000)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--confirm-seeds", type=str, default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rng = random.Random(7)
    np.random.seed(7)
    print(f"EVOLVE-NET (from scratch) | params {n_params()} | pop {args.pop} elite {args.elite} | "
          f"fitness = fruit energy per available agent-slot (proxy) on seeds {seeds} | horizon {args.horizon}", flush=True)

    ctx = get_context("spawn")
    conns = []
    for _ in range(max(1, args.workers)):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker_main, args=(b,), daemon=True)
        p.start()
        conns.append((a, p))

    pop = [random_theta(rng) for _ in range(args.pop)]
    best = (-1e9, None, None, None)
    t0 = time.time()
    gen = 0
    while gen < args.gens and (time.time() - t0) / 60.0 < args.minutes:
        gen += 1
        tg = time.time()
        # split the population across the workers
        chunks = [pop[i::len(conns)] for i in range(len(conns))]
        for (c, _p), ch in zip(conns, chunks):
            c.send(("eval", [t.tolist() for t in ch], seeds, args.horizon))
        res = []
        for (c, _p) in conns:
            res.extend(c.recv() or [])
        scored = sorted(zip(pop, res), key=lambda pr: (pr[1][0], pr[1][1]), reverse=True)
        top = scored[0]
        if top[1][0] > best[0]:
            best = (top[1][0], top[0], top[1][1], gen)
            json.dump({"gen": gen, "fruit_rate": top[1][0], "ticks": top[1][1],
                       "theta": top[0].tolist(), "seeds": seeds, "horizon": args.horizon,
                       "note": "fitness = income per agent-tick (proxy), NOT ticks"}, open(OUT_PATH, "w"))
        print(f"gen {gen}: best proxy {top[1][0]:.5f} fruit/agent-tick (ticks {top[1][1]:.0f}) | "
              f"elite mean {np.mean([s[1][0] for s in scored[:args.elite]]):.5f} | "
              f"{(time.time() - tg) / 60.0:.1f} min", flush=True)
        # (mu, lambda) selection with mutation from the elites
        elites = [s[0] for s in scored[:args.elite]]
        pop = list(elites)
        while len(pop) < args.pop:
            parent = elites[rng.randrange(len(elites))]
            pop.append(parent + np.array([rng.gauss(0.0, args.sigma) for _ in range(n_params())]))

    print(f"done: best proxy {best[0]:.5f} at gen {best[3]} (proxy ticks {best[2]:.0f}) -> {OUT_PATH}", flush=True)

    if args.confirm_seeds:
        cseeds = [int(s) for s in args.confirm_seeds.split(",") if s.strip()]
        print(f"\nCONFIRMATION on {len(cseeds)} held-out seeds at horizon {args.horizon} "
              f"(ticks are chaotic; this is the number that decides)", flush=True)
        f, tks, rs = fitness(np.asarray(best[1]), cseeds, args.horizon, rng)
        per = [r["steps"] for r in rs]
        print(f"  evolved net : mean {np.mean(per):8.1f} ticks | median {np.median(per):8.1f} | "
              f"min {min(per):6d} | proxy fruit/agent-tick {f:.5f}", flush=True)
        print(f"  per-seed    : {per}", flush=True)

    for c, p in conns:
        try:
            c.send(("stop",)); c.recv()
        except Exception:
            pass
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    main()
