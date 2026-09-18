"""evolve_meta.py - BET B: evolve a FLEET-LEVEL META-CONTROLLER (the centralized population controller).

WHY THIS IS THE BET WITH REAL LEVERAGE
--------------------------------------
Everything we measured says the missing mechanism is a FLEET-LEVEL decision, not a per-agent one:
  * the score is TIME UNTIL THE LAST AGENT DIES - a property of the run, not of any individual;
  * agents die of age after 600-1,200 ticks, so survival is a RELAY: an unbroken chain of heirs;
  * the world's production halves every 3,000 ticks, so the endgame needs a THIN, banked fleet
    (a lone lineage costs 0.211 energy/tick = 7% of the 55-57k energy the world actually yields);
  * five separate attempts to fix this with STATIC parameters lost 9-36% with a monotone dose-response,
    because a fixed trigger either never fires or fires when the fleet cannot afford it.
A static parameter cannot express "thin the fleet once food visibility collapses AND the survivors are
banked". A policy over the GLOBAL state can. And centralisation costs nothing at serving time: the grader
already sends every agent's state in ONE request, so a fleet-level controller has zero latency penalty.
Prior art agrees: Lux AI v2 (the closest published analogue) was solved with centralized control.

WHAT IS EVOLVED (8 genes - tiny, so it converges)
------------------------------------------------
Even-indexed genes set WHEN a mode engages; odd ones set WHAT it does.
  g0 engage-time for endgame mode (ticks, 0 = never)          g1 penalty for blind ticks in the trigger
  g2 target population in endgame mode (1..6)                 g3 spawn energy gate (fraction of max_energy)
  g4 movement scale in endgame mode (0.2..1.0)                g5 mean-fleet-energy fraction below which we
                                                                  conserve (0 = never)
  g6 movement scale while conserving (0.2..1.0)               g7 evade energy threshold override (100..400)
The per-agent actions remain the live controller (H1 + the E3 retreat-while-facing). Only fleet-level mode
decisions are evolved, which is exactly the gap: 39 independent scalars cannot express role allocation.

FITNESS (two-tier, because a single tier cannot see this)
--------------------------------------------------------
Tier 1 (screening, cheap): mean ticking survival over --gen-seeds seeds at --horizon. Structural changes
here move survival by thousands of ticks, which is above the seed noise only when several seeds are
averaged - hence --gen-seeds 8 with the option of more on a many-core box.
Tier 2 (decision, expensive): the top --confirm-top genomes are re-run on a 20-seed held-out set at the
official 18,000 horizon, paired against the live controller, with mean/median/min/seeds-won reported.

USAGE (on the big x86 box)
  python evolve_meta.py --seeds 1400,1401,... --pop 32 --gen-seeds 8 --horizon 12000 \
      --workers $(nproc) --confirm 1300..1319 --confirm-horizon 18000
"""
import argparse
import json
import math
import os
import random
import sys
import statistics
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.core import SimulationCore                      # noqa: E402
from src.utils.DTOs import ActionRequest                 # noqa: E402
import best_controller as bc                             # noqa: E402

H1_PARAMS = os.path.join(ROOT, "best_controller", "params.json")
NGENES = 8
OUT = os.path.join(HERE, "evolve_meta_best.json")


def live_params():
    """The DEPLOYED params (E3 as of tonight: H1 + retreat-while-facing). Never DEFAULT_PARAMS."""
    p = dict(bc.DEFAULT_PARAMS)
    try:
        blob = json.load(open(H1_PARAMS))
        p.update(blob.get("params", blob) if isinstance(blob, dict) else {})
    except Exception as e:
        print("WARNING: could not load live params:", e, flush=True)
    return p


def rand_genome(rng):
    return np.array([rng.uniform(0.0, 20000.0),    # g0 engage time (ticks)
                     rng.uniform(0.0, 1.0),        # g1 blind bonus
                     rng.uniform(1.0, 6.0),        # g2 target pop
                     rng.uniform(0.15, 0.8),       # g3 spawn gate frac
                     rng.uniform(0.2, 1.0),        # g4 move frac
                     rng.uniform(0.0, 0.5),        # g5 conserve energy threshold (0 = never)
                     rng.uniform(0.2, 1.0),        # g6 conserve move frac
                     rng.uniform(100.0, 400.0)], dtype=np.float64)   # g7 evade energy threshold


def mutate(g, rng, sigma=0.08):
    scale = np.array([4000.0, 0.10, 0.5, 0.06, 0.08, 0.05, 0.08, 30.0])
    return np.clip(g + np.array([rng.gauss(0, 1) for _ in range(NGENES)]) * scale * (1 + sigma), 
                   np.array([0, 0, 1, 0.15, 0.2, 0, 0.2, 100]),
                   np.array([20000, 1, 6, 0.8, 1.0, 0.5, 1.0, 400]))


def run_episode(g, seed, horizon, rng, return_actions=False):
    """One fleet episode where the LIVE controller acts per agent, but the GENOME sets the fleet modes."""
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)
    bc.reset_memory()
    P = live_params()
    fn = bc.make_policy(P)                 # the closure reads P per call, so mutating P in place re-tunes it
    base_P = dict(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    last_score = 0.0
    income = 0.0
    steps = 0
    agent_ticks = 0
    armed_ticks = 0
    for t in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a) for a in live) if s]
        if not states:
            break
        # ---- global features (the grader hands us every agent's state in one request) ----
        n = len(states)
        want_fruit = sum(1 for s in states
                         if not any(o.get("type") == "Fruit" for o in (s.get("observations") or [])))
        blind_frac = want_fruit / max(1, n)
        mean_ef = float(np.mean([s["energy"] / max(1.0, s["max_energy"]) for s in states]))
        nears = [o["distance"] for s in states for o in (s.get("observations") or [])
                 if o.get("type") == "Predator"]
        min_pred = min(nears) if nears else 9e9
        # ---- mode decision (the evolved policy) ----
        endgame = (g[0] > 0 and t >= g[0] and blind_frac >= (0.5 - 0.5 * g[1]))
        conserve = (g[5] > 0 and mean_ef < g[5])
        P.clear(); P.update(base_P)
        if endgame:
            armed_ticks += 1
            P["repro_global_target"] = float(g[2])
            P["repro_frac_min"] = float(g[3])
            P["repro_frac"] = max(float(g[3]), min(0.9, float(g[3]) + 0.25))
            P["spawn_cooldown"] = 300
            P["walk_frac"] = float(min(base_P.get("walk_frac", 0.49), g[4]))
            P["evade_speed_frac"] = 1.0
            P["repro_safe_radius"] = 150.0
        if conserve:
            P["walk_frac"] = min(P.get("walk_frac", 1.0), float(g[6]))
        P["evade_energy_abs"] = float(g[7])
        acts = []
        for s in states:
            a = fn(s)
            acts.append((s["agent_id"], ActionRequest(agent_id=s["agent_id"], move_distance=float(a[0]),
                                                      move_direction=float(a[1]), turn_angle=float(a[2]),
                                                      spawn_agent=bool(a[3]))))
        agent_ticks += n
        out = core.step(acts)
        dsc = out["score"] - last_score
        last_score = out["score"]
        if dsc > 0.1 + 1e-9:
            income += max(0.0, (dsc - 0.1)) * 1000.0
        steps = t + 1
    P.clear() if return_actions else None
    return {"steps": steps, "income": income, "agent_ticks": max(1, agent_ticks),
            "armed_ticks": armed_ticks, "score": core.env.score}


def evaluate(g, seeds, horizon, rng):
    rs = [run_episode(g, sd, horizon, rng) for sd in seeds]
    return (float(np.mean([r["steps"] for r in rs])),
            float(np.mean([r["income"] / (horizon * 5.0) for r in rs])),
            float(np.mean([r["armed_ticks"] for r in rs])),
            rs)


def worker_main(conn):
    rng = random.Random(999)
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            conn.send(None); return
        _, genomes, seeds, horizon = msg
        out = []
        for gg in genomes:
            m, inc, armed, _ = evaluate(np.asarray(gg), seeds, horizon, rng)
            out.append((m, inc, armed))
        conn.send(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, required=True, help="screening seeds (fewer, more seeds = better)")
    ap.add_argument("--confirm", type=str, default="", help="held-out seeds for the tier-2 decision")
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=12000)
    ap.add_argument("--confirm-horizon", type=int, default=18000)
    ap.add_argument("--confirm-top", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--minutes", type=float, default=600.0)
    args = ap.parse_args()

    scr = [int(s) for s in args.seeds.split(",") if s.strip()]
    rng = random.Random(11)
    print(f"EVOLVE-META | genes {NGENES} | pop {args.pop} elite {args.elite} | screening: {len(scr)} seeds "
          f"horizon {args.horizon} | workers {args.workers} | live params from {H1_PARAMS}", flush=True)
    lp = live_params()
    print(f"  live controller identity: fruit_weight={lp.get('fruit_weight'):.3f} "
          f"evade_mode={lp.get('evade_mode')} walk_frac={lp.get('walk_frac'):.3f}", flush=True)

    ctx = get_context("spawn")
    conns = []
    for _ in range(args.workers):
        a, b = ctx.Pipe()
        pr = ctx.Process(target=worker_main, args=(b,), daemon=True)
        pr.start()
        conns.append((a, pr))

    pop = [rand_genome(rng) for _ in range(args.pop)]
    best = (-1e9, None, None)
    scored = []
    t0 = time.time()
    gen = 0
    while gen < args.gens and (time.time() - t0) / 60.0 < args.minutes:
        gen += 1
        tg = time.time()
        chunks = [pop[i::len(conns)] for i in range(len(conns))]
        for (c, _p), ch in zip(conns, chunks):
            c.send(("eval", [x.tolist() for x in ch], scr, args.horizon))
        res = []
        for (c, _p) in conns:
            res.extend(c.recv() or [])
        scored = sorted(zip(pop, res), key=lambda pr: (pr[1][0], pr[1][1]), reverse=True)
        topg, (tm, tinc, tarmed) = scored[0]
        if tm > best[0]:
            best = (tm, topg.copy(), tinc)
            json.dump({"gen": gen, "mean_ticks": tm, "income_proxy": tinc, "genome": topg.tolist(),
                       "genes_note": "g0 engage_tick g1 blind_bonus g2 target_pop g3 spawn_gate "
                                     "g4 move_frac g5 conserve_thresh g6 conserve_move g7 evade_energy",
                       "screening_seeds": scr, "horizon": args.horizon}, open(OUT, "w"))
        print(f"gen {gen:2d}: best mean {tm:7.1f} ticks (income {tinc:.5f}, endgame armed {tarmed:.0f} ticks) | "
              f"elite mean {np.mean([s[1][0] for s in scored[:args.elite]]):7.1f} | "
              f"{(time.time() - tg) / 60.0:.1f} min | genome {np.round(topg, 3).tolist()}", flush=True)
        elites = [s[0] for s in scored[:args.elite]]
        pop = list(elites)
        while len(pop) < args.pop:
            pop.append(mutate(elites[rng.randrange(len(elites))], rng))

    print(f"\ndone: best screening {best[0]:.1f} ticks at gen {gen} -> {OUT}", flush=True)
    print(f"  genome: {np.round(best[1], 4).tolist()}", flush=True)

    if args.confirm:
        cseeds = [int(s) for s in args.confirm.split(",") if s.strip()]
        print(f"\nTIER 2: {args.confirm_top} best genomes + the live controller, paired on {len(cseeds)} "
              f"held-out seeds at horizon {args.confirm_horizon}", flush=True)
        base_m, base_inc, base_armed, base_rs = evaluate(np.array([0.0, 0.0, 5.0, 0.24, 1.0, 0.0, 1.0, 200.0]),
                                                         cseeds, args.confirm_horizon, rng)
        bp = [r["steps"] for r in base_rs]
        print(f"  LIVE (no meta override, same framework): mean {base_m:8.1f} median "
              f"{statistics.median(bp):8.1f} min {min(bp):6d}", flush=True)
        print(f"    per-seed {bp}", flush=True)
        cands = [best[1]]
        for s in scored[:args.confirm_top]:
            if not any(np.allclose(s[0], c) for c in cands):
                cands.append(s[0])
        for i, gg in enumerate(cands):
            m, inc, armed, rs = evaluate(gg, cseeds, args.confirm_horizon, rng)
            per = [r["steps"] for r in rs]
            won = sum(1 for a, b in zip(per, bp) if a > b)
            print(f"  meta#{i}: mean {m:8.1f} median {statistics.median(per):8.1f} min {min(per):6d} | "
                  f"delta {m - base_m:+8.1f} | seeds won {won}/{len(per)} | armed {armed:.0f}", flush=True)
            print(f"    genome {np.round(gg, 4).tolist()}", flush=True)
            print(f"    per-seed {per}", flush=True)

    for c, pr in conns:
        try:
            c.send(("stop",)); c.recv()
        except Exception:
            pass
        pr.join(timeout=5)
        if pr.is_alive():
            pr.terminate()


if __name__ == "__main__":
    main()
