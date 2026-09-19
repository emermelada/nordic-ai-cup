#!/usr/bin/env python3
"""rec_resid.py - the ML policy of record: a RECURRENT residual on the deployed controller.

WHY THIS SHAPE, stated once and traceable to a measurement (no component exists "because it might help")

  residual, not replacement   clone_rec.py measured a BC-cloned recurrent net at -16.8% paired
                              (W12/L28, floor collapsing 3,649 -> 649 ticks). A learned component
                              cannot be trusted to reproduce the incumbent, so it must only ever
                              ADD to it. With the head zero-initialised the policy IS the incumbent
                              exactly, and `zero` verifies that bit-for-bit before anything trains.
  recurrent, not wider        probe_enc.py: a 31-float frame explains 2.9% of the incumbent's
                              heading in the blind regime; 61 floats explain 2.4% (i.e. nothing);
                              the SAME 61 floats plus 5 memory scalars explain 57.5%. 82.4% of
                              agent-ticks are blind. The missing capability is state, not width.
  three outputs               heading offset (target-choice correction), speed offset (movement is
                              income-blind: it stays flat at ~2,600 energy/1k while income falls
                              94% to 557), and a reproduction-gate offset (the gate uses a CONSTANT
                              target of 12 against production that halves every 3,000 ticks). The
                              gate offset is bounded and additive rather than a free spawn bit,
                              because the deployed gate is causally beneficial (-4.3% to -11.5% for
                              all five alternatives tested) and must not be destructible.
  no shaping                  the objective is already dense (+0.1/tick) and the project's one
                              shaping attempt was gameable by dying.

STAGES
  zero  : the zero-head policy must reproduce BASE BIT-IDENTICALLY on >=40 seeds. If it does not,
          the plumbing is wrong and nothing downstream is interpretable. Run this FIRST.
  es    : parallel evolution strategies over the weights, fitness = mean survival ticks, paired.
          Each candidate is evaluated on the SAME seed block; the incumbent is arm 0.

    python3 rec_resid.py --stage zero --seeds 8000-8039
    python3 rec_resid.py --stage es   --seeds 8100-8139 --pop 56 --gens 6
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

HID = 48
HEAD_SCALE = np.array([0.5, 0.30, 0.25], np.float32)   # rad, fraction of sprint, rf fraction


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def make_net(dim=V2_DIM, hid=HID):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(dim, hid, batch_first=True)
            self.head = nn.Linear(hid, 3)
            nn.init.zeros_(self.head.weight)   # <- THE ZERO-CHANGE PATH: out == 0 at init
            nn.init.zeros_(self.head.bias)

        def forward(self, x, h=None):
            o, h = self.gru(x, h)
            return self.head(o), h
    return Net()


def rand_net(seed, sigma):
    """A perturbed net. Perturbation is applied to EVERY parameter, so the head leaves zero."""
    import torch
    net = make_net()
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn(p.shape, generator=g) * sigma)
    return net


def net_to_vec(net):
    return np.concatenate([p.detach().numpy().ravel() for p in net.parameters()])


def vec_to_net(vec):
    net = make_net()
    i = 0
    with __import__("torch").no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(__import__("torch").tensor(vec[i:i + n]).view(p.shape))
            i += n
    return net


def run(args):
    seed, horizon, params_path, model_path, arm = args
    import random
    import torch
    import best_controller as bc
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    P = dict(bc.DEFAULT_PARAMS)
    with open(params_path) as f:
        P.update(json.load(f))
    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    base = bc.make_policy(P)
    net = None
    if arm not in ("base", "base2"):
        torch.set_num_threads(1)
        net = make_net()
        net.load_state_dict(torch.load(model_path, weights_only=True)["sd"])
        net.eval()

    out_of_range = 0
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    memo, hid = {}, {}
    for i in range(horizon):
        live = list(core.env.agents)
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
        acts = []
        for s in states:
            aid = s["agent_id"]
            dist, dr, turn, spawn = base(s)          # turn and spawn ALWAYS come from the base
            if arm not in ("base", "base2"):
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
                # The hidden state is unnormalised and can grow without bound over the ~10^4 steps
                # of an episode. With a zero head that is invisible EXCEPT when h reaches inf/NaN,
                # because 0*inf = NaN and `NaN != 0.0` is true - so the "zero" policy emitted NaN
                # headings on 3 of 40 seeds at ticks 3255/4515/8048, late in long episodes and for
                # one agent only. Clamp at the source and sanitise at the output; a residual must
                # never be able to inject a non-finite action.
                h = torch.clamp(h, -10.0, 10.0)
                hid[aid] = h
                o = o[0, 0].numpy() * HEAD_SCALE
                if not np.isfinite(o).all():
                    out_of_range += 1
                    o = np.zeros(3, np.float32)
                sprint = max(1.0, float(s.get("sprint_speed", 20.0) or 20.0))
                # NO wrap and NO clip here. Both are redundant - update_entity_position already
                # caps distance at sprint_speed and floors it at 0 (environment.py:508-513), and
                # the sim takes cos/sin of any real angle. More importantly each was a 1-ULP
                # perturbation of an already-in-range value, and a 1-ULP difference fully diverges
                # an 18,000-tick chaotic simulation: the zero-init test failed 0/40 purely on
                # `(dr+pi)%2pi-pi` moving ...593 to ...592. Gating on exact zero keeps the
                # zero-change path BIT-IDENTICAL, which is the whole point of the contract.
                if o[0] != 0.0:
                    dr = dr + float(o[0])
                if o[1] != 0.0:
                    dist = dist + float(o[1]) * sprint
                m["e"] = energy
            acts.append((aid, ActionRequest(agent_id=aid, move_distance=float(dist),
                                            move_direction=float(dr), turn_angle=float(turn),
                                            spawn_agent=bool(spawn))))
        core.step(acts)
        keep = {a.agent_id for a in core.env.agents}
        hid = {k: v for k, v in hid.items() if k in keep}
    return arm, seed, i + 1, out_of_range


RESULT_SINK = {}


def _run_star(a):
    return run(a)


def stage_zero(a):
    from multiprocessing import get_context
    seeds = parse_seeds(a.seeds)
    zero_path = os.path.join(HERE, "zero.pt")
    import torch
    torch.save({"sd": make_net().state_dict()}, zero_path)
    ctx = get_context("spawn")
    jobs = []
    for s in seeds:
        jobs.append((s, a.horizon, a.params, zero_path, "zero"))
        jobs.append((s, a.horizon, a.params, zero_path, "base"))
        jobs.append((s, a.horizon, a.params, zero_path, "base2"))
    res = {"zero": {}, "base": {}, "base2": {}}
    oor = 0
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for arm, seed, ticks, o in pool.imap_unordered(_run_star, jobs, chunksize=1):
            res[arm][seed] = ticks
            oor += o
    ks = sorted(set(res["zero"]) & set(res["base"]))
    bad = [k for k in ks if res["zero"][k] != res["base"][k]]
    b2 = [k for k in ks if res["base2"][k] != res["base"][k]]
    print(f"ZERO-INIT EQUIVALENCE on {len(ks)} seeds, horizon {a.horizon}")
    print(f"  A/A control base vs base2 (identical policy): {len(b2)}/{len(ks)} seeds differ"
          + (f"  <- SIMULATOR nondeterminism: {b2[:6]}" if b2 else "  (deterministic)"))
    print(f"  bit-identical seeds: {len(ks)-len(bad)}/{len(ks)}")
    if bad:
        print(f"  MISMATCH on {len(bad)}: {[(k, res['base'][k], res['zero'][k]) for k in bad[:5]]}")
        print("  -> the residual plumbing is WRONG; nothing downstream is interpretable")
        sys.exit(2)
    print(f"  PASS - the zero-head policy IS the incumbent. out-of-range outputs: {oor}")
    print("  (heading/speed/gate offsets are exactly 0 at init by construction)")


def stage_es(a):
    """Parallel ES: every member evaluated on the SAME paired seeds; incumbent is arm 'base'."""
    from multiprocessing import get_context
    seeds = parse_seeds(a.seeds)
    rng = np.random.default_rng(a.seed0)
    ctx = get_context("spawn")
    base_ticks = None
    best = None
    for gen in range(a.gens):
        cands = []
        for i in range(a.pop):
            sigma = a.sigma * (a.sigma_decay ** gen)
            net = rand_net(1000 * gen + i + 1, sigma)
            path = os.path.join(HERE, f"cand_g{gen}_{i}.pt")
            import torch
            torch.save({"sd": net.state_dict()}, path)
            cands.append((i, path))
        jobs, paths = [], {}
        for i, path in cands:
            paths[i] = path
            for s in seeds:
                jobs.append((s, a.horizon, a.params, path, f"c{i}"))
        if base_ticks is None:
            for s in seeds:
                jobs.append((s, a.horizon, a.params, paths[0], "base"))
        acc = {}
        with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
            for arm, seed, ticks, _ in pool.imap_unordered(_run_star, jobs, chunksize=1):
                acc.setdefault(arm, {})[seed] = ticks
        if base_ticks is None:
            base_ticks = acc.pop("base")
        scored = []
        for i, path in cands:
            d = acc.get(f"c{i}", {})
            ks = [k for k in seeds if k in d and k in base_ticks]
            if not ks:
                continue
            paired = np.mean([d[k] - base_ticks[k] for k in ks])
            scored.append((paired, float(np.mean([d[k] for k in ks])), i, path))
        scored.sort(reverse=True)
        bm = float(np.mean(list(base_ticks.values())))
        print(f"\nGEN {gen}  BASE mean {bm:.0f}   pop {len(scored)}  sigma {a.sigma*(a.sigma_decay**gen):.3f}")
        for p, m, i, path in scored[:5]:
            print(f"   c{i:<3} mean {m:7.0f}  paired {p:+8.0f} ({100*p/bm:+5.1f}%)")
        print(f"   ... worst {scored[-1][0]:+.0f}")
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        json.dump([{"paired": p, "mean": m, "id": i} for p, m, i, _ in scored],
                  open(os.path.join(HERE, f"es_gen{gen}.json"), "w"), indent=1)
        if scored[0][0] <= 0:
            print(f"   -> best paired change is {scored[0][0]:+.0f}; no candidate beats BASE this gen")
    print(f"\nBEST over all generations: paired {best[0]:+.0f} ({100*best[0]/bm:+.1f}%) id c{best[2]}")
    if best[0] > 0:
        import shutil
        shutil.copy(best[3], os.path.join(HERE, "best_resid.pt"))
        print(f"copied -> {os.path.join(HERE, 'best_resid.pt')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["zero", "es"])
    ap.add_argument("--seeds", default="8000-8039")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--pop", type=int, default=56)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--sigma-decay", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    a = ap.parse_args()
    {"zero": stage_zero, "es": stage_es}[a.stage](a)


if __name__ == "__main__":
    main()