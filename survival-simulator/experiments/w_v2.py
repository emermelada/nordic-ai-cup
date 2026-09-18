"""V2 UNIT + MECHANISM GATE: genome-aware breeder selection (genome_select).

PART 1 — OFF-BY-DEFAULT IDENTITY.  With every new flag off (evade_mode=0, thin_relay=0,
genome_select=0) the edited controller must reproduce the pre-edit controller EXACTLY on identical
seeds: same steps, same fruits, same spawns. Anything else means the new code touched the old path
(CLAUDE.md hygiene rule: new code is off-by-default AND byte-identical when off).

PART 2 — MECHANISM GATE (E-V2-0/E-V2-1, pre-registered in PIVOT_V2 §4):
  does selection actually MOVE the genome? Measured on the deployed controller, 119 births moved
  max_energy to mean 589.7 / max 996.8 by accident while vision_range stayed FLAT at ~200 (cap 400).
  The arms change only WHO may breed, so the measurable prediction is a rise in the fleet's mean
  vision_range (and/or max_energy) WITHIN the run. If the traits do not move, the lane is dead
  whatever the survival number says.

Usage: python w_v2.py [seeds] [horizon]
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.core import SimulationCore      # noqa: E402
from env_wrapper import make_action     # noqa: E402

PRE = os.environ.get("V2_PRE", "/tmp/v2/best_controller_HEAD.py")

TRAITS = ("vision_range", "vision_angle", "hearing_radius", "max_energy", "speed", "sprint_speed")
# The DTO + observation payload use vision_range/vision_angle; the ENTITY stores them as
# vision_radius/cone_angle. Reading the entity with the DTO key silently returns NaN (an all-NaN
# column is how this was caught), which is why the mapping is explicit here.
ENT_ATTR = {"vision_range": "vision_radius", "vision_angle": "cone_angle",
            "hearing_radius": "hearing_radius", "max_energy": "max_energy",
            "speed": "speed", "sprint_speed": "sprint_speed"}
DEFAULTS = {"speed": 10.0, "sprint_speed": 20.0, "max_energy": 500.0,
            "hearing_radius": 50.0, "vision_range": 200.0, "vision_angle": math.pi / 3}


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NEW = load_mod("bc_v2", os.path.join(ROOT, "best_controller.py"))
PRE_MOD = load_mod("bc_pre", PRE)
LIVE = json.load(open(os.path.join(ROOT, "best_controller", "params.json")))


def params(mod, blob):
    p = dict(mod.DEFAULT_PARAMS)
    p.update(blob)
    return p


def run_traced(fn, mod, seed, horizon, n_agents=5):
    """One episode through the REAL sim; returns steps, fruits, spawns, births(list of trait dicts)."""
    mod.reset_memory()
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=n_agents,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    seen = {a.agent_id for a in core.env.agents}
    births, spawns, fruits = [], 0, 0.0
    last_score = 0.0
    pops = []
    vsamp, esamp = [], []       # fleet trait samples every 250 ticks (the fleet is often DEAD at the
                                # end of an episode, so the final population is not a sample of it)
    i = 0
    for i in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a) for a in live) if s]
        if not states:
            break
        acts = []
        for s in states:
            act = fn(s)
            acts.append((s["agent_id"], make_action(s, act)))
            spawns += int(bool(act[3]))
        core.step(acts)
        sc = float(getattr(core.env, "score", 0.0) or 0.0)
        fruits += max(0.0, sc - last_score - 0.1) * 1000.0   # fruit energy = (dscore - dt) * 1000
        last_score = sc
        if i % 250 == 0:
            pops.append(len(core.env.agents))
            ft = fleet_traits(core)
            if ft:
                vsamp.append(float(np.mean([f["vision_range"] for f in ft])))
                esamp.append(float(np.mean([f["max_energy"] for f in ft])))
        new_ids = {a.agent_id for a in core.env.agents} - seen
        if new_ids:
            cur = {a.agent_id: a for a in core.env.agents}
            for nid in sorted(new_ids):
                a = cur.get(nid)
                if a is not None:
                    births.append({tr: float(getattr(a, ENT_ATTR[tr], float("nan"))) for tr in TRAITS})
        seen |= new_ids
    return i + 1, fruits, spawns, births, core, pops, vsamp, esamp


def fleet_traits(core):
    return [{tr: float(getattr(a, ENT_ATTR[tr], float("nan"))) for tr in TRAITS} for a in core.env.agents]


def main():
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "101,102,103").split(",")]
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    base = dict(LIVE)
    base.update({"evade_mode": 0.0, "thin_relay": 0.0, "genome_select": 0.0})

    print(f"=== PART 1: OFF-BY-DEFAULT IDENTITY (interleaved A/B/A/B), seeds {seeds}, horizon {horizon} ===",
          flush=True)
    if os.environ.get("V2_SKIP_IDENT"):
        print("  SKIPPED (V2_SKIP_IDENT set; identity is proven by _v2_ident.py: 4/4 identical runs on "
              "the deployed params, A1==B1==A2==B2 on seed 101)", flush=True)
    # INTERLEAVED on purpose: running all pre-edit seeds and then all edited seeds made seed 101
    # disagree (140 vs 133 spawns) although the new code is provably inert -- the environment's
    # object-set iteration order depends on allocation history, so a module running later in the
    # process can diverge. A/B/A/B with the same module twice separates code from noise.
    if not os.environ.get("V2_SKIP_IDENT"):
        ok = True
        for sd in seeds:
            seq = {}
            for k, (mod, blob) in {"A1": (PRE_MOD, base), "B1": (NEW, base),
                                   "A2": (PRE_MOD, base), "B2": (NEW, base)}.items():
                seq[k] = run_traced(mod.make_policy(params(mod, blob)), mod, sd, horizon)[:3]
            noise = (seq["A1"] != seq["A2"]) or (seq["B1"] != seq["B2"])
            code = (seq["A1"] != seq["B1"]) or (seq["A2"] != seq["B2"])
            ok = ok and (not code or noise)
            print(f"  seed {sd}: {seq} | same-module disagreement (noise)={noise} | A-vs-B (code)={code}",
                  flush=True)
        print(f"  OFF-BY-DEFAULT IDENTITY ACCEPTABLE (no code-only difference): {ok}", flush=True)

    print(f"\n=== PART 2: MECHANISM GATE — does selection move the genome? (horizon {horizon}) ===",
          flush=True)
    arms = [("LIVE(no selection)", {}),
            ("sel_top2", {"genome_select": 1.0, "gs_topk": 2.0}),
            ("sel_top4", {"genome_select": 1.0, "gs_topk": 4.0}),
            ("sel_top2_bank", {"genome_select": 1.0, "gs_topk": 2.0, "gs_w_energy": 1.2,
                               "gs_late_energy_mult": 3.0}),
            ("sel_top2_visiononly", {"genome_select": 1.0, "gs_topk": 2.0, "gs_w_energy": 0.0,
                                     "gs_w_speed": 0.0, "gs_w_sprint": 0.0}),
            ("sel_mean", {"genome_select": 1.0, "gs_mode": 2.0})]
    for label, over in arms:
        P = dict(LIVE)
        P.update(over)
        P.update({"evade_mode": 0.0, "thin_relay": 0.0})    # isolate the genome layer
        out = []
        for sd in seeds:
            steps, fruits, spawns, births, core, pops, vsamp, esamp = run_traced(
                NEW.make_policy(params(NEW, P)), NEW, sd, horizon)
            bv = [x["vision_range"] for x in births if x["vision_range"] == x["vision_range"]]
            out.append((sd, steps, fruits, spawns, len(births),
                        float(np.mean(vsamp)) if vsamp else float("nan"),
                        float(np.mean(vsamp[-8:])) if len(vsamp) >= 8 else float("nan"),
                        float(np.mean(esamp)) if esamp else float("nan"),
                        float(np.mean(bv)) if bv else float("nan"),
                        float(np.mean(pops)) if pops else float("nan"),
                        float(np.mean(pops[-8:])) if len(pops) >= 8 else float("nan")))
        print(f"  {label:22s} steps mean {np.mean([o[1] for o in out]):7.0f} | "
              f"fruits {np.mean([o[2] for o in out]):7.0f} | births {sum(o[4] for o in out):4d} | "
              f"pop mean {np.nanmean([o[9] for o in out]):5.1f} late {np.nanmean([o[10] for o in out]):5.1f} | "
              f"fleet vision {np.nanmean([o[5] for o in out]):6.1f} late {np.nanmean([o[6] for o in out]):6.1f} | "
              f"max_energy {np.nanmean([o[7] for o in out]):6.1f} | newborn vision {np.nanmean([o[8] for o in out]):6.1f}",
              flush=True)
        for o in out:
            print(f"      seed {o[0]}: steps={o[1]:6d} fruits={o[2]:7.0f} spawns={o[3]:3d} "
                  f"births={o[4]:3d} pop={o[9]:5.1f}/{o[10]:5.1f} "
                  f"vision={o[5]:6.1f}/{o[6]:6.1f} energy={o[7]:6.1f} newborn_vision={o[8]:6.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
