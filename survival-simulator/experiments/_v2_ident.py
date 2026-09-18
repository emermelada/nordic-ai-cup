"""OFF-BY-DEFAULT IDENTITY, the decisive version: interleave A/B/A/B so run-to-run nondeterminism
cannot be mistaken for a code change.

Part 1 of w_v2.py reported a spawn-count difference on seed 101 between the pre-edit and edited
controller with every new flag OFF. That is either (a) my code leaking into the default path, or
(b) the environment's known run-to-run nondeterminism (the same policy+seed has been measured to
give 5,922 and 7,825 ticks in two runs; the serving containers' own B0 reference differed 8,496 vs
10,879 on the same seed). The test that separates them: run the SAME module twice in the same
process. If A != A, the delta is noise, not code.

Usage: python _v2_ident.py [seeds] [horizon]
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.core import SimulationCore      # noqa: E402
from env_wrapper import make_action     # noqa: E402

PRE = os.environ.get("V2_PRE", "/tmp/v2/h1e3_baseline.py")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(mod, P, seed, horizon, n_agents=5):
    mod.reset_memory()
    fn = mod.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=n_agents,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    spawns, births, last, eaten = 0, 0, 0.0, 0.0
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
            a = fn(s)
            acts.append((s["agent_id"], make_action(s, a)))
            spawns += int(bool(a[3]))
        core.step(acts)
        sc = float(getattr(core.env, "score", 0.0) or 0.0)
        eaten += max(0.0, sc - last - 0.1) * 1000.0     # fruit energy = (dscore - dt) * 1000
        last = sc
    return {"steps": i + 1, "spawns": spawns,
            "fruits": round(eaten, 1),
            "final_agents": len(core.env.agents),
            "fleet_min_energy": round(min([a.energy for a in core.env.agents], default=0.0), 1)}


def main():
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "101,104").split(",")]
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    NEW = load_mod("bc_new_id", os.path.join(ROOT, "best_controller.py"))
    PRE_MOD = load_mod("bc_pre_id", PRE)

    live = json.load(open(os.path.join(ROOT, "best_controller", "params.json")))
    live.update({"thin_relay": 0.0, "genome_select": 0.0})   # evade stays as deployed (1.0)
    p_new = dict(NEW.DEFAULT_PARAMS); p_new.update(live)
    p_pre = dict(PRE_MOD.DEFAULT_PARAMS); p_pre.update(live)

    print(f"PRE  = {PRE}  (the deployed artifact, extracted from the serving container)")
    print(f"NEW  = {os.path.join(ROOT, 'best_controller.py')} (edited: thin_relay + genome_select, both OFF)")
    print(f"interleaved A/B/A/B, horizon {horizon}, seeds {seeds}\n", flush=True)

    for sd in seeds:
        seq = {"A1": run(PRE_MOD, p_pre, sd, horizon), "B1": run(NEW, p_new, sd, horizon),
               "A2": run(PRE_MOD, p_pre, sd, horizon), "B2": run(NEW, p_new, sd, horizon)}
        for k, v in seq.items():
            print(f"  seed {sd} {k}: {v}", flush=True)
        noise = (seq["A1"] != seq["A2"]) or (seq["B1"] != seq["B2"])
        code = (seq["A1"] != seq["B1"]) or (seq["A2"] != seq["B2"])
        print(f"  -> same-module disagreement (NOISE FLOOR): {noise} | A-vs-B disagreement (CODE): {code}\n",
              flush=True)


if __name__ == "__main__":
    main()
