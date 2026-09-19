"""Engagement evidence for the wB* arms (local, in parallel with the VPS sweep).

The VPS harness reports survival + fruits only, so it cannot distinguish "the mode never armed" from
"the mode armed and did nothing". This script measures, per arm per seed:
  armed_ticks  -- calls during which the state trigger was latched,
  fruit_ema    -- value of the visibility EMA at the end,
  n_trace      -- fleet size every 250 ticks (thin-relay should hold 1-3 after arming),
  steps        -- survival, capped at the horizon.
Usage: python _wb_smoke.py arm1.json,arm2.json seeds horizon
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from env_wrapper import run_eval_episode  # noqa: E402

spec = importlib.util.spec_from_file_location("bc_s", os.path.join(ROOT, "best_controller.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["bc_s"] = mod
spec.loader.exec_module(mod)

arms = sys.argv[1].split(",")
seeds = [int(x) for x in sys.argv[2].split(",")]
horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 18000

for arm in arms:
    blob = json.load(open(os.path.join(HERE, arm)))
    P = dict(mod.DEFAULT_PARAMS)
    P.update(blob)
    for sd in seeds:
        mod.reset_memory()
        state = {"armed": 0}
        orig = mod.potential_controller

        def spy(s, P=P, state=state):
            a = orig(s, P)
            if mod._THIN_ARMED:
                state["armed"] += 1
            return a

        r = run_eval_episode(spy, n_agents=5, seed=sd, horizon=horizon,
                             stop_on_death=True, reset_fn=mod.reset_memory, trace=True, trace_every=250)
        ns = [(t["t"], t["n"]) for t in r["traces"]]
        print(f"{arm:36s} seed={sd} steps={r['steps']:6d} fruits={r['fruits_eaten']:5d} "
              f"armed_ticks={state['armed']:6d} fruit_ema={mod._FRUIT_EMA:.3f} "
              f"final_agents={r['final_agents']} n_trace={ns}", flush=True)
