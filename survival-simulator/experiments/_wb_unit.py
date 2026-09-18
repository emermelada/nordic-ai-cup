"""Local unit test for the thin-relay addition. Two independent checks:

 1. OFF-BY-DEFAULT IDENTITY: with H1 params (thin_relay absent => 0.0) the edited controller must
    reproduce the pristine pre-edit controller EXACTLY (same steps, same fruits, same spawns) on
    identical seeds. Anything else means the new code touched the old path.
 2. MECHANISM SMOKE: with thin_relay=1 the arm must actually ARM (time+visibility trigger) and the
    relay must produce heirs at the end; also report gpop behaviour so a "trigger never fires"
    result is visible as such rather than being mistaken for a null effect.
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from env_wrapper import run_eval_episode  # noqa: E402


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NEW = load_mod("bc_new", os.path.join(ROOT, "best_controller.py"))
ORIG = load_mod("bc_orig", "/tmp/wB/best_controller_ORIG.py")

H1 = json.load(open(os.path.join(HERE, "wH1_winEARN_liveSPEND.json")))


def params(mod, blob):
    p = dict(mod.DEFAULT_PARAMS)
    p.update(blob)
    return p


def run(mod, P, seeds, horizon=6000, trace_arm=False):
    fn = mod.make_policy(P)
    out = []
    for sd in seeds:
        if trace_arm:
            mod._THIN_ARMED = False
            mod._FRUIT_EMA = 0.0
            armed_ticks = [0]
            orig_fn = mod.potential_controller

            def spy(state, P=P, fn=fn):
                a = orig_fn(state, P)
                if mod._THIN_ARMED:
                    armed_ticks[0] += 1
                return a
            r = run_eval_episode(spy, n_agents=5, seed=sd, horizon=horizon,
                                 stop_on_death=True, reset_fn=mod.reset_memory)
            out.append((r["steps"], r["fruits_eaten"], r["spawns"], armed_ticks[0],
                        mod._FRUIT_EMA, r["final_agents"]))
        else:
            r = run_eval_episode(fn, n_agents=5, seed=sd, horizon=horizon,
                                 stop_on_death=True, reset_fn=mod.reset_memory)
            out.append((r["steps"], r["fruits_eaten"], r["spawns"]))
    return out


SEEDS = [101, 102, 103]
print("=== 1. identity with thin_relay OFF (H1 params) ===", flush=True)
a = run(ORIG, params(ORIG, H1), SEEDS)
b = run(NEW, params(NEW, H1), SEEDS)
print(f"  pristine : {a}")
print(f"  edited   : {b}")
print(f"  IDENTICAL: {a == b}", flush=True)

print("\n=== 2. mechanism smoke (wB1 arm, horizon 8000) ===", flush=True)
for arm in ["wB1_thin_c6k_v35_t2_m70.json", "wB2_thin_c4500_v45_t3_m75.json",
            "wB3_thin_c3500_v55_t1_m85.json"]:
    blob = json.load(open(os.path.join(HERE, arm)))
    r = run(NEW, params(NEW, blob), [101, 102, 103], horizon=8000, trace_arm=True)
    print(f"  {arm}\n     (steps, fruits, spawns, armed_ticks, fruit_ema_end, final_agents) = {r}", flush=True)
