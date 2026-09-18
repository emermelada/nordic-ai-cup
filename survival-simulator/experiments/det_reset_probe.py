"""Definitive: is reset_memory() actually called, and is policy state EMPTY at first policy call?

Prints, per episode: how many times the reset hook fired, the module state immediately after the
reset call, and the module state at the FIRST policy call of the episode. If reset fires and the
state is empty at the first call, but two identical setups still diverge, the leak is not module
state at all and the remaining suspect is a stream shared with the simulator.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc

with open(os.path.join(REPO, "best_controller", "params.json")) as f:
    FILE_PARAMS = json.load(f)

SEED = 100
H = 16000


def state():
    return "_MEM=%d _GC=%d _AGE=%d _EPOCH=%d" % (len(bc._MEM), len(bc._GC), len(bc._AGE), bc._EPOCH)


def episode(idx, use_reset):
    reset_calls = []
    at_first_call = []

    def rf():
        reset_calls.append(1)
        bc.reset_memory()

    P = dict(bc.DEFAULT_PARAMS)
    P.update(FILE_PARAMS)
    fn = bc.make_policy(P)

    def spy(state_dict):
        if not at_first_call:
            at_first_call.append(state())
        return fn(state_dict)

    r = run_eval_episode(spy, n_agents=5, seed=SEED, horizon=H, stop_on_death=True,
                         reset_fn=rf if use_reset else None)
    print("  ep%-2d reset=%s  hook fired=%d  state right after reset=%s"
          % (idx, use_reset, len(reset_calls), at_first_call[0] if at_first_call else "n/a"))
    print("        ticks=%6d   state now=%s" % (r["steps"], state()))
    return r["steps"]


if __name__ == "__main__":
    print("=== WITHOUT reset (baseline reproducibility) ===")
    a = [episode(i + 1, False) for i in range(2)]
    print("  -> %s" % ("IDENTICAL" if len(set(a)) == 1 else "DIFFERENT: %s" % a))

    print()
    print("=== WITH reset hook ===")
    b = [episode(i + 1, True) for i in range(3)]
    print("  -> %s" % ("IDENTICAL" if len(set(b)) == 1 else "DIFFERENT: %s" % b))
