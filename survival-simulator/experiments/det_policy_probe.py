"""What changes between episodes? Instrument the policy's own state.

Established so far:
  * constant policy (no memory) -> IDENTICAL episodes (1246 ticks x3, two processes) => sim is clean
  * real policy -> same seed gives different ticks per episode WITHIN one process
    (6576 / 5743 / 4856 / 4674 / 6576) and across harnesses (4674 / 5363 / 5368)
  * the policy has only _MEM/_GC/_AGE/_EPOCH, all cleared by reset_memory(), and calls no RNG/clock

So this prints, before and after each episode: the size of each module-level dict, _EPOCH, whether
the params dict the policy closed over GAINED keys (a policy writing into its own params would make
episodes order-dependent -- exactly the config-position bias reset_memory was meant to fix), and
whether it mutated the observation list handed to it by the sim.
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


def mod_state():
    return ("_MEM=%d _GC=%d _AGE=%d _EPOCH=%d"
            % (len(bc._MEM), len(bc._GC), len(bc._AGE), bc._EPOCH))


def run_case(label, make_fresh_params, reset_fn, episode_index):
    P = dict(bc.DEFAULT_PARAMS)
    P.update(FILE_PARAMS)
    base_keys = set(P.keys())
    extra_before = set()
    observed_mutation = []

    fn = bc.make_policy(P)

    # wrap the policy to detect in-place mutation of the observation structure it receives
    def spy(state):
        obs = state.get("observations")
        before = None
        if isinstance(obs, list):
            before = [id(o) for o in obs]
        act = fn(state)
        if isinstance(obs, list):
            after = [id(o) for o in obs]
            if before is not None and len(after) != len(before):
                observed_mutation.append(("len", len(before), len(after)))
        return act

    r = run_eval_episode(spy, n_agents=5, seed=SEED, horizon=H, stop_on_death=True, reset_fn=reset_fn)
    print("  %-26s ep%d ticks=%6d | params gained keys: %-22s | obs mutated: %s"
          % (label, episode_index, r["steps"],
             sorted(set(P.keys()) - base_keys) or "none",
             observed_mutation[:2] or "no"))
    return r["steps"]


if __name__ == "__main__":
    print("=== real policy: three episodes in ONE process ===")
    for k in range(3):
        print("  mod-state before ep%d: %s" % (k + 1, mod_state()))
        run_case("real+reset", True, bc.reset_memory, k + 1)
        print("  mod-state after  ep%d: %s" % (k + 1, mod_state()))

    print()
    print("=== same, but WITHOUT passing reset_fn (does reset_memory actually matter?) ===")
    for k in range(2):
        run_case("real+NOreset", True, None, k + 1)
