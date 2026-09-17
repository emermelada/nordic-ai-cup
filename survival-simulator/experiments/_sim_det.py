"""Isolate SIM determinism: constant policy (no controller, no randomness) -> same seed must
give an identical episode across processes."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from env_wrapper import run_eval_episode


def const_policy(state):
    return [5.0, 0.0, 0.0, 0.0]  # walk straight, never turn, never spawn


r = run_eval_episode(const_policy, n_agents=5, seed=100, horizon=3000, stop_on_death=True)
print("steps=%d score=%.4f fruit=%d pred=%d final=%d alive=%s"
      % (r["steps"], r["score"], r["fruits_eaten"], r["predated"],
         r["final_agents"], r["alive"]))