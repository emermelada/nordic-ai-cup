"""Is the episode non-determinism PYTHONHASHSEED-driven? Run the SHIPPED policy twice per mode."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
from env_wrapper import run_eval_episode

mode = "PYTHONHASHSEED=%s" % os.environ.get("PYTHONHASHSEED", "<unset>")
outs = []
for i in range(2):
    r = run_eval_episode(bc.make_policy(bc._load_params()), n_agents=5, seed=100, horizon=8000,
                         stop_on_death=True, reset_fn=bc.reset_memory)
    outs.append((r["steps"], round(r["score"], 6), r["spawns"], r["fruits_eaten"]))
    print("%s run%d ticks=%d score=%.2f spawns=%d fruits=%d"
          % (mode, i, r["steps"], r["score"], r["spawns"], r["fruits_eaten"]))
print("%s reproducible_in_process=%s" % (mode, outs[0] == outs[1]))
