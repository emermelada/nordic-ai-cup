import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_wrapper import run_eval_episode
import best_controller as bc

P = bc.load_params() if hasattr(bc, "load_params") else bc.DEFAULT_PARAMS
fn = bc.make_policy(P) if hasattr(bc, "make_policy") else (lambda s: bc.potential_controller(s, P))

for seed in (100,):
    t0 = time.time()
    r = run_eval_episode(fn, n_agents=5, seed=seed, horizon=3000, reset_fn=bc.reset_memory)
    dt = time.time() - t0
    print(f"seed {seed}: {r['steps']} fleet-ticks in {dt:.2f}s  -> {r['steps']/dt:.0f} fleet-ticks/s, "
          f"~{r['steps']/dt*r['final_agents']:.0f} agent-steps/s (final_agents={r['final_agents']})")
