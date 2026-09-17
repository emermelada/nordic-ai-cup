import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import multi_eval, summarize, HORIZON
from best_controller import best_controller

# fresh-process smoke: importable, params merge OK, produces plausible numbers
r = multi_eval(best_controller, seeds=[101, 102], horizon=3000)
s = summarize(r, "best_controller(repro-check)")
print(f"OK: mean={s['mean']:.1f} fruit={s['fruits']:.0f} pred={s['predated']:.1f} "
      f"final={s['final_agents']:.1f} alive={s['alive']}/{s['n']} t={s['t']:.1f}s/eps")