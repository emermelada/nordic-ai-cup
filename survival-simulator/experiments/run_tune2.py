import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import multi_eval, summarize, print_summary
from best_controller import make_policy, DEFAULT_PARAMS

H = 4000
SEEDS = [100, 200, 300]

variants = {
    "combo1": dict(DEFAULT_PARAMS, fruit_weight=3.0, wander_weight=0.05,
                   repro_frac=0.50, repro_global_target=10, repro_popcap=4,
                   spawn_cooldown=200, repro_safe_radius=230, walk_frac=1.0, explore_frac=0.6,
                   flee_dist=140, escape_dist=250, danger_dist=200),
    "combo2": dict(DEFAULT_PARAMS, fruit_weight=3.0, wander_weight=0.05,
                   repro_frac=0.52, repro_global_target=9, repro_popcap=3,
                   spawn_cooldown=220, repro_safe_radius=250, walk_frac=1.0, explore_frac=0.6,
                   flee_dist=145, escape_dist=260, danger_dist=210, predator_weight=4.0),
}
for name, P in variants.items():
    r = multi_eval(make_policy(P), seeds=SEEDS, horizon=H)
    print_summary(summarize(r, name))