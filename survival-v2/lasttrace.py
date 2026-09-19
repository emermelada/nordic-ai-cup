"""Print the last seconds of agents that starved young late in a game (every 1 s).

    python lasttrace.py <seed> [n_agents] [late_from_t]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastsim import SimulationCore, to_actions  # noqa: E402
from hive import Hive  # noqa: E402

seed = int(sys.argv[1])
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 6
late = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
sim = SimulationCore(seed=seed)
env = sim.env
hv = Hive(seed=seed)
actions = []
hist = {}
born = {}
seen_eat = 0
eats = {}
for k in range(30000):
    st = sim.step(actions)
    t = env.time
    for (te, aid, e, age, over, fx, fy) in env.eat_log[seen_eat:]:
        eats.setdefault(aid, []).append((round(te, 1), round(e)))
    seen_eat = len(env.eat_log)
    if st["num_agents"] == 0 or t > 3000:
        break
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    actions = to_actions(acts)
    if k % 10 == 0:
        preds = [(p.x, p.y, p.resting) for p in env.predators]
        trees = [(tr.x, tr.y, tr.age) for tr in env.trees]
        for a in env.agents:
            born.setdefault(a.agent_id, t)
            m = hv.mem.get(a.agent_id)
            dp = min((math.hypot(p[0] - a.x, p[1] - a.y) for p in preds if not p[2]), default=9999)
            dt_ = min(((math.hypot(tr[0] - a.x, tr[1] - a.y), tr[2]) for tr in trees), default=(9999, 0))
            tgt = m.target if m is not None else None
            dtg = math.hypot(tgt[0] - a.x, tgt[1] - a.y) if tgt else -1
            ft = m.food_tgt if m is not None else None
            dft = math.hypot(ft[0] - a.x, ft[1] - a.y) if (ft and m.mode == "food") else -1
            hist.setdefault(a.agent_id, []).append(
                (round(t), round(a.x), round(a.y), round(a.energy, 1), m.mode if m else "?", round(dtg), round(dft),
                 round(dp), round(dt_[0]), round(dt_[1]), len(env.agents)))
deaths = [(td, aid, age, ma) for (td, aid, age, ma) in env.death_log if td > late and age <= ma]
print(f"seed {seed} end {env.time:.0f}; young starvations after t={late:.0f}: {len(deaths)}")
for td, aid, age, ma in deaths[:n_show]:
    print(f"\n== agent {aid} starved at t={td:.0f}, age {age:.0f} (max_age {ma:.0f}), born t={born.get(aid, -1):.0f}, "
          f"eats {eats.get(aid, [])}")
    print("   t     x    y     E  mode    d_camp d_fruit d_pred d_tree tree_age pop")
    for row in hist.get(aid, [])[-40:]:
        print("  ", " ".join(f"{v!s:>6}" for v in row))
