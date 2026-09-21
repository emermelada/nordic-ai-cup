import sys, os, math, collections
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1]); T1 = float(sys.argv[2]); T0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
if os.environ.get('NO_PREDATORS'): env.spawn_predator = lambda *a, **k: None
actions = []; prev_t = {}; dist_mode = collections.Counter(); switches = 0; reasons = collections.Counter(); ticks_mode = collections.Counter()
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > T1: break
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    fm = hv.maps.get("W")
    for act in (acts if env.time >= T0 else []):
        m = hv.mem[act["agent_id"]]
        dist_mode[m.mode] += act["move_distance"]; ticks_mode[m.mode] += 1
        if m.mode == "camp":
            old = prev_t.get(m.aid)
            if old is not None and m.target != old:
                switches += 1
                if fm is not None and fm.tx.size:
                    d0 = ((fm.tx - old[0])**2 + (fm.ty - old[1])**2) ** 0.5
                    k0 = int(d0.argmin())
                    if d0[k0] > 4: reasons["old tree gone"] += 1
                    elif (env.time - fm.tfruit[k0]) >= 30 and fm.twatch[k0] > hv.p["barren_watch"]: reasons["barren"] += 1
                    else: reasons["other (shared / re-choice)"] += 1
            prev_t[m.aid] = m.target
    actions = to_actions(acts)
agent_s = sum(ticks_mode.values()) / 10
print(f"t {env.time:.0f}: agent-seconds {agent_s:.0f}; camp switches {switches} ({switches / agent_s * 60:.2f} per agent-minute)")
print("reasons", dict(reasons))
print("distance by mode per agent-second:", {k: round(v / agent_s, 2) for k, v in dist_mode.items()})
print("time share by mode:", {k: round(v / sum(ticks_mode.values()), 3) for k, v in ticks_mode.items()})
