"""Per-minute timeline of one game: population, births, deaths by cause, energy, modes."""
import sys, statistics
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
import json
seed = int(sys.argv[1]); params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(params=params, seed=seed)
if __import__('os').environ.get('NO_PREDATORS'): env.spawn_predator = lambda *a, **k: None
actions = []; last = dict(env.stat); last_births = 0; last_modes = {}
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > 3000: break
    actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
    if k % int(__import__("os").environ.get("TL_EVERY", "600")) == int(__import__("os").environ.get("TL_EVERY", "600")) - 1:
        s = env.stat; ags = env.agents
        births = env._next_agent_id - 5
        mt = hv.stats.get("mode_ticks", {}); dm = {m: mt.get(m, 0) - last_modes.get(m, 0) for m in mt}
        elite = sum(1 for m in hv.mem.values() if m.fit >= hv.best_fit - hv.p["elite_margin"])
        good = sum(1 for m in hv.mem.values() if m.fit >= hv.best_fit - hv.p["breed_gap"]); old = sum(1 for m in hv.mem.values() if m.old)
        print(f"t={env.time:5.0f} good {good:2d} old {old:2d} oldE {s['old_e']-last['old_e']:5.0f} pop {len(ags):3d} elite {elite:2d} births {births-last_births:3d} starve {s['starve_n']-last['starve_n']:3d} "
              f"(old {s['old_death_n']-last['old_death_n']:3d}) kills {s['kill_n']-last['kill_n']:3d} fruit {s['fruit_n']-last['fruit_n']:4d} "
              f"moveE {s['move_e']-last['move_e']:5.0f} liveE {s['live_e']-last['live_e']:5.0f} fruitE {s['fruit_e']-last['fruit_e']:5.0f} spawnE {s['spawn_e']-last['spawn_e']:5.0f} E {statistics.fmean(a.energy for a in ags):5.0f} spd {statistics.fmean(a.speed for a in ags):5.2f} preds {len(env.predators):2d} trees {len(env.trees):3d} fruits {len(env.fruits):3d} "
              f"modes {dict((m, round(v/ max(len(ags),1)/600*100)) for m, v in dm.items())}")
        last = dict(s); last_births = births; last_modes = dict(mt)
print("end t", round(env.time), "score", round(env.score, 1))
