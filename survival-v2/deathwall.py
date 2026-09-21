"""Deaths by cause in the last 300 s of each game (kills / old age / other starvation) and colony state."""
import sys, os, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
def run(seed):
    from fastsim import SimulationCore, to_actions
    from hive import Hive
    sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed); actions = []
    ev = []  # (t, cause)
    last = dict(env.stat)
    for k in range(30000):
        st = sim.step(actions)
        s = env.stat
        for _ in range(s["kill_n"] - last["kill_n"]): ev.append((env.time, "kill"))
        for _ in range(s["old_death_n"] - last["old_death_n"]): ev.append((env.time, "old"))
        for _ in range((s["starve_n"] - last["starve_n"]) - (s["old_death_n"] - last["old_death_n"])): ev.append((env.time, "starve"))
        last = dict(s)
        if st["num_agents"] == 0 or env.time > 3000: break
        actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
    end = env.time
    tail = collections.Counter(c for t, c in ev if t > end - 300)
    births_tail = None
    return seed, round(end), dict(tail), len(env.predators), len(env.trees)
if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1].split(",")]
    with Pool(len(seeds)) as p:
        for r in p.map(run, seeds): print(r)
