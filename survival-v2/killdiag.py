"""What were victims doing when predators caught them?   python killdiag.py 101-110"""
import math
import os
import sys
from collections import defaultdict, deque
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(seed):
    from fastsim import SimulationCore, to_actions
    from hive import Hive
    sim = SimulationCore(seed=seed)
    env = sim.env
    hv = Hive(seed=seed)
    actions = []
    hist = defaultdict(lambda: deque(maxlen=60))
    seen = 0
    out = []
    for k in range(30000):
        st = sim.step(actions)
        for (tk, aid, pid) in env.kill_log[seen:]:
            h = list(hist.get(aid, []))
            if h:
                out.append((tk, h))
        seen = len(env.kill_log)
        if st["num_agents"] == 0 or env.time > 3000:
            break
        acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
        actions = to_actions(acts)
        byact = {a["agent_id"]: a for a in acts}
        for a in env.agents:
            m = hv.mem.get(a.agent_id)
            act = byact.get(a.agent_id)
            ix = min(max(int(a.x), 0), env.width - 1)
            iy = min(max(int(a.y), 0), env.height - 1)
            dp, pr = 9999.0, None
            for p in env.predators:
                d = math.hypot(p.x - a.x, p.y - a.y)
                if d < dp:
                    dp, pr = d, p
            facing = None
            if pr is not None:
                bearing = math.atan2(pr.y - a.y, pr.x - a.x)
                facing = abs((bearing - a.direction + math.pi) % (2 * math.pi) - math.pi)
            hist[a.agent_id].append((env.time, m.mode if m else "?", env.biome_map[ix, iy].type, a.energy,
                                     a.max_energy, a.speed, dp, facing, act["move_distance"] if act else 0.0,
                                     pr.resting if pr is not None else True))
    return out


def main():
    a, b = sys.argv[1].split("-")
    with Pool(10) as pool:
        res = pool.map(run, range(int(a), int(b) + 1), chunksize=1)
    kills = [k for r in res for k in r]
    n = len(kills)
    print(f"kills {n} over {len(res)} games")
    biome = defaultdict(int)
    lock = 0
    slow = 0
    flee_ticks = []
    first_d = []
    faced = 0
    moving = []
    mode_before = defaultdict(int)
    for tk, h in kills:
        last = h[-1]
        biome[last[2]] += 1
        if last[3] < last[4] / 5:
            lock += 1
        if last[5] < 15.0:
            slow += 1
        ft = 0
        for row in reversed(h):
            if row[1] == "flee":
                ft += 1
            else:
                break
        flee_ticks.append(ft)
        pre = h[-ft - 1] if ft < len(h) else h[0]
        mode_before[pre[1]] += 1
        first_d.append(h[-ft][6] if ft else last[6])
        if last[7] is not None and last[7] < math.pi / 2:
            faced += 1
        moving.append(sum(r[8] for r in h[-10:]) / min(10, len(h)))
    print("biome at death: " + "  ".join(f"{k} {v}" for k, v in sorted(biome.items(), key=lambda kv: -kv[1])))
    print(f"below sprint lock {lock} ({100 * lock / n:.0f}%)  slower than 15: {slow}")
    ft = sorted(flee_ticks)
    print(f"flee ticks before death: median {ft[n // 2]}  p25 {ft[n // 4]}  p75 {ft[3 * n // 4]}  zero {sum(1 for x in ft if x == 0)}")
    fd = sorted(first_d)
    print(f"predator distance when fleeing began: median {fd[n // 2]:.0f}  p25 {fd[n // 4]:.0f}  p75 {fd[3 * n // 4]:.0f}")
    print(f"facing the predator at death: {faced} ({100 * faced / n:.0f}%)  mean commanded move last 10 ticks {sum(moving) / n:.1f}")
    print("mode before fleeing: " + "  ".join(f"{k} {v}" for k, v in sorted(mode_before.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
