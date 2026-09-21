"""Check fastsim against the official simulator.

    python verify_fastsim.py <seed> [ticks]

Runs both simulators in separate processes on the same seed with a deterministic test policy, then reports:
the world after generation (RNG state, obstacles, trees, fruit, agents) and the first tick at which the
per-tick digests (score, counts, agent/predator positions and energies, RNG state) differ, plus timings.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def test_policy(state_obs):
    """Order-invariant, deterministic policy: flee close predators, walk to the nearest fruit, breed at 160."""
    actions = []
    for a in state_obs:
        obs = a["observations"]
        fruits = sorted((o["distance"], o["angle"]) for o in obs if o["type"] == "Fruit")
        preds = sorted((o["distance"], o["angle"]) for o in obs if o["type"] == "Predator")
        if preds and preds[0][0] < 80:
            dist, direction = a["sprint_speed"], preds[0][1] + math.pi
        elif fruits:
            dist, direction = min(a["speed"], fruits[0][0]), fruits[0][1]
        else:
            dist, direction = a["speed"] * 0.5, 0.3
        actions.append({"agent_id": a["agent_id"], "move_distance": dist, "move_direction": direction,
                        "turn_angle": max(-0.5, min(0.5, 0.2 * direction)), "spawn_agent": a["energy"] > 160})
    return actions


def world_digest(env):
    r = env.rng.getstate()
    return {
        "rng": hashlib.sha1(repr(r).encode()).hexdigest(),
        "obstacles": [(round(o.x, 9), round(o.y, 9), round(o.width, 9), round(o.height, 9)) for o in env.obstacles],
        "trees": [(round(t.x, 9), round(t.y, 9), round(t.age, 9)) for t in env.trees],
        "fruits": [(round(f.x, 9), round(f.y, 9)) for f in env.fruits],
        "agents": [(round(a.x, 9), round(a.y, 9), round(a.direction, 9), round(a.max_age, 9)) for a in env.agents],
    }


def tick_digest(env, k):
    ag = sorted((a.agent_id, round(float(a.x), 6), round(float(a.y), 6), round(float(a.energy), 6)) for a in env.agents)
    pr = sorted((round(float(p.x), 6), round(float(p.y), 6), round(float(p.energy), 6)) for p in env.predators)
    return [k, round(float(env.score), 6), len(env.agents), len(env.predators), len(env.fruits), len(env.trees),
            hashlib.sha1(repr((ag, pr)).encode()).hexdigest()[:12],
            hashlib.sha1(repr(env.rng.getstate()).encode()).hexdigest()[:12]]


def run(mode, seed, ticks):
    if mode == "fast":
        sys.path.insert(0, HERE)
        from fastsim import SimulationCore, to_actions
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        sys.path.insert(0, os.path.join(HERE, "official"))
        from src.core import SimulationCore

        def to_actions(dicts):
            from src.utils.DTOs import ActionRequest
            return [(d["agent_id"], ActionRequest(**d)) for d in dicts]
    t0 = time.time()
    sim = SimulationCore(seed=seed)
    t_init = time.time() - t0
    out = {"world": world_digest(sim.env), "ticks": [], "init_s": t_init}
    actions = []
    t1 = time.time()
    for k in range(ticks):
        state = sim.step(actions)
        out["ticks"].append(tick_digest(sim.env, k))
        if state["num_agents"] == 0 or sim.env.time > 3000:
            break
        actions = to_actions(test_policy(state["observations"]))
    out["run_s"] = time.time() - t1
    out["n"] = len(out["ticks"])
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("fast", "official"):
        res = run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
        print(json.dumps(res))
        return
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    res = {}
    for mode in ("official", "fast"):
        p = subprocess.run([sys.executable, __file__, mode, str(seed), str(ticks)], capture_output=True, text=True)
        if p.returncode != 0:
            print(mode, "failed:\n", p.stderr[-3000:])
            return
        res[mode] = json.loads(p.stdout.strip().splitlines()[-1])
    o, f = res["official"], res["fast"]
    for key in ("rng", "obstacles", "trees", "fruits", "agents"):
        print(f"world.{key}: {'same' if o['world'][key] == f['world'][key] else 'DIFFERENT'}")
    first = None
    for a, b in zip(o["ticks"], f["ticks"]):
        if a != b:
            first = (a, b)
            break
    print(f"ticks compared: {min(o['n'], f['n'])}  (official ran {o['n']}, fast ran {f['n']})")
    print("first divergence:", "none" if first is None else first)
    print(f"final official: {o['ticks'][-1][:6]}")
    print(f"final fast:     {f['ticks'][-1][:6]}")
    print(f"init s  official {o['init_s']:.2f}  fast {f['init_s']:.2f}")
    print(f"run s   official {o['run_s']:.2f} ({1000 * o['run_s'] / o['n']:.2f} ms/tick)  "
          f"fast {f['run_s']:.2f} ({1000 * f['run_s'] / f['n']:.2f} ms/tick)")


if __name__ == "__main__":
    main()
