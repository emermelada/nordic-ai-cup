#!/usr/bin/env python3
"""Pure-logic unit test of the stray-payload guard in hive_next.py — no simulator, no workers.

Feeds a crafted request stream and checks the three cases that matter:
  1. clean stream                      -> byte-identical decisions to the old rule (no behaviour change)
  2. probe injected MID-game           -> state survives (tick keeps counting, memory intact)
  3. probe between games, then a game  -> the new game starts from a clean state, not the probe's
"""
import copy
import importlib.util
import os
import sys

HERE = "/Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/experiments"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old = load("hive_old", sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gold1815", "hive.py"))
new = load("hive_next", sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "hive_next.py"))
print("OLD = %s\nNEW = %s" % (old.__file__, new.__file__))


def agent(aid, x_off=0.0):
    return {"agent_id": aid, "energy": 300.0, "biome": "forest", "age": 30.0, "speed": 12.0,
            "sprint_speed": 24.0, "hearing_radius": 70.0, "vision_angle": 1.2, "vision_range": 320.0,
            "max_energy": 600.0,
            "observations": [{"type": "fruit", "distance": 60.0 + x_off, "angle": 0.2},
                             {"type": "tree", "distance": 40.0 + x_off, "angle": -0.1},
                             {"type": "edge", "distance": 500.0, "angle": 0.0}]}


def step(t, n=3):
    return {"game_status": "ok", "score": t, "sim_time": t, "n_agents": n,
            "agent_status": [agent(i) for i in range(n)]}


PROBE = {"game_status": "ok", "score": 123.4, "sim_time": 0.0, "n_agents": 1,
         "agent_status": [{"agent_id": 0, "energy": 100.0, "biome": "forest", "age": 1.0,
                           "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 60.0,
                           "vision_angle": 1.0, "vision_range": 300.0, "max_energy": 500.0,
                           "observations": [{"type": "edge", "distance": 100.0, "angle": 0.0},
                                            {"type": "predator", "distance": 200.0, "angle": 0.5},
                                            {"type": "tree", "distance": 50.0, "angle": -0.3}]}]}


def digest(h, n):
    return [h.decide(step(round(i * 0.1, 1), n)) for i in range(120)]


fails = []

# ---- 1. clean stream: same decisions as the old controller
a = old.Hive(seed=0)
b = new.Hive(seed=0)
da, db = digest(a, 3), digest(b, 3)
print("1. clean stream identical to old rule      :", da == db, "(120 decisions)")
if da != db:
    fails.append("clean stream diverged")

# ---- 1b. clean stream with probe_guard switched OFF reproduces the old rule exactly
c = new.Hive(seed=0, params={"probe_guard": 0.0})
print("1b. probe_guard=0 == old rule on stray stream: ", end="")
d, e = old.Hive(seed=0), new.Hive(seed=0, params={"probe_guard": 0.0})
for i in range(40):
    d.decide(step(round(i * 0.1, 1), 3))
    e.decide(step(round(i * 0.1, 1), 3))
d.decide(copy.deepcopy(PROBE))
e.decide(copy.deepcopy(PROBE))
same = True
for i in range(40, 80):
    same &= d.decide(step(round(i * 0.1, 1), 3)) == e.decide(step(round(i * 0.1, 1), 3))
print(same)
if not same:
    fails.append("probe_guard=0 path differs from old rule")

# ---- 2. probe injected MID-game: the live game must survive
h = new.Hive(seed=0)
for i in range(300):
    h.decide(step(round(i * 0.1, 1), 3))
tick_before, mem_before = h.tick, set(h.mem)
h.decide(copy.deepcopy(PROBE))
h.decide(step(300 * 0.1, 3))          # the live game resumes where it left off
print("2. mid-game probe: strays=%d restores=%d tick %d->%d mem %d->%d"
      % (h.counts["strays"], h.counts["restores"], tick_before, h.tick, len(mem_before), len(h.mem)))
if not (h.counts["strays"] == 1 and h.counts["restores"] == 1 and h.tick == tick_before + 1
        and set(h.mem) == mem_before):
    fails.append("mid-game probe damaged the live game")

# ---- 2b. the same treatment for the OLD controller, for contrast
o = old.Hive(seed=0)
for i in range(300):
    o.decide(step(round(i * 0.1, 1), 3))
otick, omem = o.tick, set(o.mem)
o.decide(copy.deepcopy(PROBE))
print("2b. old controller after the same probe   : tick %d->%d  mem %d->%d  (state WIPED)"
      % (otick, o.tick, len(omem), len(o.mem)))

# ---- 3. probe between games, then a genuinely new game
g = new.Hive(seed=0)
for i in range(400):
    g.decide(step(round(i * 0.1, 1), 3))
probe77 = copy.deepcopy(PROBE)
probe77["agent_status"][0]["agent_id"] = 77          # distinguishable from the game's own agents
g.decide(probe77)                                    # the platform's pre-flight check, after game 1
g.decide(step(0.1, 3))                               # game 2's first tick
ok = (g.tick == 2 and set(g.mem) == {0, 1, 2} and g.counts["strays"] == 1)
print("3. new game after a probe: tick=%d mem=%s strays=%d clean=%s"
      % (g.tick, sorted(g.mem), g.counts["strays"], ok))
if not ok:
    fails.append("probe leaked into the next game")

# ---- 4. game -> game without any probe must still behave like the old rule (reset each game)
h1, h2 = new.Hive(seed=0), old.Hive(seed=0)
for i in range(200):
    h1.decide(step(round(i * 0.1, 1), 3))
    h2.decide(step(round(i * 0.1, 1), 3))
same = True
for i in range(200):
    same &= h1.decide(step(round(i * 0.1, 1), 3)) == h2.decide(step(round(i * 0.1, 1), 3))
print("4. game-to-game transition identical       :", same, " tick=%d" % h1.tick)
if not same:
    fails.append("game-to-game transition diverged")

print("\n%s" % ("ALL GUARD CHECKS PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)