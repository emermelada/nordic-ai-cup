"""ENERGY LEDGER — decompose the fleet's energy budget over time.

Why: score is +0.1/tick REGARDLESS of population, metabolism is -0.1/tick PER AGENT
(energy_drain_rate is 1.0 for every biome; dt = 1/10), movement is 0.05/unit walking but
speed*0.05 + (dist-speed)*0.5 sprinting, and a spawn costs the parent a flat 100 (the child
starts at 75). Food income decays as 0.5^(t/300). So survival = how long income covers
(metabolism*n + movement + spawns).

Reconstructed ENTIRELY from the policy's own chosen actions + the sim's score deltas:
  score_delta = 0.1 + fruit_energy/1000 - victim_energy/100   -> net_income = (delta-0.1)*1000

Usage: python energy_budget.py <configs.json> <horizon> <seed> [bucket]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

if not hasattr(bc, "reset_memory"):
    raise SystemExit("FATAL: wrong best_controller loaded (%s)" % getattr(bc, "__file__", "?"))

DT = 0.1          # core.py: dt=1/10
DRAIN = 1.0       # biome.py: energy_drain_rate default 1.0, never overridden
WALK, SPRINT = 0.05, 0.5
SPAWN_COST = 100.0
# canonical params: <repo>/best_controller/params.json (experiments/ holds only a symlink to it)
PARAMS_PATH = os.path.join(os.path.dirname(HERE), "best_controller", "params.json")


def baseline():
    P = dict(DEFAULT_PARAMS)
    try:
        with open(PARAMS_PATH) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def move_cost(dist, speed):
    if dist <= 0:
        return 0.0
    if dist <= speed:
        return dist * WALK
    return speed * WALK + (dist - speed) * SPRINT


def run(P, seed, horizon, bucket):
    fn = bc.make_policy(P)
    led = []
    prev_score = [0.0]

    def recorder(i, livestates, acts, out):
        metabolism = 0.0
        movement = 0.0
        spawning = 0.0
        for pair in (acts or []):
            try:
                aid, act = pair
                s = next((x for x in (livestates or []) if x and x.get("agent_id") == aid), None)
                if s is None:
                    continue
                metabolism += DT * DRAIN
                movement += move_cost(float(act.move_distance), float(s["speed"]))
                if getattr(act, "spawn_agent", False):
                    spawning += SPAWN_COST
            except Exception:
                pass
        delta = out["score"] - prev_score[0]
        prev_score[0] = out["score"]
        led.append({"t": i, "n": out["num_agents"], "meta": metabolism, "move": movement,
                    "spawn": spawning, "income": (delta - DT) * 1000.0})

    r = run_eval_episode(fn, n_agents=5, seed=seed, horizon=horizon, stop_on_death=True,
                         recorder=recorder, reset_fn=bc.reset_memory)
    print("=" * 100)
    print("ticks=%d  score=%.1f  fruits=%d  predated=%d  births=%d  final_n=%d"
          % (r["steps"], r["score"], r["fruits_eaten"], r["predated"], r["spawns"], r["final_agents"]))
    print("bucket    n_mean |  metabolism      move     spawns |   BURN  |  income |  net")
    cum = {"meta": 0.0, "move": 0.0, "spawn": 0.0, "income": 0.0}
    for b0 in range(0, r["steps"], bucket):
        seg = [x for x in led if b0 <= x["t"] < b0 + bucket]
        if not seg:
            continue
        m = sum(x["meta"] for x in seg); v = sum(x["move"] for x in seg)
        s = sum(x["spawn"] for x in seg); inc = sum(x["income"] for x in seg)
        nmean = sum(x["n"] for x in seg) / len(seg)
        burn = m + v + s
        cum["meta"] += m; cum["move"] += v; cum["spawn"] += s; cum["income"] += inc
        print("%5d-%5d %5.1f | %11.0f %9.0f %10.0f | %6.0f | %7.0f | %+5.0f"
              % (b0, b0 + bucket, nmean, m, v, s, burn, inc, inc - burn))
    tot_burn = cum["meta"] + cum["move"] + cum["spawn"]
    print("-" * 100)
    print("TOTAL metabolism=%.0f (%.0f%%)  movement=%.0f (%.0f%%)  spawns=%.0f (%.0f%%)  BURN=%.0f"
          % (cum["meta"], 100 * cum["meta"] / max(tot_burn, 1), cum["move"],
             100 * cum["move"] / max(tot_burn, 1), cum["spawn"],
             100 * cum["spawn"] / max(tot_burn, 1), tot_burn))
    print("TOTAL income=%.0f  ->  NET=%.0f" % (cum["income"], cum["income"] - tot_burn))
    print("mean population=%.2f   metabolism per tick = %.3f" %
          (sum(x["n"] for x in led) / max(len(led), 1), sum(x["meta"] for x in led) / max(len(led), 1)))
    return r, led


if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1]))
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
    seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "100").split(",")]
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    print("ENERGY-LEDGER controller=%s" % bc.__file__)
    for c in cfgs:
        P = baseline(); P.update(c.get("params", {}))
        print("\n### %s" % c["tag"])
        for s in seeds:
            run(P, s, H, B)
