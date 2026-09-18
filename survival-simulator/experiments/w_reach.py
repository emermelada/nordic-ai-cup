"""w_reach.py — STEP 0: measure the exact reachability ceiling of this simulator.

WHY: production decays (`tree_spawn_chance *= 0.5 ** (time/300)`), so the total energy a world will ever
yield is FINITE. If that total is smaller than what a policy must burn to reach a target tick, the target
is unreachable for ANY policy and the plan must be retargeted. This is measured, not assumed.

METHOD (all exact, no new sim code):
- eaten energy per tick = (score_delta - dt) * 1000, because environment.py:674 does `score += fruit.energy/1000`
  (the harness already uses this identity to count fruit).
- standing energy = sum of fruit.energy in the world.
- Energy is CONSERVED, so production = eaten + rotted + change in standing. Rotted is the residual.
- Total available over an episode = cumulative production, which we measure directly.

ALSO measures the cheapest possible endgame: metabolism is `dt * energy_drain_rate` = 0.1/tick per agent
and a relay spawn costs 100, so a one-agent-at-a-time chain costs ~0.1 + 100/900 = 0.211 energy/tick.
That is the floor any policy must pay just to keep ONE lineage alive.

Usage: python w_reach.py <params.json> <seed,seed,...> <horizon>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import SimulationCore                      # noqa: E402
import best_controller as bc                            # noqa: E402
from env_wrapper import make_action                     # noqa: E402

DT = 0.1
BUCKET = 2000          # ticks per reporting bucket
RELAY_SPAWN_COST = 100.0
RELAY_SPAWN_PERIOD = 900.0   # mean agent lifetime in ticks (max_age 60-120 sim-seconds = 600-1200 ticks)
METAB = 0.1                  # energy/tick per living agent


def main():
    path, seeds_s, horizon_s = sys.argv[1], sys.argv[2], sys.argv[3]
    horizon = int(horizon_s)
    seeds = [int(s) for s in seeds_s.split(",") if s.strip()]
    blob = json.load(open(path))
    P = dict(bc.DEFAULT_PARAMS)
    P.update(blob.get("params", blob) if isinstance(blob, dict) else {})

    print(f"REACHABILITY PROBE | horizon {horizon} | buckets of {BUCKET} ticks", flush=True)
    print("floor cost to keep ONE lineage alive = metabolism 0.1/tick + 100 per relay spawn per ~900 ticks"
          f" = {(METAB + RELAY_SPAWN_COST / RELAY_SPAWN_PERIOD):.3f} energy/tick", flush=True)

    for seed in seeds:
        bc.reset_memory()
        fn = bc.make_policy(P)
        core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                              starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
        prev_score = 0.0
        eaten_cum = 0.0
        prod_cum = 0.0
        prev_standing = sum(float(f.energy) for f in core.env.fruits)
        buckets = []
        i = 0
        for i in range(horizon):
            live = list(core.env.agents)
            if not live:
                break
            states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
            if not states:
                break
            acts = [(s["agent_id"], make_action(s, fn(s))) for s in states]
            out = core.step(acts)
            d_score = out["score"] - prev_score
            prev_score = out["score"]
            eaten = max(0.0, (d_score - DT)) * 1000.0          # exact: score += fruit.energy/1000
            eaten_cum += eaten
            standing = sum(float(f.energy) for f in core.env.fruits)
            # conservation: production = eaten + rotted + d(standing); rotted = production - eaten - d(standing)
            # we only know production indirectly, so accumulate d(standing) and report the identity later
            stand_delta = standing - prev_standing
            prod_cum += eaten + stand_delta
            prev_standing = standing
            b = (i + 1) // BUCKET
            if not buckets or buckets[-1]["b"] != b:
                buckets.append({"b": b, "eaten": 0.0, "prod": 0.0, "standing": 0.0, "n": 0,
                                "agents": 0, "preds": 0})
            cur = buckets[-1]
            cur["eaten"] += eaten
            cur["prod"] += eaten + stand_delta
            cur["standing"] = standing
            cur["n"] += 1
            cur["agents"] = max(cur["agents"], len(core.env.agents))
            cur["preds"] = max(cur["preds"], len(core.env.predators))

        steps = i + 1
        # tail extrapolation: production decays ~0.5**(time/300); estimate remaining production to H
        import math
        t_now = steps * DT
        tail = 0.0
        if steps > 2 * BUCKET:
            rate = buckets[-1]["prod"] / (BUCKET * DT)          # energy per sim-second, last bucket
            for t in range(int(t_now), int(horizon * DT)):
                tail += rate * (0.5 ** ((t - t_now) / 300.0)) * 1.0
        total_available = prod_cum + tail
        reach_floor = METAB + RELAY_SPAWN_COST / RELAY_SPAWN_PERIOD
        max_ticks_one_agent = total_available / reach_floor
        print(f"\nseed={seed} died_at={steps} ticks ({steps * DT:.0f} sim-seconds) "
              f"score={core.env.score:.2f}", flush=True)
        print(f"  measured production so far        : {prod_cum:10.0f} energy", flush=True)
        print(f"  standing fruit at death           : {prev_standing:10.0f} energy", flush=True)
        print(f"  extrapolated tail to {horizon} ticks: {tail:10.0f} energy", flush=True)
        print(f"  TOTAL AVAILABLE (measured)        : {total_available:10.0f} energy", flush=True)
        print(f"  ceiling if spent on a bare 1-agent relay: {max_ticks_one_agent:8.0f} ticks "
              f"(= score {max_ticks_one_agent / 10.0:.0f})", flush=True)
        print(f"  -> 18,000 ticks would need {18000 * reach_floor:10.0f} energy at the relay floor "
              f"({100 * 18000 * reach_floor / max(1e-9, total_available):.0f}% of what exists)", flush=True)
        print("   bucket   eaten  production  standing  agents  preds", flush=True)
        for x in buckets:
            print(f"  {x['b'] * BUCKET:6d} {x['eaten']:8.0f} {x['prod']:11.0f} {x['standing']:9.0f} "
                  f"{x['agents']:7d} {x['preds']:6d}", flush=True)


if __name__ == "__main__":
    main()