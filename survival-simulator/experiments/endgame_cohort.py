"""ENDGAME COHORT DIAGNOSTIC — why does the relay chain break?

Hypothesis under test (H1, "synchronised cohorts"):
  `repro_global_target` / `repro_popcap` switch reproduction OFF while the population is at its
  PEAK. The crowd then ages together. Since agents are mortal by age (max_age 60-120 sim-seconds,
  after which energy drains 0.01*age per tick -> ~1.2/tick at age 120, ~20x the biome drain), a
  synchronised cohort dies together past max_age. The tail survivors are then both OLD (bleeding
  faster than they can forage) and unable to fund a spawn (sim hard-requires energy > 100, child
  starts at 75) -> zero births -> wipe.

What this prints per (config, seed):
  collapse_tick, births in the FINAL 2000 ticks, births total,
  the age/energy of the survivors at the moment the population first falls to <=3 and <=1,
  and the population trace downsampled.

Usage:  python endgame_cohort.py <configs.json> <horizon> <seeds "train"|"eval"|csv>
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# SINGLE SOURCE OF TRUTH: <repo>/best_controller.py is the only controller; experiments/ holds no
# copy. The file it loaded is asserted below so a stale duplicate can never skew a comparison.
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from env_wrapper import run_eval_episode
import best_controller as bc
from best_controller import DEFAULT_PARAMS

if not hasattr(bc, "reset_memory"):
    raise SystemExit("FATAL: controller without reset_memory loaded (%s)" % getattr(bc, "__file__", "?"))
if os.path.realpath(getattr(bc, "__file__", "")) != os.path.realpath(os.path.join(REPO, "best_controller.py")):
    raise SystemExit("FATAL: canonical controller shadowed by %s" % bc.__file__)

PARAMS_PATH = os.path.join(REPO, "best_controller", "params.json")
TRAIN_SEEDS = list(range(100, 900, 100))
EVAL_SEEDS = list(range(1000, 1800, 100))
N_AGENTS = 5
FINAL_WINDOW = 2000  # ticks; "the final stretch" from the endgame diagnosis


def baseline():
    P = dict(DEFAULT_PARAMS)
    try:
        with open(PARAMS_PATH) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def run_one(P, seed, horizon):
    fn = bc.make_policy(P)
    rec = []
    seen_keys = set()

    def recorder(i, livestates, acts, out):
        # acts == [(agent_id, ActionRequest), ...]  (env_wrapper.py:185)
        births = 0
        for pair in (acts or []):
            try:
                if getattr(pair[1], "spawn_agent", False):
                    births += 1
            except Exception:
                pass
        for s in (livestates or []):
            if isinstance(s, dict) and not seen_keys:
                seen_keys.update(s.keys())
        ages = [s["age"] for s in (livestates or []) if isinstance(s, dict) and "age" in s]
        en = [s.get("energy", 0.0) for s in (livestates or []) if isinstance(s, dict)]
        rec.append({"t": i, "n": out["num_agents"], "births": births,
                    "n_before": len(livestates or []), "ages": ages, "e": en})

    r = run_eval_episode(fn, n_agents=N_AGENTS, seed=seed, horizon=horizon,
                         stop_on_death=True, recorder=recorder, reset_fn=bc.reset_memory)
    if seen_keys and os.environ.get("EG_KEYS") != "0":
        print("   [state keys: %s]" % sorted(seen_keys))
        os.environ["EG_KEYS"] = "0"
    return r, rec


def snap(rec, pred):
    """First record satisfying pred -> (tick, ages, energies, n)."""
    for x in rec:
        if pred(x):
            return x
    return None


def report(tag, P, seeds, horizon):
    print("=" * 96)
    print("%s   seeds=%s horizon=%d" % (tag, seeds, horizon))
    print("   gate: repro_frac=%s/min=%s abs=%s target=%s popcap=%s cd=%s"
          % (P.get("repro_frac"), P.get("repro_frac_min"), P.get("repro_energy_abs"),
             P.get("repro_global_target"), P.get("repro_popcap"), P.get("spawn_cooldown")))
    rows = []
    for s in seeds:
        r, rec = run_one(P, s, horizon)
        ticks = r["steps"]
        fin = [x for x in rec if x["t"] > ticks - FINAL_WINDOW]
        late_births = sum(x["births"] for x in fin)
        early = [x for x in rec if x["t"] <= ticks - FINAL_WINDOW]
        early_births = sum(x["births"] for x in early)
        d3 = snap(rec, lambda x: x["n"] <= 3)
        d1 = snap(rec, lambda x: x["n"] <= 1)
        age3 = round(max(d3["ages"]), 1) if d3 and d3["ages"] else None
        e3 = round(d3["e"][0], 1) if d3 and d3["e"] else None
        age1 = round(max(d1["ages"]), 1) if d1 and d1["ages"] else None
        rows.append(dict(seed=s, ticks=ticks, score=round(r["score"], 1),
                         births=r["spawns"], late=late_births, early=early_births,
                         t_d3=d3["t"] if d3 else None, age_d3=age3, e_d3=e3,
                         t_d1=d1["t"] if d1 else None, age_d1=age1,
                         peak=max(x["n"] for x in rec) if rec else 0,
                         peak_t=next((x["t"] for x in rec if x["n"] == max(y["n"] for y in rec)), 0)))
    print("   %-5s %7s %7s %7s %7s %7s %8s %8s %8s" %
          ("seed", "ticks", "score", "births", "late2k", "early", "peak@t", "age@n<=3", "age@n<=1"))
    for x in rows:
        print("   %-5d %7d %7.1f %7d %7d %7d %8s %8s %8s"
              % (x["seed"], x["ticks"], x["score"], x["births"], x["late"], x["early"],
                 "%d@%d" % (x["peak"], x["peak_t"]), x["age_d3"], x["age_d1"]))
    tk = [x["ticks"] for x in rows]
    print("   MEAN ticks=%.1f median=%.1f min=%d | total births=%.1f | births in final %d ticks=%.1f"
          % (st.mean(tk), st.median(tk), min(tk), st.mean([x["births"] for x in rows]),
             FINAL_WINDOW, st.mean([x["late"] for x in rows])))
    return rows


if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1]))
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
    arg = sys.argv[3] if len(sys.argv) > 3 else "train"
    seeds = TRAIN_SEEDS if arg == "train" else (EVAL_SEEDS if arg == "eval" else [int(x) for x in arg.split(",")])
    print("ENDGAME-COHORT controller=%s" % bc.__file__)
    out = {}
    for c in cfgs:
        P = baseline()
        P.update(c.get("params", {}))
        out[c["tag"]] = report(c["tag"], P, seeds, H)
