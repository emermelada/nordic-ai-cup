"""SELECTION-PRESSURE DOSE-RESPONSE for the V2 genome ratchet.

Single arms are not evidence for a constraint: the convincing shape is a DOSE-RESPONSE (vary the
knob's strength and show the outcome moving monotonically, on more than one seed set). Here the dose
is selection pressure = how few breeders are allowed (gs_topk 1, 2, 3, 5) against no selection at all.

The two forces in tension, and why this is a measurement not an opinion:
  * STRONGER pressure (small k) concentrates the genome faster per birth;
  * but selection also CUTS BIRTHS (measured: 455 -> ~230 at 10,000 ticks), and births are the only
    source of new genomes -- a ratchet needs rolls of the dice.
So there should be an interior optimum, and its location is what this measures.

Reuses the verified harness in w_v2.py (same env, same telemetry, same identity-checked code).

Usage: python w_v2_dose.py [seeds] [horizon]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import w_v2 as W  # noqa: E402  (module-level load of the edited controller + LIVE params)


def main():
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "101,102,103,106").split(",")]
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 12000

    arms = [("LIVE(no selection)", {}),
            ("topk1", {"genome_select": 1.0, "gs_topk": 1.0}),
            ("topk2", {"genome_select": 1.0, "gs_topk": 2.0}),
            ("topk3", {"genome_select": 1.0, "gs_topk": 3.0}),
            ("topk5", {"genome_select": 1.0, "gs_topk": 5.0})]

    print(f"=== DOSE-RESPONSE: selection pressure (breeders allowed), horizon {horizon}, seeds {seeds} ===",
          flush=True)
    rows = []
    for label, over in arms:
        P = dict(W.LIVE)
        P.update(over)
        P.update({"evade_mode": 0.0, "thin_relay": 0.0})   # isolate the genome layer only
        out = []
        for sd in seeds:
            steps, fruits, spawns, births, core, pops, vsamp, esamp = W.run_traced(
                W.NEW.make_policy(W.params(W.NEW, P)), W.NEW, sd, horizon)
            bv = [x["vision_range"] for x in births if x["vision_range"] == x["vision_range"]]
            out.append({"sd": sd, "steps": steps, "fruits": fruits, "births": len(births),
                        "pop": float(np.mean(pops)) if pops else float("nan"),
                        "vis": float(np.mean(vsamp)) if vsamp else float("nan"),
                        "vis_late": float(np.mean(vsamp[-8:])) if len(vsamp) >= 8 else float("nan"),
                        "energy": float(np.mean(esamp)) if esamp else float("nan"),
                        "nb_vis": float(np.mean(bv)) if bv else float("nan")})
        m = lambda key: float(np.nanmean([o[key] for o in out]))          # noqa: E731
        rows.append((label, m("steps"), m("fruits"), m("births"), m("pop"),
                     m("vis"), m("vis_late"), m("nb_vis")))
        print(f"  {label:20s} steps {m('steps'):7.0f} | fruits {m('fruits'):7.0f} | births {m('births'):6.1f} | "
              f"pop {m('pop'):5.1f} | vision {m('vis'):6.1f} late {m('vis_late'):6.1f} | "
              f"newborn vision {m('nb_vis'):6.1f}", flush=True)
        for o in out:
            print(f"      seed {o['sd']}: steps={o['steps']:6d} births={o['births']:3d} pop={o['pop']:5.1f} "
                  f"vision={o['vis']:6.1f}/{o['vis_late']:6.1f} newborn={o['nb_vis']:6.1f}", flush=True)

    base = rows[0][1]
    print("\n=== DOSE-RESPONSE SUMMARY (vs LIVE) ===", flush=True)
    for label, steps, fruits, births, pop, vis, vlate, nb in rows:
        print(f"  {label:20s} {steps:7.0f} ticks ({steps - base:+7.1f}, {(steps / base - 1) * 100:+5.1f}%) "
              f"vision {vis:6.1f} newborn {nb:6.1f}", flush=True)


if __name__ == "__main__":
    main()
