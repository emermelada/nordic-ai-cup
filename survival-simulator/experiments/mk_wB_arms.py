"""Generate the wB* thin-relay arm configs as FULL param sets (H1 + deltas).

A partial JSON silently means "DEFAULT_PARAMS minus the keys you forgot", which has already
produced a wrong control in this project. So every arm file is written from the full H1 baseline
plus explicit deltas, with json.dump -- never hand-written.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
H1 = json.load(open(os.path.join(HERE, "wH1_winEARN_liveSPEND.json")))   # == VPS h1_baseline_params.json
assert set(H1) >= {"forage_nearest", "spawn_cooldown", "repro_global_target"}

ARMS = {
    # The axis under test is TRIGGER TIMING vs the terminal cascade: the measured fleet death is at
    # ~4,800 ticks (W3's own 10-seed mean on seeds 1200-1219: 4,803), and the refuted 'bank late'
    # arms failed precisely because their windows (8,000-15,000) were never reached. So two arms
    # engage EARLY/marginally and one reproduces the too-late window so the null is visible.
    # (1) early: engages ~3,000 ticks, comfortably before the cascade, fleet thinned to 2-3
    "wB1_thin_c3000_v55_t2_m70.json": dict(
        thin_relay=1.0, thin_trig_tick=3000.0, thin_trig_vis=0.55, thin_trig_vis_off=0.80,
        thin_target_pop=2.0, thin_spawn_energy_abs=250.0, thin_rescue_energy_abs=160.0,
        thin_spawn_cooldown=200, thin_move_frac=0.70),
    # (2) marginal: engages ~4,500 ticks, right at the observed mean death time
    "wB2_thin_c4500_v45_t2_m75.json": dict(
        thin_relay=1.0, thin_trig_tick=4500.0, thin_trig_vis=0.45, thin_trig_vis_off=0.75,
        thin_target_pop=2.0, thin_spawn_energy_abs=250.0, thin_rescue_energy_abs=160.0,
        thin_spawn_cooldown=250, thin_move_frac=0.75),
    # (3) the DEFAULT_PARAMS window verbatim (6,000 ticks / vis 0.35): expected to be unreachable
    #     in most seeds -- included because "the window was never reached" is itself the finding.
    "wB3_thin_c6000_v35_t1_m85.json": dict(
        thin_relay=1.0, thin_trig_tick=6000.0, thin_trig_vis=0.35, thin_trig_vis_off=0.70,
        thin_target_pop=1.0, thin_spawn_energy_abs=220.0, thin_rescue_energy_abs=160.0,
        thin_spawn_cooldown=150, thin_move_frac=0.85),
}

for name, deltas in ARMS.items():
    p = dict(H1)
    p.update(deltas)
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(p, f, indent=1, sort_keys=True)
    print("wrote", name, "keys:", len(p))
