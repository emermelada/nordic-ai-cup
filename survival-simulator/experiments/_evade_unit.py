"""Unit check: does the evade branch actually fire, and only when it should?"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import best_controller as bc

H1 = json.load(open(os.path.join(os.path.dirname(HERE), "best_controller", "params.json")))

def state(energy=300.0, pred_d=100.0, pred_a=0.3):
    return {"agent_id": 1, "energy": energy, "biome": "forest", "age": 10.0, "speed": 10.0,
            "sprint_speed": 20.0, "hearing_radius": 50.0, "vision_angle": 1.047,
            "vision_range": 200.0, "max_energy": 500.0,
            "observations": [{"type": "Predator", "distance": pred_d, "angle": pred_a,
                              "rel_dir": 0.0},
                             {"type": "Fruit", "distance": 80.0, "angle": 0.5}]}

P_off = dict(bc.DEFAULT_PARAMS); P_off.update(H1)
P_on = dict(P_off); P_on.update({"evade_mode": 1.0, "evade_dist": 140.0,
                                 "evade_disengage": 240.0, "evade_speed_frac": 0.8,
                                 "evade_energy_abs": 130.0})
cases = [("OFF  pred@100",        P_off, state()),
         ("ON   pred@100",        P_on,  state()),
         ("ON   pred@100 e=90",   P_on,  state(energy=90.0)),
         ("ON   pred@300 far",    P_on,  state(pred_d=300.0)),
         ("ON   pred@100 a=-0.9", P_on,  state(pred_a=-0.9))]
for label, P, st in cases:
    bc.reset_memory()
    a = bc.make_policy(P)(st)
    print(f"{label:22s} -> dist={a[0]:7.3f} dir={a[1]:+8.3f} turn={a[2]:+7.3f} spawn={a[3]:.0f}")
print()
print("expected for ON/pred@100: dist=16.0 (0.8*20), dir=+3.442 (0.3+pi), turn=-0.300 (face it)")
print("expected for OFF and for the low-energy / far cases: the normal forage action (turn=0 or facing only)")
