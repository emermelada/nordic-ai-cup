"""Finalise results: the train-seed-selected spawn threshold (0.2) NEGATIVELY transferred to the
held-out seeds (mean 1307 vs 2569), so the shipped checkpoint keeps the untuned gate-semantics
threshold 0.5.  Both runs are kept in imitation_results.json, labelled.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi
import torch

ck = torch.load(im.MODEL, map_location="cpu", weights_only=False)
ck["spawn_threshold"] = 0.5
ck["config"]["spawn_threshold"] = 0.5
ck["config"]["spawn_threshold_note"] = (
    "gate-label semantics: spawn when P(expert spawns within 120 ticks) > 0.5. Left UNTUNED on "
    "purpose: a threshold of 0.2 chosen on 3 TRAIN seeds scored higher there (mean 4988 vs 2753) but "
    "scored worse on the held-out seeds (mean 1307 vs 2569) -> small-sample selection does not transfer.")
torch.save(ck, im.MODEL)
print("reverted spawn_threshold to 0.5 in", im.MODEL)

res = json.load(open(im.RESULTS))
# reorder: 'imitation' (shipped, thr 0.5) then the overfit-threshold run
bad = res["policies"].pop("imitation", None)
res["policies"]["imitation"] = res["policies"].pop("imitation_thr0.5_default")
if bad is not None:
    res["policies"]["imitation_thr0.20_trainseed_selected"] = bad
res["policies"]["imitation_thr0.20_trainseed_selected"]["note"] = (
    "same weights, spawn threshold 0.2 selected on TRAIN seeds [200,400,600] horizon 6000 "
    "(mean 4988 there) -> did NOT transfer to held-out seeds (mean 1307): recorded as evidence that "
    "3-seed knob selection is over-fitting. Not shipped.")
res["notes"]["shipped_checkpoint"] = {"model": im.MODEL, "spawn_threshold": 0.5, "spawn_cooldown": 120,
                                     "obs": "build_obs(31) + private state(3) = 34"}
json.dump(res, open(im.RESULTS, "w"), indent=1)
for k, v in res["policies"].items():
    print("%-38s mean %6.0f median %6.0f pop_end %.1f" % (k, v["ticks"]["mean"], v["ticks"]["median"],
                                                          v["final_agents"]["mean"]))
