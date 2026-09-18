"""Apply the TRAIN-seed-selected spawn threshold to the checkpoint, then re-evaluate ONLY the
primary imitation model on the held-out EVAL seeds and merge the result into imitation_results.json.

Selection evidence: _im_sweep.py on TRAIN seeds [200,400,600] (horizon 6000) — threshold 0.50
=> mean 2753 / median 3576 with two early collapses; threshold 0.20 => mean 4988 / median 4875,
min 4089 (no collapse).  Eval seeds 1000..1700 are untouched by this selection.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi
import torch

THR = 0.2
ck = torch.load(im.MODEL, map_location="cpu", weights_only=False)
ck["spawn_threshold"] = THR
ck["config"]["spawn_threshold"] = THR
ck["config"]["spawn_threshold_selected_on"] = "train seeds 200,400,600 horizon 6000 (_im_sweep.py)"
torch.save(ck, im.MODEL)
print("set spawn_threshold=%.2f in %s" % (THR, im.MODEL))

res = json.load(open(im.RESULTS))
pol = pi.ImitationPolicy(im.MODEL, device="cpu")
t0 = time.time()
rows = im.eval_policy(pol, im.EVAL_SEEDS, im.HORIZON, reset_fn=pol.reset, label="imitation")
res["policies"]["imitation_thr0.5_default"] = res["policies"].pop("imitation")
res["policies"]["imitation"] = im.summarize(rows)
res["policies"]["imitation"]["model_path"] = im.MODEL
res["policies"]["imitation"]["seconds"] = round(time.time() - t0, 1)
res["policies"]["imitation"]["spawn_threshold"] = pol.spawn_threshold
res["policies"]["imitation"]["spawn_cooldown"] = pol.spawn_cooldown
res["policies"]["imitation_thr0.5_default"]["note"] = "same weights, spawn threshold 0.5 (calibration default)"
res["notes"] = {"spawn_threshold_selection": "train seeds only; see _im_sweep.py",
                "device": "cpu (MPS per-call dispatch overhead ~15 ms vs ~0.1 ms CPU)"}
json.dump(res, open(im.RESULTS, "w"), indent=1)
p = res["policies"]["imitation"]
print("imitation thr%.2f -> mean %.0f median %.0f std %.0f min %d max %d pop_end %.1f"
      % (THR, p["ticks"]["mean"], p["ticks"]["median"], p["ticks"]["std"], p["ticks"]["min"],
         p["ticks"]["max"], p["final_agents"]["mean"]))
