import os, sys, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im

print("agg test:", im.agg([{"ticks": 3}, {"ticks": 9}], "ticks"))
rows = im.eval_policy(im.expert_policy(), [200], horizon=600, reset_fn=im.bc.reset_memory, label="expert")
print("rows:", json.dumps(rows))
print("agg rows:", json.dumps(im.agg(rows, "ticks")))
sel = {}
sel["expert"] = {"keys": {}, "train_seed_rows": rows, "ticks": im.agg(rows, "ticks")}
print("sel ok", json.dumps(sel)[:200])

# tune-equivalent loop, one config
pol = im.ImitationPolicy(im.MODEL, spawn_cooldown=None, spawn_threshold=None)
r2 = im.eval_policy(pol, [200], horizon=600, reset_fn=im.pi.reset_memory, label="imit")
print("tune-style ok", json.dumps(im.agg(r2, "ticks")))
