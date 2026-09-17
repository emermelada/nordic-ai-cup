"""Pipeline sanity check on the small smoke dataset + a 1-seed eval (NOT a result)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi

t0 = time.time()
p = im.train_bc(path=os.path.join(im.HERE, "_im_smoke.npz"), out=os.path.join(im.HERE, "_im_smoke_model.pt"),
                epochs=3, batch=4096, verbose=True)
print("TRAIN OK %.1fs" % (time.time() - t0))
print("BC metrics:", json.dumps(im.bc_metrics(p, path=os.path.join(im.HERE, "_im_smoke.npz"))))
# single-seed eval of both policies (train seed, tiny horizon: harness smoke only)
pol = pi.ImitationPolicy(p)
rows_im = im.eval_policy(pol, [100], horizon=1500, reset_fn=pi.reset_memory, label="imit")
rows_ex = im.eval_policy(im.expert_policy(), [100], horizon=1500, reset_fn=im.bc.reset_memory, label="expert")
print("ticks imit", rows_im[0]["ticks"], "expert", rows_ex[0]["ticks"])
print("SANITY OK %.1fs" % (time.time() - t0))
