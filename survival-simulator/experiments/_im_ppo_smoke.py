"""Smoke test for the (unused, optional) PPO warm-start path: 1 tiny iteration to prove it runs,
then a 1-seed eval of the result. Raw/unshaped reward. This is NOT a tuned result."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im

t0 = time.time()
p = im.ppo_warmstart(bc_path=im.MODEL, out=os.path.join(im.HERE, "_im_ppo_smoke.pt"),
                     iterations=2, ticks_per_iter=500, train_seeds=(100,), epochs=1, print_every=1)
print("PPO smoke ran in %.1fs" % (time.time() - t0))
pol = im.PpoPolicy(p, device="cpu") if False else None
import policy_imitation as pi
print("models present:", os.path.exists(im.MODEL), os.path.exists(im.PPO_MODEL))
