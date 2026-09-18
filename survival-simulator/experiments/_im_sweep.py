"""Train-seed spawn-threshold sweep (model selection only; eval seeds 1000..1700 untouched)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi

MODEL = sys.argv[1] if len(sys.argv) > 1 else im.MODEL
H = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
seeds = [200, 400, 600]
configs = [(120, 0.5), (120, 0.35), (120, 0.2)]
out = {}
for cd, th in configs:
    pol = pi.ImitationPolicy(MODEL, spawn_cooldown=cd, spawn_threshold=th, device="cpu")
    lbl = "cd%d_thr%.2f" % (pol.spawn_cooldown, pol.spawn_threshold)
    t0 = time.time()
    rows = im.eval_policy(pol, seeds, horizon=H, reset_fn=pol.reset, label=lbl)
    out[lbl] = {"model": MODEL, "cooldown": pol.spawn_cooldown, "threshold": pol.spawn_threshold,
                "rows": rows, "ticks": im.agg(rows, "ticks"), "score": im.agg(rows, "score"),
                "spawns": im.agg(rows, "spawns"), "predated": im.agg(rows, "predated"),
                "pop_peak": im.agg(rows, "pop_peak"), "final_agents": im.agg(rows, "final_agents")}
    print("  %-14s median %.0f mean %.0f | spawns %.1f pred %.1f pop_end %.1f | %.0fs"
          % (lbl, out[lbl]["ticks"]["median"], out[lbl]["ticks"]["mean"], out[lbl]["spawns"]["mean"],
             out[lbl]["predated"]["mean"], out[lbl]["final_agents"]["mean"], time.time() - t0),
          flush=True)
    with open(os.path.join(im.HERE, "imitation_spawn_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
print(json.dumps({k: v["ticks"] for k, v in out.items()}, indent=1))
