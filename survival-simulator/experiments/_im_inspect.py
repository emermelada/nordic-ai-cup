import numpy as np, json, sys
p = sys.argv[1]
d = np.load(p, allow_pickle=True)
for k in d.files:
    a = d[k]
    print(k, getattr(a, "shape", None), getattr(a, "dtype", None))
obs = d["obs"].astype(np.float32)
act = d["act"]
print("obs min/max", obs.min(0).round(2)[:8], obs.max(0).round(2)[:8])
print("act min", act.min(0).round(3), "max", act.max(0).round(3))
print("act mean", act.mean(0).round(3))
print("spawn rate", float(act[:, 3].mean()))
print("nan obs", np.isnan(obs).any(), "nan act", np.isnan(act).any())
print("variant counts", np.bincount(d["variant_id"].astype(int)).tolist())
print("variants", list(d["variant_names"]))
print("u_t", np.unique(d["t"]).shape)
