"""Spawn-head diagnostics on the VAL data: probability distribution, AUC, F1-optimal threshold,
and how many spawn decisions the expert makes per state (cooldown => ill-posed labels)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else im.MODEL
pol = pi.ImitationPolicy(MODEL)
d = np.load(im.DATA, allow_pickle=True)
sd = d["seed"].astype(np.int64)
oid = d["variant_id"].astype(np.int64)
names = [str(x) for x in d["variant_names"]]
obs = d["obs"].astype(np.float32); act = d["act"].astype(np.float32)
for tag, sel in (("all-variants", np.isin(sd, im.VAL_SEEDS)),
                 ("expert-only", np.isin(sd, im.VAL_SEEDS) & (oid == names.index("expert")))):
    X = (obs[sel] - pol.obs_mean) / pol.obs_std
    ps = []
    with torch.no_grad():
        for b0 in range(0, len(X), 16384):
            _, lg = pol.net(torch.tensor(X[b0:b0 + 16384], dtype=torch.float32, device=pol.device))
            ps.append(torch.sigmoid(lg).cpu().numpy())
    p = np.concatenate(ps)
    y = (act[sel, 3] > 0.5).astype(np.int8)
    order = np.argsort(-p)
    ysort = y[order]
    tp = np.cumsum(ysort)
    k = np.arange(1, len(y) + 1)
    prec = tp / k
    rec = tp / max(int(y.sum()), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(np.argmax(f1))
    thr_f1 = float(p[order][bi])
    # rate-matched threshold
    thr_rate = float(np.quantile(p, 1.0 - y.mean())) if y.mean() > 0 else 0.5
    print("[%s] n=%d pos_rate=%.5f | mean_p=%.5f median_p=%.4f p99=%.3f p99.9=%.3f max=%.3f"
          % (tag, len(p), y.mean(), p.mean(), np.median(p), np.quantile(p, 0.99),
             np.quantile(p, 0.999), p.max()))
    print("   F1-optimal thr=%.3f -> F1=%.3f prec=%.3f rec=%.3f | rate-matched thr=%.3f"
          % (thr_f1, f1[bi], prec[bi], rec[bi], thr_rate))
    for t in (0.5, 0.3, 0.2, 0.1, 0.05):
        pr = p > t
        tp_ = int((pr & (y == 1)).sum()); fp_ = int((pr & (y == 0)).sum())
        fn_ = int((~pr & (y == 1)).sum())
        print("     thr=%.2f rate=%.5f (x expert %.1f) prec=%.3f rec=%.3f f1=%.3f"
              % (t, pr.mean(), pr.mean() / max(y.mean(), 1e-9), tp_ / max(tp_ + fp_, 1),
                 tp_ / max(tp_ + fn_, 1), 2 * tp_ / max(2 * tp_ + fp_ + fn_, 1)))
