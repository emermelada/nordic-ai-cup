"""w_confirm_residual.py - the 20-seed DECISION measurement for the W1 residual policy.

Runs the PURE BASE and BASE+RESIDUAL on IDENTICAL held-out seeds at the official horizon in one process,
so the comparison is paired (the only kind of claim this project trusts). numpy-only: it loads .npz
weights via NpResidual.from_npz, so it runs inside the eval image which has no torch.

Usage: python w_confirm_residual.py <weights.npz> <seed,seed,...> <horizon> <base_params.json>
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_residual import NpResidual, eval_residual          # noqa: E402
import best_controller as bc                               # noqa: E402


def main():
    npz, seeds_s, horizon_s, params_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    horizon = int(horizon_s)
    seeds = [int(s) for s in seeds_s.split(",") if s.strip()]
    blob = json.load(open(params_path))
    base_p = dict(bc.DEFAULT_PARAMS)
    base_p.update(blob.get("params", blob) if isinstance(blob, dict) else {})
    print(f"CONFIRM RESIDUAL | horizon {horizon} | {len(seeds)} paired seeds | base params: {params_path}",
          flush=True)
    print(f"  base identity: fruit_weight={base_p.get('fruit_weight'):.3f} walk_frac={base_p.get('walk_frac'):.3f} "
          f"evade_mode={base_p.get('evade_mode')}", flush=True)
    pol = NpResidual.from_npz(npz)
    print("  zero-check: residual mean output =", [round(float(x), 6) for x in pol.mu(
        __import__("numpy").zeros(34, dtype="float32"))][:3], "(informational only)", flush=True)

    m_b, per_b, f_b, fper_b = eval_residual(None, seeds, horizon, base_p)   # pure-base control
    m_r, per_r, f_r, fper_r = eval_residual(pol, seeds, horizon, base_p)    # base + residual
    won = sum(1 for a, b in zip(per_r, per_b) if a > b)
    deltas = [a - b for a, b in zip(per_r, per_b)]
    print(f"\n  PURE BASE      mean {m_b:8.1f} median {statistics.median(per_b):8.1f} min {min(per_b):6d} "
          f"fruits {f_b:7.1f}", flush=True)
    print(f"  BASE+RESIDUAL  mean {m_r:8.1f} median {statistics.median(per_r):8.1f} min {min(per_r):6d} "
          f"fruits {f_r:7.1f}", flush=True)
    print(f"  delta mean {m_r - m_b:+8.1f} ({(m_r - m_b) / max(1e-9, m_b) * 100:+.1f}%) | "
          f"median delta {statistics.median(per_r) - statistics.median(per_b):+.1f} | "
          f"seeds won {won}/{len(seeds)} | mean per-seed delta {statistics.mean(deltas):+.1f}", flush=True)
    print("\n  seed      base   residual    delta", flush=True)
    for sd, b, r in zip(seeds, per_b, per_r):
        print(f"  {sd:5d} {b:9d} {r:9d} {r - b:+8d}", flush=True)
    verdict = ("PROMOTE" if (m_r - m_b > 500 and won >= 0.6 * len(seeds))
               else "INCONCLUSIVE" if (m_r - m_b) > 0 else "REJECT")
    print(f"\n  VERDICT (bar: mean delta > 500 AND a majority of seeds won): {verdict}", flush=True)
    print(f"  raw per-seed base     : {per_b}", flush=True)
    print(f"  raw per-seed residual : {per_r}", flush=True)


if __name__ == "__main__":
    main()
