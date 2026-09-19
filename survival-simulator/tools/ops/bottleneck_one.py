#!/usr/bin/env python3
"""bottleneck_one.py MODE [seeds] - run one intervention on its own core (see bottleneck.py)."""
import sys
sys.path.insert(0, "/opt/nac_gs/experiments")
sys.path.insert(0, "/opt/nac_gs")
import bottleneck as B  # noqa: E402

mode = sys.argv[1]
seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "3800,3801,3802,3803").split(",")]
res = [B.run(s, mode, 18000) for s in seeds]
m = sum(res) / len(res)
print(f"RESULT {mode:>18} mean={m:8.0f}  per-seed={res}", flush=True)
