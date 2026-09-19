#!/usr/bin/env python3
"""Calibration analysis for the autopilot: noise floor, M_no_tree replication, board risk.
Read-only: reads /opt/nac/results/cache.jsonl."""
import json
import random
import statistics as st

R = {}
for line in open("/opt/nac/results/cache.jsonl"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("horizon") == 18000:
        R[(r["cand"], r["seed"])] = (r["score"], r["steps"], r.get("fruits"))


def diffs(a, b, seeds):
    out = []
    for s in seeds:
        if (a, s) in R and (b, s) in R:
            out.append(R[(a, s)][0] - R[(b, s)][0])
    return out


def rep(tag, v):
    if not v:
        print(tag, "NO DATA")
        return None
    sv = sorted(v)
    w = sum(1 for x in v if x > 1e-9)
    l = sum(1 for x in v if x < -1e-9)
    t = len(v) - w - l
    p10 = sv[max(0, int(0.1 * len(sv)) - 1)]
    print("%-34s n=%3d mean=%+8.1f sd=%7.1f med=%+8.1f p10=%+8.1f min=%+8.1f W%d/L%d/T%d win=%.0f%%"
          % (tag, len(v), st.mean(v), st.pstdev(v), st.median(v), p10, min(v), w, l, t,
             100.0 * w / (w + l) if w + l else 0))
    return v


S1 = list(range(2700, 2740))  # c6minus stage 2 (verified from cache)
S2 = list(range(2960, 3000))  # c6conf stage 1 (verified from cache)
S3 = list(range(3000, 3040))  # c6conf2 stage 1 (dose curve)
print("--- NOISE FLOOR: identical params, same run, seeds 2960-2999 (score units = ~official pts) ---")
rep("BASE_C6 vs BASE_C6_dup (A/A)", diffs("BASE_C6", "BASE_C6_dup", S2))
rep("BASE (==C6) vs BASE_C6", diffs("BASE", "BASE_C6", S2))
print("--- NOISE FLOOR: seeds 3000-3039 (c6conf2 batch, same three identical arms) ---")
rep("BASE_C6 vs BASE_C6_dup (A/A)", diffs("BASE_C6", "BASE_C6_dup", S3))
rep("BASE (==C6) vs BASE_C6", diffs("BASE", "BASE_C6", S3))
print("--- BEST CANDIDATE: M_no_tree (tree_weight 0.25 -> 0.0) ---")
a = rep("vs C6  seeds 2960-2999", diffs("M_no_tree", "BASE_C6", S2))
b = rep("vs old seeds 2700-2739", diffs("M_no_tree", "BASE", S1))
c = rep("vs C6  seeds 3000-3039", diffs("M_no_tree", "BASE_C6", S3))
if a and b and c:
    v = a + b + c
    rep("POOLED 120 seeds", v)
    rng = random.Random(0)
    tot = 20000
    loss = sum(1 for _ in range(tot) if sum(rng.choice(v) for _ in range(3)) < 0)
    print("P(net loss on a 3-seed board validation) = %.1f%%   (mean gain %.1f board pts)"
          % (100.0 * loss / tot, st.mean(v)))
print("--- dose curve + deletions vs C6, seeds 3000-3039 (c6conf2) ---")
for arm in ("M_no_tree", "M_tree01", "M_tree05", "M_tree_neg"):
    rep(arm, diffs(arm, "BASE_C6", S3))
print("--- deletions vs C6, seeds 2960-2999 (c6conf) ---")
for arm in ("M_no_disperse", "M_no_flee"):
    rep(arm, diffs(arm, "BASE_C6", S2))
