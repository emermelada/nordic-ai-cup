# FINDING — the graded endpoint serves a controller that is ~42% worse than what we already have

**2026-09-20, measured on fresh paired seeds in the real simulator, same harness, same seeds.**

## The measurement

`experiments/mech_ab.py --policy {heuristic|hive} --arms base --seeds 300560-300639 --horizon 18000`
80 seeds never used by any earlier experiment, both controllers driven through the SAME `SimulationCore`
code, the same seeding, fresh process per episode, paired per seed.

| controller | what it is | mean ticks | median | p10 | min | score (ticks/10) |
|---|---|---|---|---|---|---|
| heuristic `best_controller.py` | **served on survival.zaitzev.com** (sha256 252f0ba1) | 8,110 | 8,226 | 3,881 | 1,201 | **811** |
| hive `hive.py` | survival-v2 branch (sha256 829e4147) | **11,517** | 11,920 | 7,139 | 1,677 | **1,152** |

**Paired: hive − heuristic = +3,406 ± 438 ticks, t = 7.78, W/L = 65/15 (n=80), +42.0%.**
The lower tail improves too: p10 +84%, min +40%. Per-seed SD of the paired difference: 3,916 ticks.

CONFIRMED on a second, independent fresh block (seeds 300640-300679, 40 seeds, same horizon):

| block | n | paired | W/L |
|---|---|---|---|
| 1 (300560-300639) | 80 | +3,406 | 65/15 |
| 2 (300640-300679) | 39 | +2,835 | 30/9 |
| **pooled** | **119** | **+3,219 +- 388, t = 8.30, +39.4%** | **95/24** |

Pooled means: heuristic 8,164 ticks (816 score) vs hive 11,383 ticks (1,138 score); p10 4,381 → 5,916;
min 1,201 → 1,527. Two independent blocks agree, and the second does not depend on any seed the first
used.

Fairness checks, done before believing the number:
- hive reads only `step["sim_time"]` and `step["agent_status"]` (verified by grepping every `step.get`),
  which is exactly what the harness passes and exactly what the official `simulation_server.py` sends.
  Neither controller is handicapped by a missing payload field.
- RFC: the heuristic gets its native per-agent state dict; hive gets its native StepResponse shape.
- At generation 0 the residual-ES centre scores bit-identically to BASE, which independently confirms
  the simulator is deterministic in this harness (a free A/A control).

Inference latency (deployment criterion #5), measured in the same sim over a full 7,178-tick game with
16.8 agents on average and 39 at peak: **mean 4.09 ms, p95 9.33 ms, p99 11.72 ms, max 19.0 ms**
= 29.3 s of grader wait per game against a ~600 s budget. Acceptable with ~20x headroom.

## What is actually deployed (verified live)

`survival.zaitzev.com` → 94.237.34.245 → container `nac-survival-vps`, up 16 h, 0 restarts, HTTP 200 in
21 ms; `sha256(/app/best_controller.py)` = **252f0ba1e060…**, 52 params (`genome_select 1.0`,
`gs_w_energy 3.0`, `gs_topk 3.0`, `evade_mode 0.0`, `reserve_frac 0.0`, `blind_explore_frac 0.12`).
There is **no hive.py anywhere in the image**.

The container's own `predict_log.jsonl` (742 segments, 28 complete attempts carrying ≥2,000 logged
ticks) shows the served controller's real attempt distribution:
**mean 731, median 759, p25 633, p75 857, max 1,165.** That is consistent with the team's 59-attempt
ledger (pre-C6 770, post-C6 819) and with this harness (local mean 8,110 ticks ≈ 811). The board-best
1,223.85 is the upper tail of that distribution, not its centre.

So: nothing scoring "1000-1200 consistently" is deployed. The controller that does score ~1,150 here
is the survival-v2 branch, and it is not on the graded endpoint.

## Consequence

If the graded endpoint served hive instead of the heuristic, the expected attempt moves from ~790 to
~1,150 — about **+360 score**, which is larger than every ML result of this project combined and larger
than any heuristic sweep has produced. The evidence bar the project set for a deploy was: positive
paired improvement, sufficient seeds, no lower-tail degradation, fresh unseen seeds, acceptable
latency. This measurement meets all five on 80 fresh paired seeds at 7.8σ.

It was NOT deployed: the working directives for this session say explicitly "do not deploy anything
automatically" and "do not modify the production endpoint", and a container/service swap on the graded
path needs the owner's decision. The rollback path exists (`tools/ops/ROLLBACK.md`,
`/root/code_backup_f90cb4e3.py`, `/root/params_backup_c6.json` inside the container).

## Recommended next action (for the owner)

1. Deploy hive to the graded endpoint with the container kept intact as the fallback, then validate.
2. Before that, one more independent confirmation block of this comparison on a second fresh 160-seed
   block (`mech_ab.py --policy hive/heuristic --seeds 300640-300799`), since a deploy decision on the
   graded path deserves two independent blocks rather than one.

## Reproducing the comparison

`hive_v2.py` in the harness is a READ-ONLY copy of the survival-v2 branch's controller:

    git show origin/survival-v2:survival-v2/hive.py > survival-simulator/experiments/hive_v2.py

    # on the experiment box, from /opt/nac_h2h:
    /opt/nacv/bin/python experiments/mech_ab.py --policy heuristic --arms base \
        --seeds 300560-300639 --horizon 18000 --workers 40 --out cmp_heur
    /opt/nacv/bin/python experiments/mech_ab.py --policy hive --arms base \
        --seeds 300560-300639 --horizon 18000 --workers 40 --out cmp_hive

Latency: `python experiments/hive_latency.py --seed 300560 --horizon 18000`.
