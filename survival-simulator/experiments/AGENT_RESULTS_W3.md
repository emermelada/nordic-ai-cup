# W3 — ACCESS-arm paired sweep (memory-search at matched energy price)

**Agent:** W3 (access arms) · **Owner container lane:** nac-A1 + nac-A2 · **Verdict: LOSS — no access win.**

## What was tested

The fleet dies **ACCESS-limited**, not energy-limited (world yields 55,000–57,000 energy/episode;
one surviving lineage costs 3,800; agents are blind 76–87% of ticks with fruit standing in the world).
A previous `memory_search` trial was **mispriced at 2× energy/tick**, so it actually tested
"search is twice as expensive", not "search finds more". These three arms re-price the same
mechanic honestly: `commit_speed_frac` is set to the baseline's `walk_frac` (0.4932), so committing
to a remembered target costs **the same energy/tick as ordinary walking**. Nothing else changes.

| arm | params file | change vs deployed H1 |
|---|---|---|
| `A1_mem`    | `wA1_mem_matched.json` | `memory_search=1.0`, `commit_speed_frac=0.4932` (== `walk_frac`) |
| `A2_follow` | `wA2_mem_follow.json`  | A1 + `follow_agent_weight=0.6` |
| `A3_ars`    | `wA3_mem_ars06.json`   | A1 + `ars_speed_frac=0.6` |

Reference: `h1_baseline_params.json` (verified identical to the deployed H1 params).
All three arms were run **as-is** — no parameter was edited. Harness: `experiments/w_hybrid.py`,
4 arms × 10 seeds, horizon 18,000 (score = ticks/10, so 18,000 ticks ⇒ score 1,800 ceiling).

## Setup

- Containers: `nac-A1` (seeds **1200–1209**), `nac-A2` (seeds **1210–1219**), each `--cpus=1.5`, `nice -n 19`.
- Launched 2026-09-18 20:52 UTC (lane taken only after `nac-E1`/`nac-E2` exited; `nac-survival-vps` untouched).
- Both containers exited 0; **20 held-out seeds** pooled, every arm paired against H1 on the identical seeds.
- Lane protocol followed: `LANES.md` read and claim line appended **before** launch (max 2 lanes respected).

## Results — pooled over all 20 seeds (paired vs H1 deployed reference)

| arm | mean ticks | median | min | Δ mean vs H1 | % | seeds won |
|---|---|---|---|---|---|---|
| **H1 (deployed, reference)** | **8,457.2** | **8,422** | **4,720** | — | — | — |
| `A3_ars` (best arm) | 5,818.6 | 5,850 | 646 | **−2,638.7** | −31.2% | **6 / 20** |
| `A1_mem` | 5,795.6 | 6,064 | 640 | −2,661.7 | −31.5% | 4 / 20 |
| `A2_follow` | 4,297.8 | 4,233 | 447 | −4,159.4 | −49.2% | 2 / 20 |

Per-lane breakdown (used for the pooled figures above):

| lane | seeds | H1 mean | A1_mem | A2_follow | A3_ars |
|---|---|---|---|---|---|
| nac-A1 | 1200–1209 | 8,639.8 | 4,803.2 | 4,337.1 | 5,063.2 |
| nac-A2 | 1210–1219 | 8,274.7 | 6,788.0 | 4,258.5 | 6,574.0 |

Fruit eaten tracks ticks (H1 1,217–1,232 fruit/episode vs 457–952 for the arms), confirming the
arms lose **income**, not just survival time — they forage *worse*, not merely differently.

## Interpretation

- Even at **matched energy per tick**, `memory_search` does **not** buy access. The best arm still
  loses 31% of H1's survival time and wins only 6 of 20 paired seeds.
- The mechanism is not a pricing artifact: A3_ars > A1_mem > A2_follow, and adding follower-pull
  (`follow_agent_weight=0.6`) is actively destructive (2/20 seeds won, −49%).
- This is the first **fair** test of the access hypothesis, and it is **negative**. Memory-search /
  commit-to-target is not the access lever. A genuine income-side gain remains unmeasured in this project.
- Consistent with the death-mode finding: the fleet is blind, but *searching the blind region* at
  matched cost is not the fix — the remembered-target approach spends real time looking and still
  earns less than the deployed potential field.

## Raw evidence

| artifact | path |
|---|---|
| A1 lane full docker log | `experiments/_logs/wA_nac-A1.log` |
| A2 lane full docker log | `experiments/_logs/wA_nac-A2.log` |
| Parsed per-seed stats | `experiments/_logs/wA_stats.json` |
| VPS link (image `nac-eval`) | `root@212.147.239.222`, dir `/opt/nac-phase-test/` |

Log hashes: `wA_nac-A1.log` c6a9559d… · `wA_nac-A2.log` bfb38042…

No deployment performed. No candidate produced. `best_controller.py` and all params files untouched.
