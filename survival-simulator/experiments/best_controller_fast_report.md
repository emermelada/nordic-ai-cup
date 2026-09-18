# CPU cost of the shipped controller — profile, lean twin, and what is actually worth doing

**Artifact:** `experiments/best_controller_fast.py` — behaviour-identical twin of
`experiments/best_controller.py`. Nothing else in the repo was modified.
Scratch harnesses (all read-only, all in `experiments/`): `_prof_fast.py`, `_sections.py`,
`_obsdist.py`, `_verify_fast.py`, `_identity_full.py`, `_time_fast.py`, `_bench_ab.py`,
`_ab_run.py` + `_ab_episode.py`, `_breakdown.py`, `_hashseed_check.py`, `_diag_identity.py`.

## 1. Headline

| measurement | shipped | `best_controller_fast` | delta |
|---|---|---|---|
| policy call, isolated (`potential_controller`, best-of-3, 62–78 k calls/seed, 3 seeds) | **8.0 µs** | **7.4 µs** | 1.08–1.11× (−8 %) |
| **served entry** `best_controller(state)` — what `agent_server.py` calls per agent | **26.3 µs** | **8.7 µs** | **3.0× (−67 %)** |
| whole-tick `ms/tick` (bench_std definition: 1000·wall/total_ticks, horizon 16000, train seeds 100/200/300, fresh processes) | 4.616 | 4.636 | indistinguishable (noise) |

The per-request numbers above are microseconds, and this is the whole story: **the policy is a
rounding error next to the endpoint**. The endpoint's measured 1.67–4.09 ms/request on the
production hosts is dominated by request parsing, the per-request log write and HTTP, not by
`potential_controller`; and the earlier "59 ms/tick" figure was `bench_std.py`'s *whole multi-agent
simulation step* (network/sim), not policy CPU. Measured breakdown of one local tick
(seed 100, 5 agents, `_breakdown.py`):

* whole tick 4.31 ms
* `core.step` + env (sim) ≈ 95 % of it
* policy: 11 calls/tick × 9.6 µs = **0.11 ms/tick ≈ 2.5 %**
* served path *with* params re-read: 11 × 25 µs = 0.28 ms/tick ≈ 6 %

So there is no CPU lever on the score. What is left is small but genuine, and it is per-request
CPU in the container, where 0.1 ms is not free.

## 2. Where the time actually goes (shipped `potential_controller`, 65,809 real per-agent states, seed 100)

| section | µs/call | share |
|---|---|---|
| wall/edge repulsion (`math.hypot` over both endpoints of every Edge obs) | 3.79 | **41.4 %** |
| observation re-filtering (4 × list comprehension + a 5th tree rescan) | 1.82 | **19.8 %** |
| bookkeeping (`_maybe_new_episode`, `_GC`, `_mem`, `_global_alive`) | 0.87 | 9.5 % |
| wander + vector combine + steering hysteresis | 0.61 | 6.6 % |
| move-dist / repro gates / return | 0.47 | 5.1 % |
| tree attraction pass | 0.20 | 2.2 % |
| fruit attraction + O(n_fruit × n_pred) risk loop | 0.19 | 2.0 % |
| predator repulsion loop | 0.14 | 1.5 % |
| predator "facing" cone loop | 0.13 | 1.5 % |
| agent dispersion loop | 0.08 | 0.9 % |
| *(perf_counter instrumentation tax)* | 0.87 | 9.5 % |

Ground truth per call (uninstrumented, best-of-3): **8.0 µs**. Confirmed independently by
`cProfile`: 4.16 M `dict.get` calls (60 % of the get-calls are the redundant observation rescans),
1.09 M `math.hypot` calls.

Observation sizes actually seen (seed 100, 60,788 calls): **13.3 Edge / 1.15 Tree / 0.57 Fruit /
0.24 Agent / 0.03 Predator** obs per call, list length 15.3, **max 68 Edges** in one state.
(The briefing's "~100 Edge entries per observation list" is ~5–7× too high on these seeds, which is
part of why the edge loop is 41 % of a small number rather than 41 % of a large one.)

## 3. Behaviour identity — verified, with one caveat about the harness

**Verified exact (controller-isolated, paired, no sim non-determinism):**

* **223,880 calls over 3 full-length episodes** (seeds 100/200/300, horizon 16000): the shipped
  policy ran the episode and recorded the exact per-call state sequence; `best_controller_fast` was
  then replayed over that same sequence in a fresh process. **0 / 223,880 mismatches**, compared
  with `repr` equality (so even `-0.0` vs `0.0` would have been caught). `_identity_full.py`.
* **212,717 calls over 4 seeds** (100/200/300/700, horizon 4000), for both the raw
  `potential_controller` entry and the served `best_controller(state)` entry: **0 mismatches**.
  `_verify_fast.py`.
* Fresh-process episodes, seeds 100/200/300 (horizon 16000): **survival ticks, score, spawns,
  fruits eaten, predated and final agent count are all exactly equal** (e.g. seed 100:
  5846 / 611.01 / 90 / 799 / 95; seed 200: 9736 / 986.87 / 197 / 1312 / 202; seed 300:
  12245 / 1203.57 / 212 / 1967 / 217).
* Params are identical: `_load_params()` values and `DEFAULT_PARAMS` compare equal.

**Caveat (pre-existing, not caused by the rewrite):** the *per-tick action digest* is **not** a
usable identity signal in this repo. The simulator itself is not action-reproducible across
processes: three fresh-process runs of the **same fast policy** on seed 100 gave digests
`8a453d8c, 8a453d8c, 8496cfb0` at identical ticks (5846), and the shipped policy produced
`8496cfb0` in one batch and `8a453d8c` in another. Two consecutive runs of the shipped policy in one
process also differ at action level (`_diag_identity.py` Q1: `actions=False`, same ticks), while
`PYTHONHASHSEED` is *not* the cause (`_hashseed_check.py`). Fixing that is a separate job — worth
flagging to the parent because every action-level sweep currently carries unexplained variance.
The digest mismatches seen for seeds 200/300 in `_ab_run.py` are this noise, not the rewrite.

## 4. What the lean file changes (all strictly behaviour-preserving)

1. **One** pass over `observations` filling five buckets, instead of 4 comprehensions + a 5th
   comprehension rescanning the list for trees. Same order inside every bucket.
2. Edge/wall pass no longer builds `pts = [starts] + [ends]` (2 list allocations, 2 dict lookups per
   edge) and skips `math.hypot` for points that cannot satisfy `d < 90` (hypot ≥ max(|x|,|y|)); every
   surviving point still computes the same `math.hypot` and is tested with the original predicate.
   The two-part summation order (all starts, then all ends) is kept verbatim because float addition
   is not associative.
3. Fruit-risk hoists the predator-only terms (`1 − d/(danger·1.4)`, wrapped bearing) into one list
   built in predator order, and keeps the multiplication order `(1−angdiff)·penalty·scale`.
4. Pure-read hoists out of inner loops (`P["danger_dist"]`, `P.get("use_*")`, `math.cos/sin/atan2`);
   `_mem` avoids rebuilding a dict literal per call; `_wrap`/`_det_rand` keep their exact formulas.
5. **`_load_params()` memoised on (mtime, size)** — the shipped loader opens and parses
   `best_controller/params.json` on *every* call, and the served path calls it once per agent per
   request (17 µs of the 26 µs shipped served cost, measured). Same values, same defaults fallback.

## 5. Essentially free CPU savings to flag (no behaviour change, trivially reviewable)

1. **`_load_params()` memoisation** — already in `best_controller_fast.py`. Removes one file
   open+JSON parse per agent per request: **26.3 µs → 8.7 µs** per policy call, i.e. ~0.1–0.2 ms per
   5–15-agent request. Biggest free win available; needs the parent's sign-off to land in the
   deployed path.
2. **`agent_server.py` re-resolves the policy on every agent, every request**
   (`_policy = _bc.best_controller(state)` → module-level `_load_params()`). Binding once to
   `_bc.make_policy(_bc._load_params())` would kill the rest of that overhead. Touching
   `agent_server.py` is outside my mandate — flagging only.
3. **`_log_request()` writes a JSON line to disk on every request** (`json.dumps` + open/append/
   close, plus a per-observation counter loop). In a container that is a syscall-heavy per-tick
   cost, plausibly larger than the entire policy. It is deliberate grader instrumentation, so the
   trade-off is the parent's call.
4. **`state = _as_dict(agent)` (pydantic `model_dump`) per agent per request.** The controller could
   read attributes directly; that is a bigger, less trivially-reviewable change and only matters if
   the endpoint ever becomes CPU-bound again.

## 6. Verdict

* `best_controller_fast.py` is **provably behaviour-identical** where proof is possible
  (223,880 + 212,717 paired calls, 0 mismatches; exact ticks/score/spawns/fruits/predated on 3 seeds
  in fresh processes) and **leaves it in place**, as instructed.
* It cuts the **policy call by ~8 %** and the **served policy path by 3×** (26.3 → 8.7 µs).
* It does **not** move whole-tick `ms/tick` (4.62 vs 4.64, i.e. inside noise) — the policy is ~2.5 %
  of a tick and ~2–6 % of a request, so no CPU-side score ceiling exists to raise. The premise
  "halving per-request CPU raises the score ceiling" is falsified: there is only ~2–6 % to halve at
  all, and the observed 59 ms/tick was sim + network RTT.
* Keep the file, apply (1) if the parent wants the 3× on the served path, skip further
  micro-optimisation.
