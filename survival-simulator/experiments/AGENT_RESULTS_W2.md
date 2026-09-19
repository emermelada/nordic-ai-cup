# W2 — STATE-TRIGGERED THIN-RELAY ENDGAME: MEASURED NEGATIVE (do not deploy)

**Date:** 2026-09-18 · **Owner:** W2 (thin relay) · **Verdict: REFUTED as implemented.** No arm cleared
the bar (beat H1 by >500 ticks on >=15/20 paired seeds). Best arm won **7/20**. Keep the knobs
off-by-default; the mechanism stays inert under the deployed parameter set.

---

## 1. What was built (canonical `survival-simulator/best_controller.py`, +103 lines)

A **state-triggered thin-relay endgame**, entirely off-by-default (`thin_relay = 0.0`), with every new
expression guarded so that with the knob off the original value is returned unchanged (no re-derived
angles, no re-normalised vectors — a float-path leak would have been fatal in this simulator).

| param | default | meaning |
|---|---|---|
| `thin_relay` | 0.0 | 0 = off (byte-identical behaviour); >0 enables the block |
| `thin_trig_tick` | 6000.0 | (a) TIME gate, in reconstructed sim ticks via the **existing** `_SIM_TICK` estimator built from per-agent `age` + per-agent call counts (no second time source) |
| `thin_trig_vis` | 0.35 | (b) STATE gate: fleet EMA of **fruits-visible-per-agent** falls below this |
| `thin_trig_vis_off` | 0.70 | hysteresis release threshold — the latch cannot flap tick-to-tick |
| `thin_target_pop` | 2.0 | fleet size the relay keeps (settles at target..target+1 = 2–3 agents) |
| `thin_spawn_energy_abs` | 250.0 | absolute parent-energy gate ⇒ parent keeps >=150, clear of the sprint lockout at `max_energy/5` (`environment.py:512`) |
| `thin_rescue_energy_abs` | 160.0 | emergency gate when the fleet is down to its LAST agent (spawn even at the cost of the parent's sprint) |
| `thin_spawn_cooldown` | 200 | ticks between thin-relay spawns |
| `thin_move_frac` | 0.7 | movement scale while active, applied as `dist = max(speed*0.2, dist*frac)` — conserve but **the floor guarantees the agent never stops**, because movement is the income mechanism |

Reproduction is **never** blocked while the mode is active (`pop_ok = crowd_ok = True`); the population
target is enforced only as a *room* test (`gpop <= target`), and when only one agent is left the relay
fires unconditionally above `thin_rescue_energy_abs`. This was deliberate: the four refuted `bank late`
arms died because a population gate stopped ALL breeding and the age-relay broke.

### Identity check with the knob OFF (hard requirement)
Pristine pre-edit controller (`md5 0349bb93…`, byte-identical to the copy the serving box had) vs the
edited file, **fresh process per module**, identical seed sequence, `horizon 6000`:

| seeds | pristine `steps / fruits / spawns / predated` | edited | equal |
|---|---|---|---|
| 101 | 6000 / 974 / 129 / 129 | 6000 / 974 / 129 / 129 | ✅ |
| 102 | 6000 / 1131 / 134 / 135 | 6000 / 1131 / 134 / 135 | ✅ |
| 103 | 6000 / 990 / 143 / 134 | 6000 / 990 / 143 / 134 | ✅ |
| 104 | 6000 / 1172 / 122 / 117 | 6000 / 1172 / 122 / 117 | ✅ |

Also reproduced **interleaved inside one process** (`orig,new,orig,new`) on seeds 103 and 104: 4/4
identical. **Identity is verified.** (One first-pass observation in a *different* in-process sequence
order showed seed 102 at 5592 vs 6000; that reproduced in no controlled comparison and is the harness
state-sensitivity quantified in §4, not a code difference — see the caveat, it matters for reading any
paired sweep on this box.)

## 2. Configs (full H1 params + deltas, emitted with `json.dump` from `mk_wB_arms.py`)

| arm | `thin_trig_tick` | `thin_trig_vis` | `thin_target_pop` | spawn abs / rescue | cd | move | rationale |
|---|---|---|---|---|---|---|---|
| `wB0_ref_h1.json` | — (`thin_relay=0`) | — | — | — | — | — | **control**: the EXACT H1 params, placed FIRST among the candidates, so it is measured in the same process as the arms |
| `wB1_thin_c3000_v55_t2_m70.json` | 3000 | 0.55 | 2 | 250/160 | 200 | 0.70 | engages well before the cascade |
| `wB2_thin_c4500_v45_t2_m75.json` | 4500 | 0.45 | 2 | 250/160 | 250 | 0.75 | engages right at the observed mean death time |
| `wB3_thin_c6000_v35_t1_m85.json` | 6000 | 0.35 | 1 | 220/160 | 150 | 0.85 | the DEFAULT window verbatim (expected largely unreachable) |
| `wB4_thin_c3000_v55_t6_m80.json` | 3000 | 0.55 | 6 | 200/150 | 200 | 0.80 | **dose–response**: same trigger as wB1, mild thinning |

## 3. Paired results — 20 held-out seeds 1300–1319, horizon 18,000, `nac-B1`/`nac-B2`

Reference = `B0_ref` (identical params, same process). `nac-B1` was **SIGKILLed (exit 137) during its
trailing LIVE arm**, so `nac-B1` has no LIVE line; `wB0_ref` is the same parameter vector placed first,
i.e. a strictly better paired control than the trailing LIVE.

| proc:arm | mean | median | min | Δ vs wB0_ref (mean) | Δ median | seeds >+500 (win bar) | seeds <-500 |
|---|---|---|---|---|---|---|---|
| `B1:wB1` (target 2) | 6642.7 | 6798.5 | 812 | **−413.4** | −565.5 | 7/20 | 10/20 |
| `B1:wB2` (target 2, late) | 6554.6 | 6359.5 | 812 | **−501.5** | −148.0 | 5/20 | 9/20 |
| `B1:wB3` (target 1) | 6730.9 | 6437.0 | 812 | **−325.1** | 0.0 | 2/20 | 4/20 |
| `B2:wB1` (replicate) | 6495.4 | 6711.5 | 812 | **−468.4** | −262.5 | 7/20 | 9/20 |
| `B2:wB4` (target 6) | 7004.4 | 6816.0 | 812 | **+40.6** | +31.0 | 7/20 | 8/20 |
| `B1:wB0_ref` | 7056.1 | — | — | — | — | — | — |
| `B2:wB0_ref` | 6963.8 | — | — | — | — | — | — |
| `B2:LIVE` (same params as wB0_ref) | 6830.2 | 6255.5 | 812 | −133.6 | 0.0 | 1/20 | 1/20 |

**Bar: >+500 ticks on >=15/20 seeds. Achieved by no arm** (best 7/20, i.e. *below the 10/20 you would
expect by chance*). Every thinning arm is also negative on the mean.

### Raw per-seed deltas vs `wB0_ref` (seed order 1300…1319)
```
B2:wB1  [    0,  -862, +2360, -4230,  -443, +1566, +1695, +1202, -5942, -1431,
           -809, -2987, +2087,   +77, +4591,   -82, +1943, -2164, -4935, -1003]
B2:wB4  [    0, -1953, +2286, -1778,  -895,  +731,  +871,  -276, -4551, -2061,
           +318,  -536, +4627,   +62, +1555,  -721,  +191, +2947,  -629,  +625]
B1:wB1  mean −413.4 · B1:wB2 mean −501.5 · B1:wB3 mean −325.1  (per-seed vectors in /tmp/wB/wB_final.json)
```
Per-arm per-seed survival (`ticks`, seed 1300→1319, from the harness log):
```
wB0_ref(B1) [ 812, 6374, 5363, 8863, 7076, 5816, 10879, 4498, 8892, 9367, 5389, 9777, 5921, 5830, 6137, 6889, 5868, 8971, 13051, 5348]
wB1(B1)     [ 812, 5512, 7723, 4633, 6633, 7382, 10191, 5700, 10777, 7936, 4580, 6790, 8008, 4778, 10728, 6807, 7811, 6807,  5606, 3640]
wB2(B1)     [ 812, 5569, 9114, 5426, 5335, 5661,  6795, 4498,  9051, 6595, 8366, 5427, 6296, 6660,  7742, 5538, 5727, 6423, 11368, 8688]
wB3(B1)     [ 812, 6197, 5363, 8422, 6677, 5816, 10468, 4498,  7615, 9809, 5389, 6854, 5921, 5830,  7785, 8765, 5868, 7409,  9772, 5348]
wB0_ref(B2) [ 812, 6374, 5363, 8863, 7076, 5816,  8496, 4498, 11940, 9367, 5389, 9777, 5921, 5830,  6137, 6889, 5868, 8971, 10541, 5348]
wB1(B2)     [ 812, 5512, 7723, 4633, 6633, 7382, 10191, 5700,  5998, 7936, 4580, 6790, 8008, 5907, 10728, 6807, 7811, 6807,  5606, 4345]
wB4(B2)     [ 812, 4421, 7649, 7085, 6181, 6547,  9367, 4222,  7389, 7306, 5707, 9241,10548, 5892,  7692, 6168, 6059,11918,  9912, 5973]
LIVE(B2)    [ 812, 6374, 5363, 8863, 7076, 5816, 10879, 4498,  6885, 9367, 5389, 9777, 5921, 5830,  6137, 6889, 5868, 8971, 10541, 5348]
```
Fruits eaten: `wB0_ref` 1137.5 / 1120.0 vs `wB1` 946.8 / 933.3, `wB2` 992.4, `wB3` 1068.2, `wB4` 1038.8.
**Every thinning arm eats less than the control** — the mechanism reduces income.

## 4. Why: the mechanism engaged, and the harm scales with the dose

* **It armed.** Local instrumented runs (`_wb_smoke.py`, trigger + EMA sampled inside the policy):
  `wB1` armed **5,498 ticks** on seed 1301 (fleet trace 14 → 7 → 5 → 3 across ticks 3,250–4,000);
  `wB3` armed 1,956 ticks; `wB2` 971. So this is a genuine null on an engaged mechanism, not an
  unreachable window — the failure mode that killed the four earlier `bank late` arms. (Seed 1300 dies
  at tick 812 for every arm: an unwinnable seed, which is also why its delta is 0 in every row.)
* **Same seed, direct comparison** (seed 1301, H1 vs `wB1`): H1 holds 4–11 agents through ticks
  4,000–6,250 and dies at 6,498; `wB1` is driven down to 3 agents by tick 3,750 and dies at 4,055 —
  **−2,443 ticks**.
* **Dose–response, same trigger, same process:** thinning target 2 → **−468**; thinning target 6 →
  **+41** (neutral). The harm is monotone in how hard the fleet is thinned, which locates the cause in
  the fleet-size reduction itself.
* **Mechanism of harm:** income in this sim is ACCESS-limited (agents blind 76–87% of ticks). The
  fleet's search capacity scales with the number of searchers, while the metabolic saving per removed
  agent is only 0.1/tick. Thinning therefore destroys more income than it saves — visible directly in
  the fruits column (946.8 vs 1137.5). A tiny fleet also removes the redundancy that absorbs a
  predation hit; the labelled premise ("the target is not energy-limited") is true, but it does not
  follow that fewer agents are better — they are the *income instrument*, not just a cost.

### Harness caveat found while doing this (affects every paired sweep on this box)
`wB0_ref` and `LIVE` are the **same parameters**, evaluated in the **same process**, and they disagree
on 3/20 seeds by up to **−5,055 / +2,383** ticks (mean +133.6, median 0.0). Two containers running the
identical `wB0_ref` line disagree on 3/20 seeds (mean +92.2, min −3,048, max +2,510). So: per-seed
deltas under ~±2,600 are noise-dominated, ~3/20 seeds are chaotic, and only the 20-seed mean is
interpretable (SE ≈ 220 ticks). Any single-seed claim on this harness — including any arm that "wins"
one seed — should be treated as unmeasured. This is why the sweep used a same-process control
(`wB0_ref`) rather than relying on the trailing LIVE alone.

## 5. Verdict

**Negative, and it is not a missing-trigger artefact.** The thin-relay endgame *armed* on the seeds
where it mattered, kept a 2–3 agent chain alive, and still lost: −325 to −500 ticks on the mean, no
seed-won bar progress, fewer fruits eaten, and a dose–response that implicates the thinning itself.
The prior from this session's own numbers was wrong in its conclusion: the endgame is not
energy-limited *for the fleet*, but the fleet is the instrument that converts the world's 55k–57k
energy into score, and shrinking it shrinks the conversion. Recommendation: keep `thin_relay = 0.0`
(already the default, byte-identical behaviour verified) and **do not deploy**; the next lever worth
trying is the opposite direction — INCREASING search capacity in the endgame (more searchers, or better
access), not reducing burn.

Artifacts: `best_controller.py` (off-by-default knobs), `experiments/mk_wB_arms.py`,
`experiments/wB0_ref_h1.json`, `wB1_thin_c3000_v55_t2_m70.json`, `wB2_thin_c4500_v45_t2_m75.json`,
`wB3_thin_c6000_v35_t1_m85.json`, `wB4_thin_c3000_v55_t6_m80.json`, `experiments/_wb_ident2.py`
(identity, fresh process per module), `experiments/_wb_smoke.py` (engagement instrumentation),
raw logs + parsers in `experiments/wb_evidence/` (`B1.log`, `B2.log`, `wB_final.json`).

**Concurrent-edit note:** at 00:32, while this sweep was running, another lane added its own
off-by-default `genome_select` family to the same canonical `best_controller.py`. The thin-relay block
was unaffected (both families are additive and gated on their own knob, which no wB* config sets), and
the merged file was re-checked for identity-with-the-knob-off against the pristine pre-edit controller
(see `experiments/wb_evidence/`). The measurement below was produced with the thin-relay edit only
(the copy the sweep containers imported).
