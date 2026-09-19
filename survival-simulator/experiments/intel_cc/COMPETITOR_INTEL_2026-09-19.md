# Competitor intel — tested against OUR simulator (2026-09-19)

Source: unverified screenshot of another team's experiment. Treated as hypotheses, not ground truth.
Everything below is measured in this repo, on the **live artifact** (`served_controller_252f0ba1.py` +
`served_params_252f0ba1.json`, i.e. the controller serving today, gs en_top3, official 1,075.28).

Tools (all in `experiments/intel_cc/`):
* `oracle_percept.py` — the oracle-vs-perception experiment + camping/economics diagnostics
* `oracle_percept_report.py` — paired verdicts from its JSON
* `food_economics.py` — break-even travel economics, constants read from the sim source
* `gs_diversity.py` — breeding-gate diversity measurement
* results: `oracle_x86_700.json`, `oracle_mac_700.json`, `gs_div_x86.json`, `noage.log`

---

## Verdicts in one line each

| competitor claim | verdict in our sim |
|---|---|
| agents camp/starve while ripe fruit stands farther away, because of a too-small food-search radius | **NO camping, and there is no search radius to widen.** The radius is the *sim's sensor*, hard-capped at 400 px. Agents move away from a visible fruit in **0.5%** of ticks. |
| travelling farther for fruit can still net positive | **TRUE and quantified — up to 333–1000 units of walking, i.e. 1.7–5x our 200 px vision.** But **sprinting** farther is net-negative (break-even 71–214). |
| perfect food knowledge still dies ⇒ decision/energy, not perception, is the bottleneck | **Replicates, and the mechanism is a real defect in our controller** — but their inference is confounded by the aging/breeding economy, which is the actual binder here. |
| genome selection can collapse diversity around one lucky mutant | **We cannot suffer their exact failure** (re-ranked fraction gate + guards), and measured diversity does **not** collapse — but our gate does cut births and genome spread by **~39%**, the same mechanism. Their proposed fix is already implemented and buys nothing. |

---

## 0. Operating conditions (please read)

A **parallel agent is working this same repo and this same machine right now**. During this session it:
* live-edited `best_controller.py` twice (hash moved `d6e2041e…` → `7c69e317…` between two of my commands);
* renamed my `experiments/gs_diversity.py` to `gs_diversity.py.PARKED` (repo-hygiene "parking");
* SIGKILLed my measurement processes **three times on the Mac** and **once on the experiment box 212.147.236.122** (exit `-9`). Two runs were lost mid-flight; one 16-seed arm survived only 13 seeds.

Consequence: my scripts now live in `experiments/intel_cc/` (my own subdir) and every result below is
tagged with the seeds it actually completed. **Recommendation: one owner per repo / per lane.** Two
agents editing one controller makes every hash-pinned measurement a moving target.

---

## 1. Is there a food-search-radius hypothesis in our code? — NO

Read from source, not assumed:

* **The controller has no radius.** Live params have `forage_nearest = 1.0`, and the code is
  `tf = min(fruits, key=lambda f: f["distance"])` — steer at the nearest *observed* fruit. There is no
  distance cutoff anywhere in the foraging path (`best_controller.py:669-724`).
* **The radius belongs to the simulator's sensor.** `creature.py:138-139`:
  `hearing_radius` 50 (omnidirectional) + `vision_radius` 200 inside `cone_angle` π/3 (±30°), with
  river-edge occlusion. One agent therefore surveys **20,944 px² = 1.09%** of the 1600x1200 world per tick.
* **Both are heritable genes with hard caps** (`environment.py:335-344`): hearing ≤ chunk/4 = **100**,
  vision ≤ chunk = **400**, cone ≤ π/2. The widest sensor the sim permits is
  **125,664 px² = 6.54%**, exactly **6x** the default. A policy **cannot set a gene** — it can only
  decide *who breeds*, which is what `genome_select` / `gs_w_vision` are for.

**So "90 px → 500 px" has no analogue here.** The only legal versions of "look further" are
(a) select for the vision gene up to 400 px, (b) memory/ballistic search — which this project already
refuted three times (−18% to −50%). This is why the idea must not be copied across.

## 2. Energy economics of travelling farther — the profitable horizon EXCEEDS our eyesight

Constants read from source (`environment.py:501-518`, `creature.py:25`, `fruit.py:19`):
walk **0.05**/unit · sprint **0.5**/unit on the distance beyond walk speed · metabolism **0.1**/tick ·
turn `|Δ|/(2π)` ≤ 0.5 · fruit **20 → 60** energy (grows 0.2/tick, rots at age 100 s) · biome
`move_penalty` multiplies the *displacement* (so cost per unit of ground covered is `0.05/penalty`).

Break-even walking distance, including the metabolism you pay while travelling
(`food_economics.py`, computed from the live Biome classes):

| fruit energy | forest/grassland walk | forest **sprint** | swamp walk | river walk |
|---|---|---|---|---|
| 20 (fresh) | **333** | 71 | 167 | 100 |
| 30 | **500** | 107 | 250 | 150 |
| 40 | **667** | 143 | 333 | 200 |
| 60 (ripe cap) | **1000** | 214 | 500 | 300 |

Three consequences:

1. **The gap is real**: profitable walking range 333–1000 units vs a 200-unit perceptual horizon. Past
   200 px a fruit is worth fetching and *invisible*. This is the mechanical reason the world is
   access-limited (the project's own locked-in finding: fleets die with 42–129 fruit standing uneaten).
2. **Never sprint for distance.** Sprint break-even is 71–214 units — below the vision radius. Any
   "travel farther" rule must travel at **walk** speed. (The current live config already sprints only
   when fleeing.)
3. **Opportunity cost is small, risk is not.** Measured travel needed is already inside the band
   (BASE: **263 units per fruit** on x86, 266 on macOS). The risk side is visible in the arms that
   travel more: `ORACLE_RIPE` 365 units/fruit, **deaths/1k 15.9 vs 13.1** for BASE — more time in the
   open and in the <20% sprint-lockout costs lives faster than the extra fruit pays.

## 3. Are our agents camping? — NO (measured two ways)

* `away_frac` = share of agent-ticks **with a fruit in view** where the chosen move heads *away* from it:
  **0.48%** (x86, 16 episodes) / 0.67% (macOS). The controller almost never refuses visible food.
* Independently, the existing recording survey (`PHASE3_REPORT.md`, 16 recorded x86 episodes): *"ticks
  choosing a direction away from a visible fruit: median 19 per lifetime"* (~1.5% of a lifetime).

The real condition is the opposite one: agents are **blind 84–85% of their ticks** and see **0.35–0.39
fruits** each. The constraint is that food is out of sight, never that it is ignored.
`starve+food` (ticks at <20% energy **with** a fruit in view) is 4–7% for BASE.

## 4. Oracle vs normal perception — replicating their result, and finding the real mechanism

Same controller, same seeds, same world; **only the observation is changed** (every fruit injected at its
true relative bearing) — except where flagged. A/A control (two identical configs) on x86: **+7 steps,
median 0, min −24, 1W/1L** — the harness ties exactly, so these are real effects.

**x86, 16 paired seeds, 12,000 ticks (deployed params, BASE mean 8,256):**

| arm | mean steps | paired vs BASE | W/L | blind | fruits seen | travel/fruit | income/1k | deaths/1k |
|---|---|---|---|---|---|---|---|---|
| BASE | 8,256 | 0 | — | 0.850 | 0.35 | 263.4 | 6,642 | 13.1 |
| BASE_DUP (A/A) | 8,263 | +7 | 1/1 | 0.850 | 0.35 | 263.3 | 6,637 | 13.0 |
| **ORACLE_FRUIT** | 5,327 | **−2,929** | 3/13 | 0.000 | 56.6 | 363.4 | 9,563 | 15.5 |
| ORACLE_FRUIT_SLOW (`forage_speed` 0.35) | 6,690 | −1,566 | 5/11 | 0.000 | 98.6 | 265.6 | 6,049 | 11.5 |
| ORACLE_RIPE (only fruit ≥ 40 energy) | 6,743 | −1,513 | 4/12 | 0.016 | 27.9 | 365.2 | 10,957 | 15.9 |
| **ORACLE_NOPRED** (perfect food **+ no predators**) | 7,900 | **−356** | 8/8 | 0.000 | 55.7 | 391.0 | 8,331 | 13.6 |
| **WIDE_VISION** (sensor at the gene cap 400 px / 90°) | 9,308 | **+1,052** | 9/6 | 0.720 | 0.82 | 293.4 | 7,877 | 13.3 |

macOS reproduced the same ordering and signs (BASE 7,962 · ORACLE_FRUIT −2,683 · SLOW −1,382 ·
RIPE −1,181 · NOPRED −62 (9W/6L) · WIDE_VISION +780 (10W/6L)), with a slightly noisier A/A (−37).

**What this means:**

1. **Perception is not the bottleneck.** Give every agent perfect 360° knowledge of every fruit *and*
   remove predation, and survival is **identical to normal perception** (−356, 8W/8L). Their headline
   replicates in our sim.
2. **But the loss from perfect perception is a *decision/energy defect*, and it is specific.** With food
   always visible the controller's direct-forage branch fires on **every** tick, so agents travel at
   `forage_speed = 1.0` permanently and **never enter the energy-throttled blind branch** that conserves
   them 85% of the time today. `starve+food` rises from ~5% to **25–58%**: they can see food, are at
   <20% energy, and cannot sprint out of trouble. Throttling the forage speed to 0.35 recovers about
   **half** the loss (−1,566 vs −2,929). *The blindness is load-bearing.*
3. **WIDENING SENSORS IS THE ONE POSITIVE LEVER** (and the closest legal analogue of their idea):
   at the genetic cap (vision 400, cone 90°, hearing 100 = 6x the area) survival **+1,052 steps**, blind
   85%→72%, fruits seen 2.4x, income +19%. Note the ceiling: a 6x area increase still leaves **72% of
   ticks blind** — no sensor change fixes this sim's access problem.
4. **"Travel farther for the ripe fruit" is refuted here**: `ORACLE_RIPE` is −1,513 (4W/12L) with the
   worst predation risk in the set. Ignoring unripe fruit loses.

### 4b. The confound their oracle does not control — and it is the real binder

Every agent has a hard `max_age` of **60–120 s** (`agent.py:26`), i.e. it dies of old age after
600–1,200 ticks **however well fed it is**. That makes "still died with perfect information" ambiguous
between *decision/energy* and the *age/reproduction economy*. Held off explicitly:

**x86, 16 paired seeds @12,000 (BASE mean 8,256; A/A noise +7):**

| intervention | paired vs BASE | median Δ | p10 Δ | min Δ | W/L |
|---|---|---|---|---|---|
| aging OFF only (`BASE_NOAGE`) | **+1,367** | +528 | −1,889 | −3,594 | 9/7 |
| perfect food + no predators (aging ON) | −356 | −88 | −3,889 | −5,271 | 8/8 |
| perfect food + no predators + aging OFF | +1,334 (13 seeds) | +869 | −3,642 | −6,577 | 10/3 |

**Removing aging alone is the largest single intervention in the whole experiment (+17%), and adding
perfect perception on top of it changes nothing (+1,334 ≈ +1,367).** In our simulator the binder is the
**age/reproduction economy**, not perception and not "decision" in the abstract. Their inference
"perfect knowledge still dies ⇒ decision/energy is the bottleneck" is therefore confounded: it is also
what a breeding/aging ceiling looks like.

*Honest limits:* 13–16 seeds; a heavy right tail (median +528 but p10 −1,889), so this is a real
central tendency with genuine downside. `max_age` is set by the simulator and cannot be changed by a
policy — this is diagnostics, not a deployable arm.

## 5. Can our breeding gate collapse diversity? — structure says no, and the measurement agrees

**Structure (from the live artifact).** The gate is `rank(genome utility) ≤ gs_topk` **over the living
fleet**, re-ranked **every epoch** over a TTL-pruned table, with two guards: `gs_rescue_pop` (never gate
when population ≤ 1) and `gs_min_known` (no gating while <3 genomes are known). There is **no
"within 0.4 of the best" threshold anywhere**, so their exact failure — 21 of 23 retired at t=240 — cannot
happen: the threshold is a moving rank on a re-ranked population, not a frozen bar. Deployed `gs_topk=3`.

**Measured (x86, 12 paired seeds @8,000, deployed controller+params):**

| arm | births | living | distinct genomes | distinct/living | top genome share | mean pairwise genome distance | max_energy late |
|---|---|---|---|---|---|---|---|
| GS_TOPK3 (deployed) | 87.1 | 8.6 | 4.7 | 0.540 | 0.495 | 0.0376 | 767 |
| GS_MODE2 (breed above fleet mean) | 82.3 | 8.2 | 4.7 | 0.552 | 0.546 | 0.0367 | 655 |
| GS_OFF (no selection) | 142.5 | 11.4 | 7.0 | 0.641 | 0.394 | 0.0617 | 552 |
| **paired vs GS_OFF** | **−55.4 (−39%)** | −2.8 | −2.3 | −0.10 | +0.10 | **−0.0241 (−39%)** | +215 |

* **Diversity does NOT collapse.** With the gate on, the fleet still holds **4.7 distinct genomes among
  8.6 living agents**; the commonest genome holds **half** the fleet, not all of it. The colony keeps
  exploring.
* **But their mechanism is present, attenuated**: the gate cuts **births by 39%** and **genome spread by
  39%**. That is the same "fewer births ⇒ less exploration" force they describe, just not fatal here.
* **Their proposed fix is already implemented and does not help.** "Compare against the 75th percentile
  instead of the single best" ≈ our `gs_mode=2` (breed if above the **living fleet's mean** ≈ the 50th
  percentile, strictly weaker than their 75th). Measured: steps **+73** (noise) and genome spread
  **0.0367 vs 0.0376** — no diversity benefit either. **Do not adopt their change on this evidence.**

**The real risk is the coupling to 4b.** If the binding constraint is the age/reproduction economy, then a
gate that removes **39% of births** is squeezing exactly the load-bearing quantity. That is the one
place their intuition points at something true — but the correct test is "does the gate help or hurt
when reproduction is the binder?", not "choose a different percentile".

---

## 6. Cheap, falsifiable next steps (ranked; none is a broad sweep)

1. **`gs_w_vision` is under-weighted while vision is the only positive lever.** Deployed `gs_w_vision=0.15`
   against `gs_w_energy=3.0`, and `WIDE_VISION` was the only arm that gained (+1,052). One paired test at
   ≥40 unseen seeds: `gs_w_vision` 0.15 → 1.0/3.0. Mechanism gate first: does the fleet's mean
   `vision_range` actually rise, and does `blind_frac` fall? (Both are already instrumented here.)
2. **Fix the speed defect the oracle exposed.** One rule: **never sprint for distance, and scale forage
   speed with energy while food is continuously visible** — physics-justified by the break-even table
   (walk 333–1000, sprint 71–214). Testable as "does travel/fruit fall without income falling?".
3. **Reproduction vs the age cliff.** Measure the birth rate needed to hold N against a 600–1,200-tick
   age cliff, and test whether the gate's −39% births costs survival in the 6,600–8,000 collapse window.

**Do NOT do:** widen a search radius (there isn't one) · sprint to distant fruit (net-negative) ·
gate on "within X of best" (already impossible here) · copy 90→500 px (no analogue).

## 7. Cross-check against the other agent's independent ablation

After my runs finished, the parallel agent's `HANDOFF.md` disclosed the kills above and published its own
single-constraint ablation (different harness `bottleneck.py`, 4 seeds, control 9,420 ticks, an older
param set). The two experiments were built independently and agree where they overlap:

| intervention | their ablation (4 seeds) | my arms (12-16 paired seeds) | agree? |
|---|---|---|---|
| perfect vision (they set `vision_radius=5000`; my true 360° oracle) | **−17.8%** | ORACLE_FRUIT **−35%** | **yes, same sign** |
| no predators | +8.4% | ORACLE_NOPRED −4% (8W/8L, ~neutral) | weak conflict (their n=4) |
| block reproduction | **−82.5%** | aging-OFF **+17%** (the mirror image: both say the birth/age economy binds) | **yes, same mechanism** |
| lower movement cost | **+33.5% (their largest gain)** | break-even analysis: walking is the profitable mode, sprinting to distance is net-negative | **yes** |

Their summary — *"movement/travel economics is the binding constraint; reproduction is the survival
MECHANISM (not the disease); predators are minor"* — is consistent with everything measured here, with
one refinement this session adds: the movement problem is **not** "our agents don't travel far enough"
(travel/fruit is already 263 units, inside the profitable 333–1000 band). It is that **the speed rule
spends energy the agent cannot afford whenever food is continuously visible** — the defect the oracle
exposed, and the reason their 5,000 px "perfect vision" cell came out *negative* while the legal 400 px
cap came out positive (+1,052).

Also worth carrying forward from their handoff: **the official score is the MEAN of 3 runs** (board keeps
the best attempt), so the floor matters as much as the peak — which is why the p10 columns above are
reported next to the means.

## 8. Limits

Local x86 and macOS both ran and agreed on ordering and signs, but magnitudes differ ~10–20%, and the
grader's seeds are harder than ours — treat these as **ranking evidence**, not predicted board points.
12–16 seeds per arm; the no-age arms lost 3 seeds to an external kill. Right tails are heavy, so the
p10 column matters as much as the mean. The oracle arms change **only the observation** fed to the
policy; the simulator is untouched except where explicitly flagged (no-predator and no-age arms).
