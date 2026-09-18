# W5 — SCENARIO-GENERATION CURRICULUM + RESIDUAL-RL RED TEAM
### replaces self-play for a single-agent-vs-fixed-environment long-horizon survival problem

Status: **design only — no code, no compute, no deployment.** Written 2026-09-18 for the survival-simulator
(Nordic AI Cup), target 1,800 score = 18,000 ticks, current ~848 (official samples 768.68 / 918.33).

Every claim below is tagged **[SIM]** = read out of this repo's source, **[MEASURED]** = already measured in
this project (file given), **[SRC]** = external literature with URL, **[INFER]** = my inference/derivation,
not a source and not yet measured. Nothing here is to be believed on train seeds: §3 is the only scoring rule.

---

## 0. The three numbers that decide the whole design

| fact | value | source |
|---|---|---|
| the score is time alive | `score += dt` once per tick, `dt=0.1`; a full fruit is `+0.06`; being eaten is `-energy/100` (≤5); **no episode cap** — a run ends when the last agent dies | **[SIM]** `src/elements/environment.py:760`, `:674`, `:726` |
| the world decays | `tree_spawn_chance *= 0.5 ** (time/300)`, `time` in sim-seconds, 1 tick = 0.1 s ⇒ **production halves every 3,000 ticks**: ×0.5 @3,000, ×0.125 @9,000, ×0.0156 @18,000 (64× down) | **[SIM]** `environment.py:742` |
| predator pressure is a function of TIME, not of the fleet | `predator_spawn_chance = (1/max(1,N_pred)) * dt * time * 0.0001`, independent of fleet size, and **predators are never removed** (they only go `resting`) | **[SIM]** `environment.py:763`, `:729-731` |

**[INFER] Derivation from those mechanics (matches this project's own observations).** With `dN/dt_ticks = 1e-6 · t_ticks / N`, integrating gives **`N_pred ≈ 1e-3 × tick`** — i.e. ~2 predators at t=2,000, ~4 at t=4,000,
~5 at t=5,000, ~9 at t=9,000, ~18 at t=18,000 (rate-limited in practice by `_is_position_free` placement
failures, so treat late values as upper bounds). The mid-game predictions **exactly match the recorded predator
counts** in `JOURNAL_2026-09-18.md` §16 (1→3 over t=1,001→4,001) and the PIVOT_DESIGN note "predators rising
2 → 4 → 5". Consequence: **"predator count" is not an independent curriculum axis — it is a deterministic
function of `t0`.** The curriculum must therefore treat `t0` as the primary axis and harvest predator state
from real trajectories rather than inventing it.

**The gap the curriculum has to cover.** The base (H1) dies at 7,894-9,069 ticks on 20 fresh held-out seeds
horizon 18,000 **[MEASURED `JOURNAL` §18.1]** ⇒ roughly **9,000-10,000 of the 18,000 target ticks are states
the base policy never visits at all**. The deploy bar is +500 ticks; the target needs +9,000. A curriculum that
only samples states the base already reaches cannot produce the second half — §2.3 (the moving archive) is the
mechanism that does.

---

## 1. What transfers from the literature, and what does not

### 1.1 Transfers (with the adaptation each one needs)

**(a) Prioritised Level Replay — the curation half of UED. [SRC]** Jiang, Grefenstette & Rocktäschel, ICML
2021, <https://arxiv.org/abs/2010.03934>, code <https://github.com/facebookresearch/level-replay>. PLR's
requirements are exactly satisfied here: (i) a notion of "level" exists, (ii) levels are samplable in an
identifiable way (a seeded simulator — ours is deterministic by construction **[MEASURED]** `JOURNAL` §2,
462.3278 reproduced ×8), and (iii) the environment can be set to a level to collect new experience. PLR's
score — time-averaged **L1 value loss / TD-error magnitude** over the last episode on that level, mixed with a
staleness distribution so old scores do not drift off-policy — is the cheapest defensible learning-progress
signal and needs no extra model. **Adaptation:** our "levels" are *snapshot states*, not procedural
parameters, so the level identifier is `(world_seed, t0)` plus a snapshot hash (§2.2).

**(b) Regret-based design (PAIRED / ACCEL) — the frontier half. [SRC]** Dennis et al. 2020
<https://github.com/ucl-dark/paired>; Parker-Holder et al. ICML 2022 (ACCEL)
<https://arxiv.org/abs/2203.01302>, <https://proceedings.mlr.press/v162/parker-holder22a.html>. PAIRED needs a
*protagonist and an antagonist*, which we do not have. **Adaptation, and this is the design contribution that
makes regret usable here: substitute the FROZEN BASE POLICY for the antagonist.** We can evaluate the base's
return on any scenario exactly and for free, so
`regret(scenario) = return_composite(s) − return_base(s)`
is an exact, unbiased relative-regret estimate with **no teacher network and no second learner** — which
removes the two failure modes ACCEL exists to patch (a learned teacher that proposes unsolvable levels, and
teacher-training instability, <https://arxiv.org/abs/2308.10797>). Positive regret = the scenario is one where
learning has not yet matched the base (too hard / pathological); strongly negative regret and zero variance =
plateau (too easy). This is also the correct *shipping* signal, because the composite is only better than the
base where this quantity is positive.

**(c) ACCEL's mutation operator. [SRC]** Same paper. Small *edits* to curated high-regret levels
(mutate `t0` by ±750, fleet size by ±1, predator count by ±1, bank by ±20%) produce an open-ended frontier with
~0.05% of POET's sample budget on a single GPU. Cheap here: an edit is a parameter change plus a snapshot
re-capture, and the edit is only kept if it clears the regret gate.

**(d) Reverse / start-state curricula. [SRC]** Florensa et al. CoRL 2017 <https://arxiv.org/abs/1707.05300>;
Backplay <https://openreview.net/forum?id=H1xk8jAqKQ>; RFCL (reverse then forward), ICLR 2024
<https://proceedings.iclr.cc/paper_files/paper/2024/file/cd062f8003e38f55dcb93df55b2683d6-Paper-Conference.pdf>;
SCOUT, per-context reset curricula <https://arxiv.org/abs/2607.26417>. All of them: *begin training partway
through the task and move the start states outward as competence grows.* One key negative from that line:
**a forward curriculum alone cannot work in a sparse-reward setting — the reverse/start-state stage is
load-bearing** because there is no easy initial state to expand from. Mapping: our "distance from goal" is
`t0`; our reverse stage is "start at t0=3,000 with a healthy harvested fleet" and the forward stage expands
`t0` outward and re-mixes tick-0 episodes back in.

**(e) Go-Explore's archive. [SRC]** <https://arxiv.org/abs/1901.10995>; Nature 590, 580-586 (2021),
<https://doi.org/10.1038/s41586-020-03157-9>. Remember promising states, *return* to them, then explore. The
"return" step is free for us (we can restore a snapshot), which is the *only* reason a mid-decay curriculum is
affordable at all. Using it: §2.3.

**(f) Potential-based reward shaping. [SRC]** Ng, Harada & Russell, ICML 1999
<https://mlanthology.org/icml/1999/ng1999icml-policy>: `F(s,a,s') = γΦ(s') − Φ(s)` leaves the optimal policy
invariant. Used below to make "advantage over the base" the learning signal without changing what is being
optimised (§4, risk R3).

**(g) Residual Policy Learning. [SRC]** Silver, Allen, Tenenbaum & Kaelbling
<https://arxiv.org/abs/1812.06298>, <https://k-r-allen.github.io/residual-policy-learning/>. Three transferable
properties: the base need not be differentiable (`∇θ πθ = ∇θ fθ`); the residual is initialised to **zero** so
start-of-training == the known-good controller; and it is *more sample-efficient than learning from scratch*,
succeeding on long-horizon sparse-reward tasks where from-scratch RL is intractable. Two details from the paper
that our design must copy: **a critic "burn-in" period before the policy is allowed to move** (they train the
critic alone until its loss dips below β=1.0, precisely so that a bad critic cannot wreck a good initial
policy) — and a history/recurrence for POMDPs (history length 1 sufficed for them).

**(h) DAgger's diagnosis. [SRC]** Ross et al., AISTATS 2011
<https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf>: behaviour cloning is not i.i.d. —
a small per-step error moves the learner into states the expert never visited, and errors compound as O(T²ε).
This is the *theoretical* statement of the failure we measured (§6).

**(i) Long-horizon credit assignment. [SRC]** RUDDER (return decomposition / reward redistribution),
<https://arxiv.org/abs/1806.07857>: for dominated-immediate-reward problems, redistributing return so that
expected future reward ≈ 0 makes TD unbiased. Held as the second-line tool if the shaped reward of §4/R3
under-delivers.

**(j) SMiRL. [SRC]** Berseth et al., <https://arxiv.org/abs/1912.05510>. Unsupervised survival drive
("maximise the probability of visited states under a density model") learns Tetris/HoldTheLine/fall-avoidance
with *no* task reward. Relevant as a **diagnostic/fallback**: if the residual cannot be made to want income, an
intrinsic stability term gives it a reason to keep the world predictable instead of idling. Not a first move —
it optimises entropy, not score.

### 1.2 Does NOT transfer

- **Self-play, league training, opponent pools, PFSP/Best-of-N.** The Lux S3 1st-place recipe's *central*
  ingredient (<https://www.kaggle.com/competitions/lux-ai-season-3/writeups/ecobangbang-ecobangbang-s-approach-yet-another-rl->
  — IMPALA + V-trace from scratch, self-play against past selves, frozen teacher KL, opponent pool, ~20 B steps).
  **We have no adversary**: the predators are a fixed scripted policy **[SIM]** `src/elements/predator.py`.
  Every "improve by playing a better version of yourself" mechanism is unavailable. Scenario generation is a
  strictly weaker substitute: self-play supplies an *adversarial curriculum with a regret guarantee*; scenario
  generation supplies a *curated distribution*. The compensating advantage is that our base regression cannot
  happen by construction (§6).
- **The teacher-KL anti-forgetting trick.** It exists to stop a *from-scratch* network forgetting earlier
  opponents. With a frozen-forever base and a bounded residual there is nothing to forget; the analogue is
  keeping a fraction of episodes at the *target* distribution (§2.4), which is cheap.
- **"Self-play with the zero-sum output of the baseline model."** The winner's own words
  (<https://www.kaggle.com/competitions/lux-ai-season-3/writeups/ecobangbang-ecobangbang-s-approach-yet-another-rl->):
  zero-sum alone was *not enough*; what made it work was playing against *past selves*. We cannot.
- **20 B steps of scale.** Their budget is ~2-3 orders of magnitude above ours (§2.5). Their recipe is
  therefore not "the recipe minus self-play" — it is not affordable at all. Only the *sample-efficient* line
  (residual RL + curricula) is in reach.
- **IL as a pretraining stage for RL.** 4th place: "the RL approach did not succeed... complete collapse of
  the IL policy or no improvement... 2-3 weeks of trial and error"
  (<https://kaggle.com/competitions/lux-ai-season-3/writeups/yumeneko-4th-place-solution-imitation-learning-app>);
  3rd place: "nearly impossible to train an IL agent to outperform its teacher". Our own clone reproduced the
  failure independently (§6). Do not re-open this.
- **Map-shaped CNN policy heads (UNet over the full grid).** Works when the observation IS a global map
  (Lux: fixed 24×24, several teams' wins). Ours is a **31-dim per-agent local vector** with partial
  observability **[SIM]** `experiments/env_wrapper.py::build_obs`. A grid head would require inventing an
  occupancy/visibility map that does not exist in the served interface — out of scope for the compute we have.
- **Goal relabelling / HER.** HER relabels *achieved goals*; our episodes have no goal, one terminal event
  (fleet wiped) and a monotone scalar. There is nothing to relabel. The nearest equivalent is scenario
  relabelling of `t0` (a snapshot taken at 9,000 can be trained on as "survive 2,500 more ticks"), which §2.2
  uses — but labelled as a start-state trick, not HER.
- **MAP-Elites/QD as the *primary* search.** Justified in this repo for *diversity across seeds* under
  seed-to-seed outcome variance, **not** for evaluation noise (evaluation is deterministic in a fixed harness
  — the earlier "±1,000-tick noise" framing was corrected in `PLAN_H1PLUS`). It is a candidate *second*
  curriculum mechanism after PLR-regret, not a replacement.
- **Warp RL (2025) and RLPP on F1TENTH (2024)**, cited in `PIVOT_DESIGN.md` §2, are **not re-verified here**
  (no URL in the repo). Treated as claims to check before relying on the "geometric correction" escape hatch.

---

## 2. THE CURRICULUM, CONCRETELY

### 2.1 What a "scenario" is

A scenario `θ` is `(world_seed, t0, snapshot_hash)` where `snapshot` is the **full semantic state** of an
`Environment` at tick `t0`. Restoring a snapshot makes the episode start at `t0` with a real, self-consistent
world: the biome map, the tree/fruit population and its age structure, the fleet with its real energies, ages
and headings, the predators with their real energies and `resting` flags, and the RNG stream position.

Snapshot must capture (from the source, **[SIM]** `src/elements/environment.py`):
`rng.getstate()`, `time`, `score`, `_next_agent_id`, `_next_fruit_id`, `biome_map` (ndarray copy), and the full
attribute dict of every `agents`/`fruits`/`trees`/`predators`/`obstacles` entry (agents: `x,y,direction,energy,
age,speed,sprint_speed,max_energy,hearing_radius,vision_radius,cone_angle,size,max_age,color,agent_id`; fruits:
`x,y,radius,energy,age,fruit_id`; trees: `x,y,radius,age`; predators: `x,y,direction,energy,size,speed,
sprint_speed,resting,max_energy`; obstacles: `x,y,width,height,color`). The four spatial grids are **rebuilt**
by `_update_spatial_grid()` on restore, and `agent_observations` is cleared. `pygame` surfaces are **not**
captured: training runs headless (`SDL_VIDEODRIVER=dummy`, already the repo convention) and restore assigns the
lists directly rather than calling `spawn_*` (which is what draws to `shadow_surface`/`obstacle_surface`).

> **Acceptance test for the snapshot API (non-negotiable, this project's own determinism standard):**
> capture at `t0`, restore into a fresh process, roll 20 ticks, and require a **bit-identical** action digest
> and per-tick world fingerprint versus the un-interrupted run. **[MEASURED]** precedent: the object-ordering
> fix had to reproduce `462.3278` exactly ×4 in-process and ×4 in a fresh process (`JOURNAL` §2). A snapshot
> path that is only *approximately* deterministic is a measurement bug, not a curriculum.

### 2.2 The two scenario families

**Family H — HARVESTED (physically self-consistent).** Roll the **frozen H1 base** from tick 0 on a bank
world seed, capture a snapshot at each checkpoint in
`t0 ∈ {3,000, 4,500, 6,000, 7,500, 9,000}` (5 checkpoints), and record that policy's remaining survival from
each checkpoint. Not every seed reaches every checkpoint: H1's 20-seed means are 7,894.4 / 9,068.9, so
**`t0 = 9,000` is reachable only on the stronger seeds** — that asymmetry is *deliberate* and is itself
information (a scenario bank is allowed to be seed-imbalanced; the *evaluation* is not, §3).
Advantages: the fleet's age distribution, the predator count, the standing-fruit level and the biome
realisation are all **emergent facts of a real trajectory**, so no invented state is physically impossible.

**Family S — SYNTHESISED (for `t0` beyond the base's death).** Since the base cannot reach `t0 > 9,000`, states
there must be constructed. Recipe (each step is a direct read of the mechanics):
1. Set `env.time = t0/10` **before** any growth: `time` alone drives `decay_factor = 0.5**(time/300)`
   **[SIM]** `:742` and `predator_spawn_chance = (1/N)·dt·time·1e-4` **[SIM]** `:763`. Setting `time` is
   therefore the whole of "put the world into its late-game transient".
2. Spawn the fleet as a **relay-ready cohort**: `K` agents with ages spread uniformly in `[0, max_age]`
   (agents die of age at 600-1,200 ticks **[SIM]** `creature.py`/`PLAN_H1PLUS`), energies drawn from the
   measured `t0=9,000` marginals (see 4).
3. Spawn predators to `N_pred(t0) ≈ round(1e-3 · t0)` **[INFER, §0]**, each with energy `U(100,200)` and
   `resting=False` (they wake above 50% of max energy **[SIM]** `:680`) — a deliberate worst case; also make a
   `resting=True` variant so the scenario axis is not silently the awake-count.
4. **Burn in `t_burn = 600` ticks with the fleet's movement disabled** so that production at the decayed rate
   establishes a standing-fruit equilibrium instead of starting at the tick-0 fruit density. Without this,
   a synthesised `t0=15,000` scenario hands the fleet a population of fruit that the decayed world could not
   have produced — a silent reward leak.
5. **Curation gate (accept/reject):** measure production over the burn-in with the `w_reach.py` identity
   (`production = eaten + standing_delta`, exact because `score += fruit.energy/1000`) and **reject any
   synthesised scenario whose production per 1,000 ticks is more than ±25% away from the analytic decay curve**
   (≈12-13 e/tick at t=0, halving every 3,000 ticks **[MEASURED `PIVOT_DESIGN` §1.1, `w_reach.py`]**).
   This is the whole defence against "invented states that do not exist".

**Family R — RELAY (targets the endgame mechanism of W2).** A special synthesised case: `t0` anywhere in
`[9,000, 15,000]`, `K = 3` agents, one of them holding `≥ 250` energy (above the 100-energy sprint lockout
plus the 100-energy birth cost **[SIM]** `:512`, `:621-623`; **[MEASURED]** the L2/L4 arms at 250/350 showed the
*strategy* loses but the *arithmetic* — a parent must keep ≥150 after paying for a birth — is unchanged), and
hereditable `max_energy` at the high end of the mutated range. The one-lineage cost is `0.1/tick metabolism +
100 per relay spawn per ~900 ticks = 0.211 e/tick = 3,800 energy over 18,000 ticks, 7% of what the world
produces` **[MEASURED `PIVOT_DESIGN` §1.1]** — so these scenarios are *solvable*, and their difficulty is
access and predation, not supply. Family R exists so that the residual is at least asked the endgame question.

### 2.3 The MOVING ARCHIVE — the answer to "late-game states are rare and precious"

A fixed bank cannot contain `t0 > 9,000`, because no policy in the project has ever survived that far. The bank
must therefore be regenerated from the *current* policy — this is Go-Explore's archive growth with ACCEL's edits:

```
Stage 0  SEED BANK      500 world seeds × Family H checkpoints           (base policy only)
Stage 1  TRAIN 30M      t0 <= 6,000; PLR-regret curation; frontier grows as wins accrue
Stage 2  RE-HARVEST     after every 200 PPO updates: roll the CURRENT composite, capture at
                        t0' = previous frontier + 1,500; keep only snapshots whose
                        (a) regret gate passes and (b) the scenario is reachable by the composite
                        itself at least 1 time in 20 tries  -> state is physically reachable, not invented
Stage 3  SYNTHESIS      when the composite still dies before t0', fall back to Family S/R for that tier,
                        subject to the ±25% production curation gate
Stage 4  SHIP CHECK     tick-0-only fine-tune (10M) + the residual_scale sweep
```

**Bank-build cost (the load-bearing feasibility number).** 500 seeds × ~9,000 average ticks =
**4.5 M fleet-ticks = 1.65 core-hours (~99 min)** at the stated ~756 fleet-ticks/s/core. On the 4-vCPU VPS at
2 CPUs (LANES rule: max two `nac-*` containers × `--cpus=1.5`) that is **one overnight bank build**, repeated at
most twice (Stage 2). *This is what makes the whole approach affordable: the curriculum's cost is dominated by
harvesting, and harvesting is two orders of magnitude cheaper than training.*

> **[INFER] Performance note that the cost estimate depends on.** `Environment.__init__` calls
> `_render_biome_surface`, which does `width×height = 1.92 M` `set_at` calls per episode. At ~9,000 ticks per
> episode that is a material fraction of episode cost. **Cache the biome surface / `biome_map` by `world_seed`**
> and reuse one `Environment` instance per worker, re-populating it from snapshots rather than re-constructing.
> Measure before/after; if the cached build is not ≥20% faster per episode, the 99-minute figure must be revised.

### 2.4 The sampler: which slices to start from, and how to prioritise them

**Level vector (the design space, deliberately small):**

| axis | values | justification |
|---|---|---|
| `t0` (start tick) | `{3,000, 4,500, 6,000, 7,500, 9,000}` (H) then frontier `+1,500` steps to 16,500 (S/R). Spacing 1,500 because decay halves every 3,000 ticks — a 1,500 gap is a 1.41× change in production, enough to be a different problem, few enough to keep tiers populated. | **[SIM]** `:742` |
| fleet size `K` | 1-14 (harvested: whatever it is; synthesised: 3, 6, 10) | observed mid-game range "10-16 agents" (`JOURNAL` §11) |
| fleet energy | harvested: real; synthesised: per-agent mean ∈ {60, 150, 400} × `max_energy`. **Explicitly include sub-100-energy starts**, because the sim clamps distance to walk speed below `max_energy/5` and **58.1% / 54.4% of all agent-ticks are already spent in that lockout zone**, and 92% of predation deaths happened inside it | **[SIM]** `:512`; **[MEASURED]** `JOURNAL` §18.3 |
| bank | standing fruit energy within the fleet's own 9-chunk neighbourhood: `{0, 1,000, 3,000}` **plus** harvested reality. Note the world holds 2,000-4,900 e standing while agents starve, so "bank ≈ 0 with 3,000 standing nearby" is a *real* and *important* state | **[MEASURED]** `JOURNAL` §4, §18.3 |
| predator count / state | `round(1e-3·t0)` (derived), each awake with `U(100,200)` or `resting` | **[INFER]** §0; **[SIM]** `:680` |
| standing food (global) | harvested; synthesised: whatever the decayed burn-in produces, gated ±25% | §2.2 |
| biome at spawn / patch quality | NOT an axis — it is inherited from `world_seed`. Do not let the sampler "place the fleet in forest", because that is exactly the kind of invented advantage the curation gate exists to reject | |

**Do NOT feed `t0` (or any scenario id) to the policy.** The base already reconstructs the clock from `age`
and call counts to **max error 1 tick** **[MEASURED]** `JOURNAL` §11(a). Feeding `t0` invites scenario-specific
hacks that then fail on held-out seeds; making the policy infer time from observables forces generality.
(Ablate this once, cheaply, and report it.)

**Prioritisation — three signals, combined:**

```
per scenario i, after every episode from it:
  PLR score      S_plr,i  = mean_t |V_hat(s_t) - V_target(s_t)|        (L1 value loss, PLR Eq. 2)
  regret         R_i      = return_composite(i) - return_base(i)        (exact; base is frozen)
  staleness      C_i      = c - C_i      (episodes since last sampled)
  replay prob    P(i) ∝ 0.5 * score_softmax(S_plr) + 0.5 * score_softmax(R) , then mixed with staleness
  K = 200 per t0 tier ; p_replay = 0.6 ; 40% uniform-random over the tier (exploration)
```
- `V_target` for the L1 loss is the observed return-to-death from that snapshot (a truncated-episode target,
  §2.6). The PLR form is used verbatim including the staleness mixture, because the score is estimated under an
  older policy and drifts off-policy otherwise **[SRC]** PLR §3.2.
- **The regret term is the one that matters for shipping**, for the reason in §1.1(b): a scenario where the
  composite beats the base is a scenario where the residual is *doing something right*. Tiers where the
  composite's regret is strongly negative and flat are **retired** (the plateau signal).
- **Tier pacing (the reverse-curriculum part):** start with `t0 ∈ {3,000}` only. Promote the frontier tier when
  the composite's win-rate against the frozen base on the last 200 frontier episodes is **≥60%**; demote it if
  **<30%**. This is the Florensa/SCOUT "move the start states outward as competence improves" rule with the
  base as the competence reference.
- **A deliberate floor of uniform tick-0 episodes: 20-30% of every rollout batch must be real tick-0 episodes.**
  This is the task-distribution anchor and the substitute for the winner's teacher-KL anti-forgetting device.
  Without it the policy is optimised on a distribution whose marginal differs from the graded distribution and
  we would be shipping a scenario specialist.
- **Fallback, if PLR-regret proves too noisy at ~1,000-tick granularity: ALP-GMM** **[SRC]**
  <https://arxiv.org/abs/1910.07224>, code <https://github.com/flowersteam/teachDeepRL> — cluster the level
  vectors with a GMM and sample in proportion to |Δreturn| over a sliding window. It is the right tool for
  *continuous* parameter spaces with a noisy progress signal, and ours is 4-6 dimensional. Run one arm; do not
  run it by default (ACCL found ALP-GMM degrades when the design space grows, ACCEL §6).

### 2.5 Budget, horizon and discount

**Budget (60 M agent-steps, ≈0.3% of the Lux winner's 20 B):**

| stage | steps | composition |
|---|---|---|
| 1 — early tiers | 30 M | `t0 ≤ 6,000`, 70% curated / 30% uniform within tier |
| 2 — frontier | 20 M | curated `t0 = 7,500-12,000` + moving-archive re-harvest + ACCEL edits |
| 3 — target fine-tune | 10 M | **tick-0 real episodes only** + `residual_scale` sweep |
| eval (never trained on) | 6 M | the held-out arms of §3, run in the same process |

> **Fix this conversion before spending anything.** The brief gives two calibrations that do not agree:
> "~756 fleet-ticks/second/core" (`10M agent-steps ≈ 5 min` is implied to be *steps*, but 756 fleet-ticks/s
> with a mean ~6 live agents is ≈4,500 agent-steps/s ⇒ 10 M steps ≈ 37 min). They differ by ~4-8×. **First
> implementation step: a 5-minute wall-clock benchmark of the actual training entrypoint** (fleet in the loop,
> `build_obs`, forward+backward) reporting *fleet-ticks/s, agent-steps/s and episodes/hour*; then re-express this
> table in that unit. Budgeting in "steps" without the benchmark is how a 2-hour plan becomes a 2-day plan.

**Episode length: `Δt = 2,500` ticks of training episode after `t0`, then truncate (bootstrap, do not treat as
terminal).** Reasons: (i) 2,500 ticks spans at least two generational relays (ages 600-1,200) and one
predation-cascade window (the measured collapse is "14 of 16 agents gone in 1,000 ticks", `JOURNAL` §16);
(ii) shorter episodes mean more distinct scenarios per step, which is the currency of a curriculum;
(iii) it keeps the effective horizon inside the reach of GAE. Concretely: at ~6 live agents, 2,500 fleet-ticks
≈ 15,000 agent-steps/episode ⇒ **60 M agent-steps ≈ 4,000 scenario episodes.** Truncation at fleet death is
automatic and free (`stop_on_death=True` semantics).

**Discount — `γ = 0.9995` (half-life 1,386 ticks), GAE `λ = 0.95`; ablation `γ = 1.0` with bootstrapped `V` at
truncation.** Half-lives: `γ=0.999 → 693`, `0.9995 → 1,386`, `0.99977 → 3,013`, `0.9999 → 6,931` ticks
(computed). A `γ` whose half-life is far below `Δt` biases the learner against surviving past `Δt` — which in
this problem is the whole target — so `γ ∈ [0.9995, 1.0]` is the defensible window and `γ ≤ 0.999` is a bug.

### 2.6 Reward, and why the obvious reward is wrong

`score` accrues `+0.1/tick` **identically for every policy** at a fixed horizon: **[MEASURED]** `RL_JOURNAL`
"per-step reward ~0.1001 = the survival bonus; fruit-bonus steps 0.15% of steps ⇒ training on raw score gives
NO gradient". With episodic termination the *return* does depend on survival length, so raw-score-delta PPO is
not vacuous — but at a 2,500-tick scale the value function is dominated by "expected remaining lifetime", whose
variance dwarfs the behavioural signal, and the gradient toward *doing something different* is tiny.

**Recommended (and this is the single most important recommendation in this document):**

```
r_t = (score_delta_t  -  (γ·V_H1(s_{t+1}) - V_H1(s_t)))          # potential-based, Φ = V_H1
      + w_fruit · (fruit_energy_eaten_t / 1000)                   # w_fruit = 1.0 (the sim's own term)
      - w_income · 0                                        # income is a DIAGNOSTIC, not a reward
```
- `V_H1` is a critic **pre-trained by supervised regression on the base's own rollouts** (this is RPL's
  critic burn-in, §1.1(g), reused as a potential). By **Ng et al. 1999** the shaping term is potential-based,
  so it cannot change the optimal policy of the underlying score — it only converts the learning signal into
  "behave better than the base", which is exactly what a residual should learn **[SRC]**
  <https://mlanthology.org/icml/1999/ng1999icml-policy>. Free bonus: it fixes the scale problem, because the
  shaped reward's mean is ≈0 for a policy that behaves like the base.
- Keep the fruit term: it is *part of the real score* (`+0.06` per full fruit ≈ 0.6 tick of survival) and it is
  the anti-passivity instrument **[MEASURED]** the passivity optimum is documented (791 → 1,195 ticks with
  income flat at 1.3-9.3 against the baseline's 118).
- **Do not** add a large hand-shaped income reward: survival time is the objective and income is its mechanism.
  A big income term would optimise the proxy against the metric (the repo already has a documented case of
  this: the extreme fruit term in the `evolve_det.py` ranking had a **headcount bias** — bigger fleets eat more
  total fruit by having more mouths — and had to be capped at `FRUIT_W = 0.05`, `JOURNAL` §13).
- **Do not** use an income-weighted shape as the primary reward "because income drives survival". It is true
  **[MEASURED]** (income collapses after t≈3,000 and goes negative after t≈4,000, `JOURNAL` §3), but turning it
  into reward re-creates the trap the project already measured: every measure that reduced income to save
  energy lost 9-36% monotonically (five arms, `JOURNAL` §18.1). Income is a *constraint to be logged*, survival
  is the *reward*.

---

## 3. KEEPING THE EVALUATION HONEST

### 3.1 The only number that counts

**Primary bar (pre-registered, from `PLAN_H1PLUS` step 1): the candidate must beat the frozen H1 snapshot by
more than 500 ticks on ≥15 of 20 PAIRED held-out real episodes from tick 0, horizon 18,000, on Linux, in one
process, with the H1 arm in the same run.**

Arithmetic (computed, not asserted):
- **Wins test.** Under the null (no effect) each paired seed is a coin flip. `P(≥15/20) = 0.0207`,
  `P(≥16/20) = 0.0059`. So the 15/20 rule is a one-sided test at **α ≈ 2%** — it is a genuine test, not a vibe.
- **Effect test.** With mean paired delta 500 ticks and paired sd `s ∈ {700, 800, 900, 1000}` at `n = 20`:
  SE = 157/179/201/224, t = 3.19/2.80/2.48/2.24, **power = 0.93/0.86/0.78/0.69**. At `n = 30`:
  power = 0.99/0.96/0.91/0.84. At `n = 40`: **0.99 / 0.99 / 0.97 / 0.90**.
  ⇒ **n = 20 is adequate to SCREEN a +500-tick effect at ~70-90% power, and n = 40 to SHIP it.**
  For shipping require **≥28 of 40 wins** (`P = 0.0083` under the null) **and** mean delta CI lower bound > 0.
  `s` is *assumed* at 700-1,000 (paired deltas remove seed/world variance, and the project's measured
  per-seed spread is ±1,000-2,000 ticks unpaired, `JOURNAL` §5). **Measure `s` from the first 20 paired runs
  and re-derive the required n; that is the honest sequence — do not inherit this assumption.**
- **Why >500 ticks (not +314):** the measured seed-noise band is ±500 ticks and H1-vs-previous-controller at
  +314 (+4.5%) was explicitly *not* shippable **[MEASURED]** `PLAN_H1PLUS`; a single official validation swings
  ±150 score (768.68 vs 918.33 on the same controller) **[MEASURED]** `JOURNAL` §17.1. Anything inside that band
  is unmeasurable and unshippable.

### 3.2 Seed hygiene

```
BANK world seeds   2000-2499   (curriculum harvesting; never evaluated on)
TRAIN/probe seeds  100-899     (existing convention: probing the scenario marginals)
SCREEN seeds      1301-1320   (20 paired; used ONLY for the screen)
SHIP seeds        1301-1340   (40 paired; a superset of the screen, decided in advance)
MID-DECAY eval    1501-1520   (20 paired at t0 = 9,000 -- secondary, never the headline)
```
Rules inherited unchanged from this project and to be applied without exception: resample fitness seeds per
generation; never believe a train-seed win; report **mean AND median AND min AND seeds-won**; a negative result
is a result; one heavy job at a time; prove determinism with the *real* policy before trusting a ranking.

### 3.3 The single-write-per-seed universe problem (the trap specific to this design)

A scenario-based curriculum is scored on scenario episodes; the grader scores real episodes. Therefore:
- **`residual_scale` is a deployment parameter**: serve `residual_scale ∈ {0, 0.25, 0.5, 1.0}`; `0` is *exactly*
  H1. The ship decision is "which of these four wins the 40-seed paired test", which is a monotone family with
  the baseline inside it, satisfying this project's own "include the baseline arm in the same run" rule.
- **Never** enter a scenario-episode number into the headline. It is a training diagnostic.
- **A scenario-specialist check**: the winning `residual_scale` must also show a *positive* delta on the
  20 mid-decay paired episodes. A residual that wins only at `t0 ≥ 9,000` and loses at tick 0 is a specialist
  and is not deployable (the grader starts at tick 0).
- **A reachability check**: the residual must be *runnable from tick 0*. A time-gated residual (active only
  after `t > t_switch`) is legal and testable; a residual that only exists inside a synthesised scenario is not.

### 3.4 Diagnostics that must be logged every run (so that the failure modes are visible, not inferred)

`death mix (starved / eaten / aged)`; `income per agent-tick`; `blind fraction` (share of agent-ticks with no
fruit visible); `lockout fraction` (share of agent-ticks with `energy < max_energy/5`); `fleet size trajectory`;
`mean |Δdist|, |Δturn|` and the histogram of pre-squash residual logits; `predator count vs tick`;
`standing fruit energy vs tick`; `ticks where the composite lost agents in a single 1,000-tick window`.
The **passivity flag** is a computed conjunction: *survival up ≥10% while income per agent-tick down ≥20%* on
the same run ⇒ stop and read it as the documented passivity optimum, not as progress.

---

## 4. RED TEAM — one page on the residual-RL design

**R1 — Boundedness vs expressiveness: the residual's ceiling is the base's capability class.**
The measured binding failure is **ACCESS**, not supply: agents are blind **76-87% of ticks** while
**2,000-4,900 energy of fruit stands in the world**, seeing 0.26-0.42 fruits each **[MEASURED]** `JOURNAL` §18.3.
The plausible fixes (ballistic commitment after failure, patch memory, dispersion-limited following) are
*changes to what the agent remembers and how it commits*, not a translation of the current action. A bounded
additive residual `(|Δdist| ≤ 20% of speed, |Δturn| ≤ 0.3 rad)` **cannot create memory** and can only shift the
existing potential field. Honest expectation: the residual will improve predation-adjacent geometry (where a
small steering change is causally close to the outcome) and will do little for the blinding. **Mitigation and
escalation ladder:** (i) first allow the residual to raise `move_distance` toward a *sprint commitment* while
blind (that is inside the box and is the one change with a measured mechanism: foraging is "the income
mechanism, not discretionary burn", `JOURNAL` §11); (ii) if it plateaus, **widen the BASE** — put a properly
priced access mechanism into H1 (walk-speed commitment, dispersion-limited following, retested at *matched*
energy per tick) and residual-learn on top of the new base; (iii) only then the geometric/parametric correction
route (Warp RL, unverified here §1.2).

**R2 — The residual sitting at its bound.** A `tanh`-squashed residual trained mostly on survival will
saturate: a constant `Δdist = −20%` is a cheap way to cut the 0.225 e/tick walk burn to 0.18 while income
falls, and `|pre-squash logit| > 3` for most ticks means the "policy" is a constant. **Detection:** logit
histogram + the fraction of ticks with `|Δ| > 0.8 × bound`; the passivity flag of §3.4. **Mitigations:**
(a) an L2 penalty on pre-squash logits; (b) a **bound schedule** — start at 10% of speed / 0.15 rad and raise to
20% / 0.3 rad once the critic has burned in; (c) keep a **state-independent exploration std on the residual and
do not anneal it to zero**, so the composite remains a distribution over behaviours rather than a fixed offset.

**R3 — Credit assignment over 10k-tick episodes with a `+0.1/tick` reward.** See §2.6. The raw `score_delta`
gives a **stepwise-constant** reward **[MEASURED]** `RL_JOURNAL`, so the entire learning signal is (i) the
episode-length term, whose variance is enormous, and (ii) the tiny fruit/eaten terms. Mitigation is the
`Φ = V_H1` potential-based shaping (§2.6), i.e. learn the *advantage over the base*; ablation `γ=1.0`,
second-line tool RUDDER **[SRC]** <https://arxiv.org/abs/1806.07857> if the shaped signal still under-delivers.
**Falsifiable prediction:** if the shaped-reward arm does not beat the raw-reward arm on the 40-episode
*scenario* diagnostic, the shaping theory is wrong for this sim — record it and stop.

**R4 — Reward definition: raw score delta vs income-weighted shape.** Argued in §2.6: raw score delta is the
grade and must stay the reward; income must be a *logged constraint* and a *small* tie-break term, or the
proxy gets optimised against the metric (precedent: the `FRUIT_W` headcount bias, `JOURNAL` §13). The
counter-risk is the reverse: with income only logged, the learner can find a low-income high-lifetime
walk-shuffle. The passivity flag is the tripwire, and **the flag must be checked before any deploy**, because
the project has already measured the passivity optimum (791 → 1,195 ticks, income flat 1.3-9.3 vs 118).

**R5 — Scenario leakage: training states that the real run never creates.** A snapshot at `t0=12,000` comes
with a banked, healthy fleet that a real episode only earns by surviving 12,000 ticks. A residual tuned to that
bank distribution can be a **scenario specialist**. Mitigations: fit scenario marginals to H1's *measured* joint
distribution (probe with `w_deaths.py`-style instrumentation at each checkpoint); the ±25% production curation
gate; the reachability check (≥1 in 20 restore-and-roll attempts by the composite itself); the tick-0-only
Stage 3 fine-tune; and the mid-decay plus tick-0 dual evaluation of §3.3. **Any scenario whose state cannot be
reached by rolling the *current* composite forward is a design fiction, not a curriculum item.**

**R6 — Chaos makes single-episode outcomes uninformative.** The sim is deterministic but chaotic: identical
code and seed give **4,472 ticks on macOS vs 7,816 on Linux** **[MEASURED]** `JOURNAL` §6, and a ±20% action
change de-correlates trajectories within a few hundred ticks. Consequences: (i) **all training and all
evaluation must happen on Linux** (the grader's platform), or the residual is trained on a different dynamical
system; (ii) per-episode "this looks better" judgements are worthless — only paired multi-seed means count;
(iii) single-episode value-loss as a curriculum signal is noisy, which is why the regret term (a *mean* over
replays) carries half the weight.

**R7 — The non-Markov trap we have already fallen into once (highest-probability failure).** The base is
**stateful**: `_MEM[aid] = {flee hysteresis, spawn_clock, last_steer, clock}`, plus `_SIM_TICK`, `_BLIND_EMA`,
`_FRUIT_EMA`, `_THIN_ARMED`, `_GC` (population estimate) **[SIM]** `best_controller.py:212-289`. The 31-dim
observation does **not** contain any of it. Behaviour cloning died precisely here — "expert depends on internal
state absent from the 31-dim observation" — and the 34-dim variant (31 + 3 private-state features: ticks since
own spawn request, ticks since last flee distance, TTL population estimate) still only reached **1,307-2,721
ticks against an expert's 5,701** **[MEASURED]** `IMITATION_REPORT.md`. A *stateless* residual bolted onto a
stateful base is a bounded, zero-initialised version of exactly that failure: the base's action at tick `t`
depends on history the residual cannot see, so the residual cannot predict the consequences of its own Δ.
**Mitigation (mandatory, cheap):** the residual's input is
`[31-dim build_obs] ++ [base-state vector: spawn_clock, flee flag, last_steer, clock, _SIM_TICK,
_BLIND_EMA, _FRUIT_EMA, _THIN_ARMED, len(_GC)] ++ [frame stack of the last 4 observations]`,
and the ablation (31-dim only) is run as a first-class arm. If the full-input arm does not beat the 31-dim arm,
the diagnosis is wrong and the plan must be re-read before more compute is spent.

**R8 — The learner overfits the scenario id.** Covered in §2.4: do not feed `t0`, fleet size or predator count
to the policy. If they *are* fed, run the held-out-scenario ablation (train on `t0 ≤ 9,000`, evaluate at
`t0 = 12,000`) to expose it.

**R9 — "Never worse at init" is a statement about initialisation, not about training.** RPL's zero-init
guarantee holds at step 0 only **[SRC]** <https://arxiv.org/abs/1812.06298>. The learned residual **can** be
worse than the base. Therefore: keep H1 frozen and shippable forever; the deployment knob is `residual_scale`;
a rollback is "set the scale to 0" — one parameter, no rebuild of logic.

**R10 — Compute and contention.** 20 B steps is unreachable (§2.5). The VPS runs a **graded endpoint** that must
never miss a deadline, and a mid-run container restart has already contaminated a validation once
**[MEASURED]** `JOURNAL` §7 and `LANES.md`. The curriculum therefore needs to claim lanes in `LANES.md`
*before* harvesting (the bank build is the big job: ~99 min × 2 CPUs) and must never touch
`nac-survival-vps`. Train on the Mac only for smoke tests, never for the numbers (R6).

### 4.1 Cheap early-kill tests (≈1 hour each, before any PPO budget is spent)

**KILL-1 — "Inside-the-box headroom" (does the residual's action box contain a better behaviour at all?).**
40 scenarios (`t0 ∈ {6,000, 9,000, 12,000}`, harvested + synthesised, curated). Fit nothing. Instead run a
**random search over a 6-parameter hand-parameterised residual that lives strictly inside the proposed box**
(e.g. `blind_speed_mult ∈ [1.0, 1.2]`, `commit_len ∈ {0, 50, 150, 400}`, `blind_turn_gain ∈ [0, 0.3]`,
`flee_turn_gain ∈ [0, 0.3]`, `pred_dist_target`, `forage_gain ∈ [-0.2, 0.2]`), 200 samples per scenario, and
measure remaining survival from `t0`. **Kill criterion: if no sample inside the box beats the frozen base by
≥10% of remaining survival on ≥60% of the 40 scenarios, then the residual's EXPRESSIVENESS is the binding
constraint** — stop the PPO plan and either widen the base (R1-ii) or move to the hand-parameterised mechanism
directly. This is the cheapest possible falsification of the whole W1 hypothesis, and it costs null GPU time.

**KILL-2 — "Can the critic even see the state?" (observability / credit-assignment test).** Collect 200k
`(obs, realised return-to-death)` pairs from base rollouts on the bank seeds; fit a 2×256 MLP. **Kill criterion:
held-out `R² < 0.6` on the 31-dim observation, and `< 0.75` after adding the base-state vector and 4-frame
stack ⇒ the state is not predictive and PPO cannot assign credit.** The correct response is not "train longer";
it is to add memory (the R7 input set, or a GRU) or to kill the ML track. This test also *ranks* candidate
input sets before any RL time is spent, and it is the direct modern analogue of the diagnosis that killed BC.

**KILL-3 (free, continuous) — saturation monitor.** >60% of ticks with `|pre-squash logit| > 3`, or the
passivity flag of §3.4, ⇒ pause and treat as R2/R4, not as progress.

**Order: KILL-2 → KILL-1 → then the bank build.** KILL-2 tells you whether a policy-gradient method can work;
KILL-1 tells you whether the residual box can help even in principle; only if both survive is 99 minutes of
harvesting justified.

---

## 5. WHAT MUST NOT BE REPEATED (this project's own history, with the numbers that closed each door)

**N1 — Behaviour cloning of a stateful controller. DO NOT RE-OPEN.** Measured outcome on 8 held-out seeds,
horizon 16,000, identical harness **[MEASURED]** `IMITATION_REPORT.md`: **expert 5,701 ticks vs imitation
2,569 / 2,130 / 2,721** for the three clone variants, and the train-seed-selected threshold variant
**1,307 ticks vs the expert's 5,701**; steering MAE **1.32 rad** (~75°); the spawn head's event-F1 was
**0.003** (spawning is gated by a hidden cooldown, so it is not imitable per-tick at all). Independent
reproduction from the outside: 4th place in Lux S3 — "the RL approach did not succeed... complete collapse of
the IL policy or no improvement over default behaviour. After 2-3 weeks of trial and error without success"
(<https://kaggle.com/competitions/lux-ai-season-3/writeups/yumeneko-4th-place-solution-imitation-learning-app>);
3rd place — "nearly impossible to train an IL agent to outperform its teacher". Theory: DAgger/covariate shift
(<https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf>). **The residual design is the
*replacement* for BC, not a prelude to it: it needs no clone because the base IS the heuristic.**

**N2 — Per-tick re-randomised wandering in a patchy world. DO NOT RE-BUILD IT THIS WAY.** Measured: memory-only
**−14.5%** (5,864 vs 6,862 ticks), memory+social **−38.5%**, full trio **−36.4%**, 20 held-out seeds
**[MEASURED]** `JOURNAL` §12. The independent review's corrections are the operative part and must be honoured:
the trial was **mis-priced at ~2×** energy per tick (`commit_speed_frac 1.0` vs a walk at 4.5 units ⇒ 0.225 e/tick
**[SIM]** `:501-518`); **ARS was mis-applied** (the clamp throttled the *direct approach* to a fruit already
being targeted, so it could not behave as area-restricted search); and there were **no ablation arms**.
**Rule for any search/commit behaviour from now on: (i) price it to the baseline's own energy per tick —
commit at WALK speed; (ii) one arm per mechanism, never bundled; (iii) an identical baseline arm in the SAME
run; (iv) if the mechanism is ARS, it must fire only when *no* fruit is visible, or it is not ARS.**
Note also the mechanical reason per-tick re-randomised wandering is structurally wrong here: the world is
patchy (fruit is tree-associated, trees die after 50-150 s **[SIM]** `:751`), so any behaviour that re-draws a
random direction every tick can never hold a patch; the correct primitive is a *committed run* of 50-400 ticks.

**N3 — Training a single isolated agent instead of a fleet. DO NOT.** `FleetEnv` is built with
`starting_agents=1, starting_predators=0` **[SIM]** `env_wrapper.py:113-118` — i.e. **no predators at all**,
while the real failure is a *fleet-level* predation cascade (19 → 2 agents inside 1,000 ticks; 35-43% of all
deaths are predation; each killed agent takes its bank with it: fleet energy 2,767 → 51) **[MEASURED]**
`JOURNAL` §16. And this project has already measured that **the crowd IS the foraging engine**: shrinking the
fleet cut fruit eaten 1,041 → 334 → 185 **[MEASURED]** `JOURNAL` §5. A single-agent trainer cannot see any of
this. For the residual track specifically, the fleet-in-the-loop requirement is *doubled*: the reward couples
agents (score is per-tick, fleet-independent; births cost 100 e and pay 0.1 e/tick of extra metabolism), so
the residual must be **trained and selected with the real fleet and real predators**, with the shared policy
applied to every agent. Keep `FleetEnv` only for debugging.

**N4 — A 3-seed fitness. DO NOT.** Measured: with 8 seeds and per-seed std ≈2,000 ticks the standard error is
≈700, and **every** train-seed win of 500-1,500 ticks died on held-out seeds (six hypotheses, `JOURNAL` §5);
at horizon 9,000 the fitness **saturates** — generation 4's winner had per-seed `[7984, 9000, 9000]`, two of
three seeds capped, so its gain was a LOWER BOUND and every capped candidate ties **[MEASURED]** `JOURNAL` §12.
Rule: **never fewer than 20 paired held-out seeds for any claim; resample fitness seeds per generation; horizon
≥12,000 with a tie-breaker; report mean AND median AND min AND seeds-won.**

**N5 — Do not re-tune the reserve / spawn-margin axis.** Five separate arms, monotone dose-response, losing
9-36% in **both** independent seed halves (L1 −16.6%/−11.5%, L2 −24.5%/−8.8%, L3 −19.3%/−18.3%,
L4 −36.0%/−24.2%) **[MEASURED]** `JOURNAL` §18.1. The axis is closed; the mechanism (movement is the income
mechanism, births are load-bearing because mortality is by age) is *why* it is closed, and that reasoning
constrains the curriculum: **no scenario, and no residual bound, may be configured so that the composite's
expected income per agent-tick falls materially below the base's.**

**N6 — Do not infer a mechanism from reading the source.** The "facing makes predators close 40% slower"
claim was wrong in both magnitude and framing; the measured effect is **1.583 / 1.359 units/tick opening while
facing vs 0.126 open / 0.091 CLOSING while turned** (~12× the separation) **[MEASURED]** `JOURNAL` §18.3.
Every mechanism in this document marked **[INFER]** (notably the `N_pred ≈ 1e-3·t0` relation and the
`_render_biome_surface` cost) is a *thing to measure first*, not a fact.

**N7 — Do not compare across platforms.** 4,472 (macOS/ARM) vs 7,816 (Linux) ticks on the same seed and
controller **[MEASURED]** `JOURNAL` §6. Train and rank on Linux only.

---

## 6. Implementation checklist (what must exist before Step 1 of §2.5)

1. `Environment.semantic_state() -> dict` and `Environment.load_semantic_state(dict) -> None`, capturing exactly
   the fields in §2.1, rebuilding all six grids, clearing `agent_observations`. **No pygame surface capture.**
2. **Determinism proof**: capture→restore→roll 20 ticks == uninterrupted run, bit-identical action digest
   (the `462.3278` standard, ×4 in-process and ×4 in a fresh process).
3. `scenario_bank.jsonl` (one line per scenario: `world_seed, t0, family, snapshot_hash, curation_pass,
   production_per_1k, base_remaining_ticks`) plus the harvest driver and the ±25% curation gate.
4. `biome_map`/surface cache by `world_seed` + one reused `Environment` per worker (perf; see §2.3 note).
5. The PLR buffer (K=200/tier, staleness mix) + the exact-regret evaluator (`return_base` from the frozen
   snapshot, once, cached per scenario).
6. Tier pacer (promote ≥60% / demote <30% over the last 200 frontier episodes).
7. Residual head with the R7 input set; `residual_scale ∈ {0,0.25,0.5,1.0}`; bound schedule 10%→20% of speed,
   0.15→0.3 rad; critic burn-in to the `V_H1` regression before any policy update.
8. The diagnostics logger of §3.4 (including the passivity flag), on by default, every run.
9. `LANES.md` claim before the bank build; `tools/check_controller.py` provenance gate for the base snapshot and
   the residual weights.
10. KILL-2 and KILL-1 harnesses (§4.1), run and reported **before** the bank build.

## 7. Sources

**External (all URLs checked this session unless noted):**
- Prioritized Level Replay (PLR), Jiang/Grefenstette/Rocktäschel, ICML 2021 — <https://arxiv.org/abs/2010.03934>,
  code <https://github.com/facebookresearch/level-replay>
- Dual Curriculum Design / PLR⊥ theory, Replay-Guided Adversarial Environment Design, NeurIPS 2021 —
  <https://doi.org/10.48550/arxiv.2110.02439>
- PAIRED — Dennis et al. 2020, code <https://github.com/ucl-dark/paired>
- ACCEL — Parker-Holder et al., ICML 2022, <https://arxiv.org/abs/2203.01302>,
  <https://proceedings.mlr.press/v162/parker-holder22a.html>, code <https://github.com/facebookresearch/dcd>
- Stabilizing UED with a Learned Adversary — <https://arxiv.org/abs/2308.10797>
- ALP-GMM — Portelas et al. <https://arxiv.org/abs/1910.07224>, code <https://github.com/flowersteam/teachDeepRL>
- Reverse Curriculum Generation — Florensa et al., CoRL 2017 — <https://arxiv.org/abs/1707.05300>
- Backplay — Resnick et al. 2018 — <https://openreview.net/forum?id=H1xk8jAqKQ>
- Reverse-Forward Curriculum Learning — ICLR 2024 —
  <https://proceedings.iclr.cc/paper_files/paper/2024/file/cd062f8003e38f55dcb93df55b2683d6-Paper-Conference.pdf>
- SCOUT: Per-Context Reset Curricula — <https://arxiv.org/abs/2607.26417> (abstract-level only; full PDF not
  extracted)
- Go-Explore — <https://arxiv.org/abs/1901.10995>; First return, then explore, Nature 590:580-586 (2021) —
  <https://doi.org/10.1038/s41586-020-03157-9>
- Residual Policy Learning — Silver/Allen/Tenenbaum/Kaelbling — <https://arxiv.org/abs/1812.06298>,
  <https://k-r-allen.github.io/residual-policy-learning/>
- Potential-based reward shaping / policy invariance — Ng, Harada & Russell, ICML 1999 —
  <https://mlanthology.org/icml/1999/ng1999icml-policy>
- RUDDER — <https://arxiv.org/abs/1806.07857>
- DAgger — Ross, Gordon & Bagnell, AISTATS 2011 —
  <https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf>
- SMiRL — Berseth et al. — <https://arxiv.org/abs/1912.05510>, <https://sites.google.com/view/surpriseminimization>
- Lux AI Season 3, 1st place (EcoBangBang) —
  <https://www.kaggle.com/competitions/lux-ai-season-3/writeups/ecobangbang-ecobangbang-s-approach-yet-another-rl->
- Lux AI Season 3, 4th place (YumeNeko, IL + IL-as-RL-pretrain failure) —
  <https://kaggle.com/competitions/lux-ai-season-3/writeups/yumeneko-4th-place-solution-imitation-learning-app>
- Lux AI Season 3, 9th place (two-stage IL, UNet) —
  <https://kaggle.com/competitions/lux-ai-season-3/writeups/team-k-9th-place-solution>
- Lux AI Season 3, 14th place (MARL, reward-structure failure) —
  <https://kaggle.com/competitions/lux-ai-season-3/writeups/3comets-multi-agent-rl-silver-solution-by-3comets->
- Lux AI Season 3, xLSTM + self-play + curriculum + PPO (5th) — <https://epub.jku.at/download/pdf/12796831.pdf>
- Lux AI Season 3 specs — <https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md>

**Not re-verified this session:** "Warp RL 2025" and "RLPP on F1TENTH 2024", cited in `PIVOT_DESIGN.md` §2
without URLs; "SELFish" (Hahn et al., ALIFE 2019), "predator-prey swarming" (NJP 2023), "dilution of risk"
(arXiv 2209.06338) and "MAP-Elites with adaptive sampling" (GECCO 2019), which are cited in
`PLAN_H1PLUS_2026-09-18.md` §17.3 — treat those four as repo claims until checked.

**Internal (this repo):** `src/elements/environment.py`, `src/elements/predator.py`, `experiments/env_wrapper.py`,
`experiments/w_reach.py`, `experiments/w_deaths.py`, `best_controller.py`, `PIVOT_DESIGN.md`,
`JOURNAL_2026-09-18.md`, `PLAN_H1PLUS_2026-09-18.md`, `RL_JOURNAL.md`, `experiments/IMITATION_REPORT.md`,
`experiments/LANES.md`.

## 8. One-paragraph summary

Freeze H1. Build a scenario bank by rolling H1 forward on ~500 fresh world seeds and snapshotting the whole
world at `t0 ∈ {3,000, 4,500, 6,000, 7,500, 9,000}` (~99 core-minutes), then grow it into the ticks H1 never
reaches by re-harvesting from the current composite and, where that fails, by synthesising late states —
setting `time` to `t0` so the decay and predator-pressure transients are correct, burning in 600 ticks with the
fleet frozen, and admitting only states whose production is within ±25% of the analytic decay curve. Curate with
PLR's value-loss score plus an exact regret term in which **the frozen base plays the antagonist PAIRED cannot
give us**; pace `t0` outward at a 60%/30% win-rate frontier. Keep 20-30% of every batch at tick 0 so the
training distribution never drifts from the graded one. Reward with potential-based shaping against a
`V_H1` pre-trained critic, so the learning signal is *advantage over the base*, and keep income as a logged
constraint with a passivity tripwire. Give the residual the base's own hidden state and a frame stack, or it
repeats the failure that killed behaviour cloning. Truncate episodes at 2,500 ticks. Judge everything on ≥15 of
20 paired held-out real tick-0 seeds (α≈2%), ship on ≥28 of 40, with `residual_scale = 0` as the built-in
baseline arm and rollback. And run KILL-2 (can a critic predict `V_H1` at all?) and KILL-1 (does the residual's
own bounded action box contain *any* better behaviour?) before spending a single gram of GPU time.
