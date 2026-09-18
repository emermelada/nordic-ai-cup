# PIVOT DESIGN — beyond the heuristic (drafted 2026-09-18, for approval before any work starts)

Prompt from the user: stop micro-tuning the heuristic; either use the best heuristic to seed a model and
improve it drastically, or start from scratch; research how others solved this; run many parallel
workstreams that talk to each other and route validation requests to the user; the agent orchestrates.

This document is the cleaned, improved, red-teamed version of that idea. Nothing is started until it is
approved.

---

## 1. Our own evidence: what the heuristic can and cannot do

- H1 (deployed) = ~848 local mean over 20 fresh seeds; official samples 768.68 and 918.33.
- Five separate spend-reducing changes were measured and ALL lost, 9-36%, monotone in how hard they
  constrained energy (banking ×4, sprint-reserve/spawn-margin ×4). The parameter axis is exhausted.
- Death mix (3 seeds, official horizon): starved 32-51%, eaten 23-41%, aged 16-27%. Income is
  ACCESS-limited: agents are blind 76-87% of ticks with 2,000-4,900 fruit energy standing in the world.
- Therefore the missing capability is not "better parameters". It is STATE-DEPENDENT BEHAVIOUR that
  changes across phases: mid-game income maximisation, end-game survival on a bank. A potential-field
  controller with 39 scalars expresses none of that.

### 1.1 The reachability arithmetic (MEASURED — it corrected this document's own first draft)
The first draft estimated the world's total production as ~24,000 energy and concluded a 3x shortfall.
That was wrong: it extrapolated from the FLEET'S INCOME instead of measuring the WORLD'S OUTPUT.
Measured with `w_reach.py` on the deployed controller (production = eaten + standing-delta; the identity is
exact because `score += fruit.energy/1000`):

- total production available over an 18,000-tick episode: **55,405 / 57,072 energy** (2 seeds)
- the world produces ~12-13 energy/tick early, halving every 3,000 ticks (the tail is small, ~7-8k)
- keeping ONE lineage alive — metabolism 0.1/tick + 100 per relay spawn per ~900 ticks — costs
  0.211 energy/tick = **3,800 energy over 18,000 ticks, i.e. 7% of what the world produces**
- and the fleet DIED at 4,624 / 4,896 ticks with **6,469 / 6,505 energy of fruit still standing**

CONCLUSION, the opposite of the draft: the target is NOT energy-limited — 1,800 is reachable with roughly
a 15x energy margin. The binding constraint is ACCESS (the fleet starves surrounded by food) plus survival
through the mid-game predation cascade (in these seeds the fleet went 19 -> 15 -> 6 -> 0 agents inside
~600 ticks, with predators rising 2 -> 4 -> 5).

This is consistent with every refutation in §1: the five thrift measures each cut the fleet's ability to
REACH food faster than they cut its costs. It also re-orders the portfolio — income-per-agent and
mid-game survival outrank anything about conserving energy.

## 2. Outside evidence: how this class of problem was actually solved

Closest published analogue is Kaggle/NeurIPS **Lux AI Season 3** — multi-agent, partial observability,
long horizon, public leaderboard:

- **1st place:** large-scale RL FROM SCRATCH (IMPALA + V-trace), self-play against past selves, a frozen
  "teacher" KL loss to prevent forgetting, and an opponent pool. ~20 BILLION steps. They also masked
  which of two models played (85% weak / 15% strong) to avoid leaking their best policy to imitators.
- **3rd place:** started rule-based, hit a wall, switched to imitation learning from the top teams'
  replays — "the IL-based bot outperformed my best rule-based agent" — but also "I know it's nearly
  impossible to train an IL agent to outperform its teacher".
- **4th place:** IL only, two UNets, heavy replay preprocessing. Explicitly tried the idea in the user's
  proposal: "I attempted to use the IL model as pretrained weight for RL. However, the RL approach did
  not succeed... results were either a complete collapse of the IL policy or no improvement over default
  behaviour. After 2-3 weeks of trial and error without success, I concluded that ... it was nearly
  impossible to train for a sufficient number of steps."
- **8th place:** IL ensemble with per-agent-ID conditioning to cope with differing teacher distributions.

Plus the robotics line of work on **residual RL** (Residual Policy Learning, Silver et al. 2018; RLPP on
F1TENTH 2024; Warp RL 2025): keep a classical controller FROZEN and learn only a small correction.
Properties that matter to us: the base policy need not be differentiable; the residual is
zero-initialised so the system starts as the known-good controller and can never be worse at step 0;
reported ~10x better sample efficiency than learning from scratch; and the correction is bounded so the
system cannot wander into unknown behaviour.

**Dreamer / world models** (latent imagination, value beyond the imagination horizon) are the standard
answer to long-horizon sample efficiency, but our bottleneck is behaviour and credit assignment in a
fast, tiny-state, deterministic sim — not sample count. Keep as a fallback, not a first move.

## 3. The cleaned plan of record

**W1 — RESIDUAL RL ON H1 (primary ML track).** Freeze H1 as the base. Learn a small, zero-initialised
residual (Δmove_distance, Δturn_angle) with PPO. Bounded: |Δdist| ≤ 20% of speed, |Δturn| ≤ 0.3 rad.
Spawn stays the base's rule (it was never imitable and it is not what needs fixing). Reward = the sim's
own score delta (survival) + a small income term, and BOTH are logged so passivity is detectable.
Why this and not the user's literal phrasing: it needs no clone (the base IS the heuristic, so the BC
failures are irrelevant), it cannot regress (init = H1 exactly), and it sidesteps the documented
IL-then-RL collapse, because there is no policy collapse to have.

**W2 — ENDGAME STRUCTURE (mechanism).** Search the 3-agent metabolism-only relay implied by §1.1.
Trigger on STATE (time reached AND visible-food collapse), not a fixed late tick — the four banking
attempts all had unreachable or wrongly-keyed windows. Objective: a dedicated "arrive at tick 9,000 with
a bank and a funded heir chain" score, not raw ticks.

**W3 — ACCESS/INCOME AT MATCHED PRICE (staged, ready).** Memory/ballistic-search, dispersion-limited
following, area-restricted search — priced to the baseline's own energy per tick (the earlier trial was
mis-priced at 2x, so it tested "search twice as expensive"). Plus biome preference (forest/grassland
over swamp/desert/river) since `biome` is observable. Starvation is the plurality death cause.

**W4 — EVASION GEOMETRY (running now).** Retreat-while-facing; the separation effect is already measured
(1.583 / 1.359 / 0.477 units/tick while facing vs −0.126 / +0.091 / +4.221 turned).

**W5 — FROM-SCRATCH MARL (exploratory, explicitly capped).** Per-agent role conditioning plus a
curriculum that samples hard late-game states as initial conditions. This is the 1st-place recipe minus
self-play (see risk 3). Only pursued if W1/W2 do not move the metric and compute allows.

**W6 — MEASUREMENT + VALIDATION LANE (cross-cutting, non-negotiable).** Every claim on ≥20 paired
held-out seeds at the official horizon, reported as mean AND median AND min AND seeds-won. The
validation queue is a file; the user runs validations and deploys.

## 4. Weaknesses found (stated before starting, not after)

1. **The user's literal idea — "teach a model then improve it drastically" — is the exact route that
   failed for a top-4 team** in the closest competition, and it failed in the same way our own
   BC→PPO attempt did (collapse, no improvement). It must be re-scoped to residual RL, where the base
   is never destroyed. This is the single most important correction in this document.
2. **IL cannot surpass its teacher** (3rd place's own words). IL alone cannot reach the target; it can
   only reproduce ~848.
3. **There is no adversary here**, so self-play/league training — the winner's key ingredient — is
   unavailable. It must be replaced by scenario/curriculum generation, which is weaker.
4. **Compute asymmetry: ~20 B steps was the winning recipe's scale.** Ours is a 4-vCPU VPS plus a Mac.
   From-scratch deep RL is therefore unlikely to win on its own; sample-efficient methods (residual,
   off-policy replay, curricula) are mandatory rather than nice-to-have.
5. **Measurement still binds.** ±100-150 per-run spread, official = mean of 3. A learned policy must
   beat 848 substantially to be provably better; anything smaller is unmeasurable and unshippable.
6. **Time: ~1.5 days.** So the portfolio must be 4-5 workstreams that can each deliver a validated
   artifact, not ten idea-streams. Parallelism is capped by validation capacity, not by agent count.
7. **Repo/VPS contention:** another agent has already re-run my sweep on the same box tonight. Needs a
   lane convention (a LANES file the agents read before launching) or work will clobber work.
8. **Reward hacking / the passivity optimum** is already measured in this project (ticks up, income
   flat). The residual must be rewarded on income as well as survival, and both logged.
9. **Residual limits:** additive residuals only translate the base action distribution; if the base is
   mis-calibrated (not merely biased) a bounded residual cannot fix it. Warp RL's geometric correction
   is the escape hatch if W1 plateaus.
10. **The endgame may require a fleet so thin that income collapses**, making the relay self-defeating.
    If Step 0's arithmetic says the maximum reachable score is well under 1,800, we retarget honestly:
    maximise the mean and take the best leaderboard position available, rather than chasing a number.

## 5. Order of work once approved

- **Step 0 (immediate, ~30 min):** compute the exact reachability ceiling with real production and
  standing-fruit numbers; publish "max reachable score" and the implied endgame burn. Everything else
  is then aimed at a known distance instead of a guess.
- Then W4 (already running) → W1 and W3 in parallel → W2 → W5 only if needed.

## 6. Orchestration protocol

- One workstream = one subagent, one git branch, one config family, one deliverable file.
- Compute lanes: max two VPS sweeps at once; a lane file records claims so agents do not collide.
- Validation queue: one entry per candidate, with paired evidence, expected delta, and the deploy
  command. The user owns validation and deploy; the orchestrator reviews every claim first.