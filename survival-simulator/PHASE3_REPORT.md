# PHASE 3 REPORT — what watching/recording reveals vs what telemetry already knew

Source: 16 recorded episodes, deployed controller, **x86 (the graded platform)**, 18,000-tick horizon,
seeds 1500-1515. Recordings produced by `experiments/replay.py` (fidelity proven by
`experiments/_loyalty_check.py`: recorder vs the proven `env_wrapper` harness = **identical survival
ticks and score on all seeds tested, on x86**). Analysis: `experiments/replay_survey.py`.
Viewer: `experiments/viewer.py`. No controller was changed; nothing was deployed.

## 1. What is visually/quantitatively obvious

| observation | number (16 episodes) |
|---|---|
| Fleets end with ≤3 agents **while 42-129 fruit stand uneaten** | **15 of 16 episodes** |
| Share of a dead agent's LIFE spent in the <20% energy sprint-lockout | **median 54%** (45-66%) |
| Predation deaths occurring inside that lockout zone | **median 86%** (67-96%) |
| Predation deaths where a predator WAS in the dying agent's observations | **median 94%** (blind only 6%) |
| Population peak → collapse | peak ~20, collapse to ≤2 at **median tick 6,664** (3,310-9,590) |
| Ticks an agent spends blocked by a wall | median 60 (13-150) |
| Ticks choosing a direction away from a *visible* fruit | median 19 (10-24) |
| Oscillation (repeated sign-flipping turns) | median 0.2 — negligible |
| Final mean energy of the fleet | ~0 |

## 2. What the old telemetry already told us (confirmed, not new)

* The world is **access-limited, not energy-limited** (6,400+ energy standing at fleet death) - now
  reproduced on 15/16 fresh x86 episodes.
* Predation deaths cluster in the sprint-lockout zone (previously measured on 3 seeds: 92-96%; here
  median 86% across 16).
* Mid-game extinction is modal.

## 3. What the recordings add that aggregate telemetry did NOT have

1. **The fleet is chronically energy-poor, not occasionally unlucky.** Median agents spend **54% of
   their entire lives below 20% energy**. Previous telemetry showed end-state energy (~23) and death
   energies; it never showed that this is a *life-long* condition rather than an endgame state.
2. **Predation is not a detection failure.** In 94% of predation deaths the predator was already in
   the agent's own observations. So evasion geometry (E3) could never have fixed it: they see the
   threat and cannot outrun it, because they cannot sprint. This explains the E3 null result
   mechanistically rather than statistically.
3. **The collapse is a step function, not a decline.** Population peaks ~20 and is at ≤2 by median
   tick 6,664 - a fast cascade, which is what an age-cohort cliff plus a starvation floor looks like,
   not a slow attrition.
4. **Wall blocking and fruit-avoidance are real but small** (median 60 and 19 ticks per lifetime).
   Both were plausible visual hypotheses; the numbers say neither is a primary driver. This is the
   survey doing its job: it kills two attractive stories cheaply.
5. **The dying fleet is surrounded by food it does not eat**, with the population at 1 - i.e. the
   limiting factor is the ability to *convert standing fruit into energy fast enough*, not the supply.

## 4. Hypotheses these suggest (observation → hypothesis, NOT proof)

* **H-A: energy buffer is the defence against predation.** If 86% of predation deaths are in lockout,
  then keeping agent energy above the sprint threshold should cut predation deaths directly, with no
  change to evasion. Previously-tested "thrift" arms cut *income* and failed; the untested version is
  raising *income per agent* or lowering *travel cost per fruit*, not slowing down.
* **H-B: population overshoot.** ~20 agents share a fruit flux that decays as 0.5^(t/3000); the fleet
  collapses to 2 shortly after production has fallen ~4x. If the fleet is over-sized for its income,
  the fix is a *smaller well-fed* fleet - and note the earlier thin-relay failures shrank the fleet by
  constraining behaviour (losing income); this would shrink it by *not spending* energy on births that
  cannot be fed.
* **H-C: the fleet never banks.** Production decays exponentially, so late survival must be paid for by
  mid-game savings; 54% life-in-lockout says nothing was banked. Refuted "banking" arms were
  late-game-only relaxations of behaviour; this would be an early-game accumulation objective.

## 5. Which of these are quantitatively testable, and how

All three reduce to measurable quantities from recordings alone, before any controller change:
* life-fraction in lockout vs survival ticks (per episode, 16 points) - already computed;
* income vs movement-cost budget per agent (needs a per-tick energy ledger in the recorder);
* fleet size vs per-agent income at the collapse tick (correlation across the 16 episodes).

The next cheap step is therefore **not** a controller change: add the energy ledger to the recorder,
re-run the 16 episodes, and check whether income-per-agent collapses *before* the population does. If
it does, H-B/H-C become a targeted experiment; if it does not, the recording has falsified them for
free.

## 6. Honest limits

* 16 episodes, one controller, one platform. Fine for hypotheses, not for effect sizes.
* The recorder is a second implementation of the step loop: it is *proven* bit-identical on the seeds
  tested, but the loyalty check should be re-run after any change to `replay.py`.
* I cannot see the rendered frames myself (no vision on this model). The conclusions above come from
  the recorded data and the text renderer; the human watching the viewer is the one who can catch
  things neither channel encodes - which is exactly why Phase 3 exists.
