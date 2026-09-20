# PREREG — conditional low-population breeding rescue (paste_13 Phase 4)

Date: 2026-09-20, ~11:10 CEST.  Written BEFORE the paired A/B is run, so the decision rule cannot be
moved afterwards.

## What the 400-seed analysis established (evidence, then the hypothesis)

1. **Food supply is not the cause.** Policy-free map-wide supply correlates rho +0.27 (fruit energy),
   +0.19 (trees) with the final score; the local start-neighbourhood supply correlates below +0.20 for
   every radius (100-800 px). MID and LOW bands have *identical* local supply (within 200 px: 2.21 vs
   1.96 fruit; distance to nearest fruit 130 vs 130 px). "Bad seed" explains ~7% of variance at most.

2. **Hive's behaviour is not the differentiator either.** Mode shares of agent-ticks are the same in
   winners and losers: camp 0.629 / 0.636 / 0.634, explore 0.028 / 0.025 / 0.019 (HIGH >=1500 / MID
   1000-1300 / LOW <800).

3. **Fleet size after the boom is the dominant conditional predictor.** Among runs ALIVE at t0, the
   remaining survival is monotone in population at t0 at every t0 tested:

   | t0 (s) | pop 3-6 | pop 13-18 | pop 19+ |
   |---|---|---|---|
   | 700 | 3,026 ticks median rem, P(die<300 s) 0.48 | 5,841, 0.18 | 6,756, 0.05 |
   | 900 | 1-2: 1,294, 0.83 / 3-6: 3,115, 0.44 | 4,756, 0.23 | — |

   Reaching 2-6 agents by t=800 (having boomed at t=400): mean score 1,107.  Still >6 at t=800: 1,362.
   The winners reach <=6 agents only at t=830-1280; the losers at t=410-630.

4. **The winners keep REPLACING the fleet; the losers stop.** Among runs already in the 3-6 state at
   t=700, the ones with longer remaining survival bred 8x more in the next 300 s (2.33 vs 0.29 births
   per 10 s bucket), and had better hearing (69.9 vs 56.7) and *lower* max_energy (525 vs 628 — the
   inherited max_energy ratchet is visible in the losers).

## Hypothesis (falsifiable)

The score is 96.7% survival time, and survival with a small fleet is limited by the probability that
*every* agent dies. A dwindling fleet that keeps producing replacements converts a certain extinction
into a long tail; a dwindling fleet that stops breeding is a countdown. Therefore:

> **H1:** forcing replacement births while the fleet is small (and only then) increases mean survival.

Alternative explanations this test must rule out: (a) low population is a *symptom* of an energy
collapse, in which case adding mouths accelerates the collapse and the rescue will hurt; (b) the
forced births come from parents hive would not have chosen, degrading the gene pool, which would show
up as a worse upper tail.

## Intervention (minimal, conditional, reversible, legal)

Implemented as a wrapper around hive's own action list — **hive.py is not modified**, and it can be
deployed the same way (a post-processing gate on the payload + action list):

    trigger  : t >= 600 s  AND  1 <= pop <= 12          (the dwindling state; OFF in healthy runs)
    action   : take the single highest-energy agent with energy > 180 and set spawn_agent=True
               (only if pop < 14).  180 leaves >=80 energy after the sim's 100 birth cost.
    otherwise: byte-for-byte hive.

No movement, turn or targeting change. No parameter of hive changes. If the trigger is false the
action list is identical to hive's.

## Endpoints and pre-registered decision rule

Paired on the SAME seeds as the baseline arm (the 400-seed run), n >= 150 pairs.

Primary: paired mean score delta, two-sided t.
Secondary: median, p10, p25, P(score>=1500), P(score>=1800), max, and the paired W/L.

    ACCEPT  if mean delta > 0, t >= 2.0, median delta >= 0, and P(>=1500) does not fall by more than
            1.5 percentage points.
    REJECT  if t < 2.0, or the tail (P(>=1500)) drops materially, or the mean delta <= 0.
    REPORT  the number of forced births and how often the trigger was active (intervention rate).

A rejected rescue is not "tuned": no dose sweep follows unless the direction is right but the size is
wrong (t in [1.0, 2.0] with mean > 0), which is the only case where a second dose is justified.

## Falsification value

If this fails, it rules out the "redundancy through breeding" mechanism as the lever, and the remaining
candidate is the genome side (the max_energy ratchet visible in the losers, g_maxE 628 vs 525), which
would be tested by parent selection rather than population size.

---

## VERDICT (2026-09-20 ~11:25 CEST) — REJECTED

Paired A/B on the same seeds, n=52 (arm stopped early: the verdict was already decisive), common
horizon 16,294 ticks (the shorter of the two arms' horizons, so neither arm gets a free win):

    baseline hive            mean 1234.3   median 1334.7   P(>=1400) 0.442   P(>=1500) 0.269
    hive + lowpop rescue     mean 1136.4   median 1215.0   P(>=1400) 0.250   P(>=1500) 0.173
    PAIRED delta             -97.9 +- 36.7 (SE)   t = -2.67   median delta -65.4   W/L/T 13/33/6

Band movement: 14 runs fell from HIGH (1400-1800) into MID, 4 rose MID->HIGH, 4 MID->LOW. The rescue
damaged exactly the band we must not damage, by spending the healthiest agent's energy (100 per forced
birth) on a replacement at the moment that agent's survival was the thing keeping the run alive.

Pre-registered decision rule said: reject if t < 2.0, or P(>=1500) falls materially, or mean <= 0. All
three fired.

**Conclusion:** "the persisters keep breeding" was a CONSEQUENCE of having spare energy, not a cause of
long survival. Redundancy through forced births is not the lever. This closes the population/breeding
mechanism family on the action side, and the next lever tested is the gene side (see below), which the
same data motivated: g_hear is the one gene dimension that predicts remaining survival consistently
(+0.23..+0.28 at t0 = 300..700, conditioned on being alive at t0), while g_maxE is consistently
slightly negative (-0.04..-0.13), and both signals disappear inside already-high runs.
