# Route to 0.85 and 0.90 on evidence localization

**Update, 2026-09-20 06:05 CEST — the first step is done and validated.** A medoid vote
over three independently trained producers (stage B, the extractor, a new listwise span
ranker) returned **0.8307888** on the platform, +0.0140 over the locked build, taking hidden
mean tIoU from 0.69462 to **0.71798**. What this document called "the one thing that
validated" was the right family; what it got wrong was assuming the third voter had to be a
prompt variant. See [tools/evidence_training/RESULTS.md](tools/evidence_training/RESULTS.md).

Remaining distance, restated against the new baseline: **0.85 needs hidden tIoU 0.7500
(+0.032), 0.90 needs 0.8333 (+0.115)**. The candidate-pool oracle is 0.9243 and the
word-grid oracle 0.9859, so the room exists; nothing measured so far closes it, and the two
routes that could — a chooser trained to rank spans, and in-domain data with the right span
convention — are both now known to be harder than they looked (see the closing sections).

Written 2026-09-20, updated with the measured results of the mixed-source
extractor run. Builds on
[research/evidence_research_20260919/RESEARCH.md](research/evidence_research_20260919/RESEARCH.md)
and the saved control audit beside it. All figures are from the recomputed
runtime-faithful control on the 390-question public set.

## Measured on the platform, 2026-09-20: the answer half is finished

A validation run with every evidence span deliberately collapsed returned
**exactly 0.400**, and that score is `0.4 x accuracy` alone. So on the hidden set:

| Quantity | Value |
| --- | ---: |
| Answer accuracy | **1.0000** |
| Mean tIoU | **0.69462** |
| Score | 0.8167711209 |
| tIoU for 0.85 | 0.7500 (+0.055) |
| tIoU for 0.895 | 0.8250 (+0.130) |

**Spend nothing further on answers.** Accuracy is perfect, so the rescue
threshold, the phonetic-name rule and every yes-rate argument are closed, and the
0.4 term cannot grow. Each 0.01 of tIoU is worth 0.006 raw, and that is the only
currency left.

The hidden tIoU (0.6946) also sits below the public one (0.7183), so public
measurements read about 0.024 tIoU optimistic before any other bias.

## The headroom is real and it is concentrated

`raw = 0.4 × accuracy + 0.6 × tIoU`. Local accuracy is 1.000, so the whole
problem is tIoU. The control holds **139.7 of 195 IoU points**, leaving 55.3 on
the table, and they are not spread thin:

| Bucket | Questions | Points lost | Share |
| --- | ---: | ---: | ---: |
| Zero overlap — wrong occurrence or total miss | 21 | 21.0 | 38% |
| 0 < IoU < 0.5 — right region, wrong extent | 32 | 21.1 | 38% |
| 0.5 ≤ IoU < 0.9 — loose boundaries | 37 | 10.4 | 19% |
| IoU ≥ 0.9 — already fine | 105 | 4.2 | 8% |

**76% of the loss sits in 53 questions that are already in the right
neighbourhood.** That is a boundary and occurrence-selection problem, not a
comprehension problem.

What each target costs:

| Target | Required tIoU | Points to recover | Share of the leak |
| --- | ---: | ---: | ---: |
| 0.85 raw (local) | 0.7500 | 6.6 | 12% |
| 0.85 on the platform (local reads ≈0.013 higher) | 0.7717 | 10.9 | 20% |
| 0.90 raw | 0.8333 | 22.8 | 41% |

Three measured ceilings say the room exists:

- **Oracle over the two systems now in hand: 0.8029 tIoU -> 0.8817 raw, +0.0520
  over the control.** The trained extractor and the 27B evidence stage disagree
  on 71 of 195 questions (36 extractor wins, 35 control wins, 124 ties). A
  perfect chooser between two systems already built beats the gate by itself.
  This is measured, not estimated.

- **Candidate-pool oracle 0.8619 tIoU → 0.9171 raw.** The spans the pipeline
  already generates are worth 0.917 if the right one is picked. The selector
  currently extracts 0.716 from that pool. **That 0.146 tIoU gap is pure
  selection**, available without generating anything new.
- **Base-coordinate word-span oracle 0.9859 tIoU → 0.9915 raw.** The ASR grid
  is not the limit.

0.85 needs one fifth of the leak. The pool gap alone is three times that size.

## The one thing that validated: a medoid vote over diverse producers

Measured 2026-09-20 against the **current build** (`stageb-on-own-named-perq-v3`,
0.718340 tIoU / 0.831004 raw on 195 answered positives), not the older control.

The rule is parameter-free: among the candidate spans for a question, keep the one
with the greatest total temporal overlap with the others. Reproduce any row with
`python -m tools.evidence_training.ensemble`.

| Vote (always includes the build) | Raw | Gain | Conversation bootstrap 95% | Above zero |
| --- | ---: | ---: | --- | ---: |
| + extractor + 122B | 0.854756 | +0.023752 | [+0.013136, +0.034306] | 100% |
| + extractor + 9B (Mac MLX) | 0.846397 | +0.015393 | [+0.004331, +0.026075] | 100% |
| **+ extractor + retrieved variant** | **0.844766** | **+0.013763** | [+0.003530, +0.024334] | 100% |
| + extractor + no-draft variant | 0.842957 | +0.011953 | [+0.000381, +0.022805] | 98% |
| + retrieved + no-draft (no extractor) | 0.839157 | +0.008153 | [+0.000438, +0.016563] | 98% |
| + extractor + rerun of the same config | 0.835357 | +0.004353 | [+0.000131, +0.010215] | 99% |

**Diversity of the producer is what pays.** A different model is worth the most
(+0.024 for the 122B, +0.015 for a 9B), a different prompt variant about +0.012 to
+0.014, and a rerun of the same configuration almost nothing (+0.004). The trained
extractor is the most valuable single partner: dropping it from the best
deployable rule costs more than half the gain.

**The rule must be fixed in advance.** Choosing the voter subset by its own score
gives +0.016; the same search under nested leave-one-conversation-out validation
gives **-0.000589** with 47% of resamples above zero. With 39 conversations, free
subset choice is pure overfitting. The rules above are each pre-specified and
bootstrapped individually.

All voters share the identical answer pass (390/390 questions) and the same
`stageb-perq` mode, differing only by the `--variant` flag the pipeline already
implements, so a deployed vote reproduces these spans exactly.

### What it would take to deploy the +0.0138 version

1. A second stage-B pass with `--variant retrieved` on the same 27B, roughly
   doubling stage-B latency. The budget has room on paper — worst round trip is
   about 23 s against 60 s — but this must be measured, not assumed.
2. The extractor checkpoint served alongside vLLM (435M, about 1 GiB, small
   against the 27B's 61 GiB).
3. The medoid combination in the evidence stage, then end-to-end validation and a
   latency check before any platform attempt.

The 122B version scores higher but its weights were deleted, and at 74 GB it
cannot co-reside with the 27B in 96 GiB of VRAM, so it is a measurement rather
than an option.

## Mechanisms, in the order worth trying

### 1. Boundary refinement inside a local window

Keep the 27B's chosen span, hand the trained span extractor a window around it,
let it re-pick start and end *within that neighbourhood*.

**The 2026-09-20 run measured exactly the advantage this needs.** Where both
systems land somewhere real (167 questions), the extractor scores 0.8198 against
the control's 0.8024, and it cuts the 0 < IoU < 0.5 bucket from 32 to 26 while
lifting IoU >= 0.9 from 105 to 109. It is the better boundary model. What it is
*not* is a better occurrence finder: it fixes 6 of the 21 zero-overlap misses and
creates 7 new ones. Restricting it to a local window keeps the part that works
and discards the part that does not.

- Targets the 69 questions holding **31.5 points**.
- Cannot invent a new occurrence error, so the 105 good predictions barely move.
- Recovering 40–60% of those points is **+0.039 to +0.058 raw** — 0.85 territory
  on its own.
- Gold evidence is a *convention* ("minimal self-contained span"), and a
  convention is learnable from 195 real examples even where extraction from
  scratch is not. Prompting has never taught it; supervision can.

This is the highest-value use of the extractor because its downside is bounded
by construction.

### 2. Independent re-ranking of the existing candidate pool

Score every candidate in the saved pools with the trained extractor's start/end
log-probabilities and take the argmax.

- Attacks the **21 zero-overlap cases** directly: a span model sees every
  occurrence at once, which is exactly what occurrence selection needs.
- Capturing a third of the 0.146 tIoU pool gap is **+0.029 raw**; capturing a
  third of the measured two-system oracle is **+0.017 raw**. The simplest version
  — choose between the control's span and the extractor's span per question — is
  a two-way decision with a +0.052 ceiling and needs no new generation at all.
- A useful first probe is whether the extractor's own margin (its span score
  minus its CLS score, already computed in `decode_window`) predicts which system
  is right. If it does, the chooser is nearly free.
- The eight selection mechanisms that failed earlier (verification, probability
  re-ranking, multiple choice, reasoning, ensembling, question-likelihood,
  locate decomposition, post-hoc rules) were all the same model judging its own
  candidates. A separately trained model with a different inductive bias is a
  genuinely different signal, and it needs no new data — it runs over pools
  already on disk.

### 2b. Tested and rejected, 2026-09-20 — span geometry is not the selector

Combining the two systems from their *spans alone* was tried and does not work.
On the 195 positives, with the control at 0.716181 tIoU:

| Rule over the two predicted spans | Raw | Change |
| --- | ---: | ---: |
| Take the extractor whenever the spans overlap at all | 0.838195 | +0.008486 |
| Union of the two where they overlap | 0.842218 | +0.012509 |
| Intersection where they overlap | 0.825086 | −0.004623 |
| Overlap threshold chosen leave-one-conversation-out | 0.844454 | +0.014745 |
| **Same, with the rule family also chosen out-of-fold** | **0.827216** | **−0.002493** |

The last row is the honest one: bootstrap [−0.017281, +0.011083], 36% of
resamples above zero, and the procedure cannot even settle on a rule
(`gate@0.2` for 27 conversations, `gate_longer@0.0` for 12). Everything above it
is the cost of picking a rule family by looking at the full set. **With 39
conversations and 195 positives, a selector built from hand-crafted span
geometry will overfit.** The oracle's +0.0520 is real, but realizing it needs a
signal from inside a model, not the geometry of two boxes.

Span dilation is also dead: every amount hurts (−0.022 raw at ±0.1 s, −0.094 at
±0.5 s) and out-of-fold selection correctly picks zero. The median gold-minus-
control edge offset is **exactly 0.000 s on both sides** — when the control finds
the right region its boundaries are already right. The apparent length shortfall
(gold 3.21 s, control 2.98 s, extractor 2.78 s) is an artifact of the complete
misses, not systematic under-coverage.

**The decode-margin chooser was tested on 2026-09-20 and failed.**
`tools/evidence_training/rerank.py` scores both systems' spans under the same
model, CLS-normalized, and takes the argmax. On the 75 questions where the two
systems differ it picks the better span **46.7% of the time** — worse than chance.
The mean margin gap is +2.055 when the extractor is right and +1.869 when it is
wrong, which is noise: a model's confidence in its own argmax is not a calibrated
comparator. The rule scores +0.005870, below simply always trusting the extractor.

A working chooser therefore needs a model trained to *rank* candidate spans by
overlap with gold, not a re-used extractor. The medoid vote above sidesteps this
by using agreement among independent producers instead of any confidence score.

### 3. The extractor as the evidence stage

Measured on 2026-09-20: **0.8361 raw, +0.0064 over the control**, consistent
across folds (+0.0132, +0.0083, −0.0022), against the original-data pilot's
−0.0300. External pretraining repaired a clear regression, but the result sits
below the +0.03 gate and its interval [−0.0204, +0.0339] contains zero. As a
wholesale replacement it is not ready; as the component behind mechanisms 1 and
2 it is. Full numbers in
[tools/evidence_training/RESULTS.md](tools/evidence_training/RESULTS.md).

### 4. Answer side — closed

Measured at accuracy 1.0000 on the hidden validation set, so there is nothing
here at all. The reasoning below described an older build and no longer applies.
Each 1% of accuracy would be worth 0.004 raw. All that is established is
`false negatives − false positives = 3` on the hidden set, so the rescue
threshold is worth calibrating, but this is a top-up, not a route.

## What 0.90 takes

41% of the leak means fixing essentially every zero-overlap case *and* most of
the low-overlap group. That is the data program from the research doc, and it is
a multi-day build, not a deadline-day patch:

1. Split by consultation first; every derivative of a source stays in its fold.
2. Generate 1,000–2,000 short synthetic consultations with the intended evidence
   written **before** the surrounding dialogue, so exact offsets are known.
   Vary discourse structure — short confirmations, pronouns, joined facts,
   repeated plans, corrections, drug names, units, negative findings.
3. Filter by support *and* exact location recovery; semantic consistency alone
   cannot certify which occurrence was annotated.
4. Build hard negatives from the control's actual mistakes: the other true
   occurrence, over-long and shortened spans, facts belonging to a neighbouring
   question.
5. Train exact spans first, then preference learning on temporal-IoU
   preferences — the RadQA-DPO shape, which moved clinical extractive QA from
   63.6 to 77.5 F1.
   **Keep the span-length convention:** SIMORD's 16 in-domain dialogue questions,
   whose sentence-level provenance has a median of 43 words against the target's
   8, cost −0.0342 mean tIoU when added to the folds. Generated evidence must be
   minimal self-contained spans, or in-domain data will actively hurt.
6. Render a matched subset as speech and re-transcribe it to learn realistic ASR
   corruption. Not a first dependency.

Step 0 is done and it came back positive: external rationale supervision
(CoQA + MASH-QA, 66,279 windows) took dev macro word-IoU from 0.270 to 0.706 in
57 minutes, and that transferred to a real if small gain on actual consultations.
44k out-of-domain examples bought +0.036 raw over the data-starved pilot. The
question the data program answers is what in-domain examples with the right span
convention buy on top of that.

## The hard part nobody has solved yet

Locally, 53 of 195 questions carry 76% of the loss, and the research doc's
examples show what they are. "Will the treatment last two weeks?" has gold
"After a meal every day for two weeks." while the prediction "Sporanox, 100
milligrams daily for two weeks." scores zero. Both are true, both name two weeks,
and the annotators chose one. Likewise "Nothing abnormal to report." against
"Your chest and heart both sound normal."

These are **occurrence** failures, not comprehension or boundary failures, and the
preference they encode exists in exactly one place: the 195 annotated spans.
Generated data teaches clean localization, because a uniqueness filter keeps only
spans that are the sole answer to their question, which is precisely the easy
case. Nothing generated can teach which of two true passages an annotator picks.

That is the wall between 0.695 and 0.825, and it is why the mechanisms above are
worth single-digit thousandths while the target needs 0.130.

## How to verify anything here

Same protocol every time, or the number means nothing:

- Conversation-disjoint outer folds, inner split for selection, pooled outer
  predictions reported once.
- Bootstrap by consultation, not by question.
- Frozen answers and ASR timestamps; keep the official labels and denominator.
- Report raw, mean tIoU, zero-overlap count, IoU ≥ 0.9 count, and per-fold
  consistency. One good fold and two bad ones is not a gain.
- Gate at **+0.03 raw** before spending more compute, then fresh audio-to-
  response evaluation and latency check before any platform attempt.
- Never run an experiment on the box while a scored attempt is in flight;
  contention has already cost 0.016 once.

## Status, updated 2026-09-20 06:05 CEST

- Serving build: **0.8307887829** platform (three-producer medoid), zero errors, round
  trips 12-17 s of a 60 s budget. This is the thing to submit.
- Fallback: the locked **0.8167711209** build, five validations spread 0.0002.
- The earlier vote over stage-B *prompt variants* validated **0.8160730408 twice** and was
  reverted. Its honest out-of-fold estimate was +0.0138 and the platform returned -0.0007;
  replacing one prompt-variant voter with a separately trained model turned the same rule
  into +0.0140. **The voter's independence, not the rule, was the whole difference.**
- Two measurement traps found and recorded: a two-conversation development set saturates and
  selects epoch 1 (it faked a -0.0153 result), and averaging seeds into one voter lowers a
  vote because it removes the disagreement the vote feeds on.
- Mixed-source pretraining complete: 0.8361 raw on the folds (+0.0064), the
  checkpoint kept at `/workspace/medical-evidence-training-codex/runs/pretrain-mix-001/model`
  on Vast 51489967 (stopped; disk retained). See
  [tools/evidence_training/RESULTS.md](tools/evidence_training/RESULTS.md) and
  [tools/evidence_training/EXTERNAL_DATA.md](tools/evidence_training/EXTERNAL_DATA.md).
