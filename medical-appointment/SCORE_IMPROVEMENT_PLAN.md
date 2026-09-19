# Route to 0.85 and 0.90 on evidence localization

Written 2026-09-20, updated with the measured results of the mixed-source
extractor run. Builds on
[research/evidence_research_20260919/RESEARCH.md](research/evidence_research_20260919/RESEARCH.md)
and the saved control audit beside it. All figures are from the recomputed
runtime-faithful control on the 390-question public set.

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

**The one cheap shot left at the oracle** is the extractor's own decode margin —
its best span score minus its CLS score, already computed in `decode_window` and
currently discarded by `prediction_spans`. If that margin predicts which system
is right, the chooser is nearly free. It needs one short GPU inference pass over
the three fold checkpoints to emit margins alongside spans. Test that before
concluding selection is closed.

### 3. The extractor as the evidence stage

Measured on 2026-09-20: **0.8361 raw, +0.0064 over the control**, consistent
across folds (+0.0132, +0.0083, −0.0022), against the original-data pilot's
−0.0300. External pretraining repaired a clear regression, but the result sits
below the +0.03 gate and its interval [−0.0204, +0.0339] contains zero. As a
wholesale replacement it is not ready; as the component behind mechanisms 1 and
2 it is. Full numbers in
[tools/evidence_training/RESULTS.md](tools/evidence_training/RESULTS.md).

### 4. Answer side

Each 1% of accuracy is worth 0.004 raw. All that is established is
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

## Status when this was written

- Locked build: **0.8167711209** platform, five validations spread 0.0002.
  Unchanged and still the thing to submit unless a candidate clears the gate.
- Mixed-source pretraining complete: 0.8361 raw on the folds (+0.0064), the
  checkpoint kept at `/workspace/medical-evidence-training-codex/runs/pretrain-mix-001/model`
  on Vast 51489967 (stopped; disk retained). See
  [tools/evidence_training/RESULTS.md](tools/evidence_training/RESULTS.md) and
  [tools/evidence_training/EXTERNAL_DATA.md](tools/evidence_training/EXTERNAL_DATA.md).
