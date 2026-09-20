# Evidence extractor experiments — 19–20 September 2026

**Decision: the three-producer medoid is deployed and validated at 0.8307888**, up from
the locked build's 0.8167711. What made the difference was not a better single evidence
model but a third *independent* one: a listwise span ranker, which loses to stage B on its
own and is worth +0.0176 raw out of fold as a voter. Details in
[the vote section](#span-ranker-and-the-three-producer-vote--20-september-2026).

Earlier in the same effort, two GPU experiments on the extractor alone: the first, on the
original examples only, was a clear regression; the second added external rationale
supervision and reversed it into +0.0064 raw, below the +0.03 gate and inside its own
interval. That extractor is now the vote's second producer, which is where its value was.

The second experiment's most useful output is not its headline: the trained
extractor and the 27B evidence stage **disagree on 71 of 195 questions, and an
oracle that picked the better of the two per question would score 0.8817 raw,
+0.0520 over the control**. The two systems fail differently, which is a measured
selector opportunity rather than an estimated one. See
[SCORE_IMPROVEMENT_PLAN.md](../../SCORE_IMPROVEMENT_PLAN.md).

## Platform results, 20 September 2026 — what actually transferred

Three builds went to the competition's validation endpoint. The locked build's
five earlier validations spread 0.0002, so differences above that are real.

| Build | Platform score | Against the locked build |
| --- | ---: | ---: |
| Locked (perq v3, rescue 0.24) | 0.8167711209 | — |
| Evidence vote (3 producers) | 0.8160730408 (twice, identical) | **−0.0006981** |
| Spans deliberately collapsed | 0.4000000000 | diagnostic only |

### The hidden score, decomposed

The third run replaced every evidence span with a worthless one, so its score is
`0.4 x accuracy` alone. It returned **exactly 0.400**:

| Quantity | Hidden validation set |
| --- | ---: |
| Answer accuracy | **1.0000** |
| Mean tIoU | **0.69462** |
| Score | 0.8167711209 |
| tIoU needed for 0.895 | **0.825** |

**The answer half is finished.** Accuracy is perfect, so the rescue threshold, the
phonetic rule and the yes-rate arguments have nothing left to win, and an earlier
note treating missed positives as the gap does not describe this build. Every
remaining point is evidence localization: 0.69462 -> 0.825 is +0.130 tIoU, and
each 0.01 of tIoU is worth 0.006 raw.

### Why the vote looked good locally and was not

End-to-end on the 39 public conversations the vote scored 0.850 against the
build's 0.832 (tIoU 0.720 -> 0.751, worst round trip 23.0 s -> 27.3 s of a 60 s
budget). **That measurement was invalid.** The extractor it deployed,
`fit-extractor-001`, is trained on all 39 public conversations, and it was then
scored on those same conversations. The honest out-of-fold estimate for the same
rule was +0.0138, and the platform returned −0.0007.

The mechanism worked exactly as designed, which is what makes the result
informative rather than a bug: zero evidence-budget skips, and the vote moved
spans on 28 of 78 conversations (50 unchanged, 22 with one span moved, 6 with two
or three). Those relocations simply were not better on unseen conversations.

Two rules follow, and the rest of this work should be read through them:

1. **Never score a model on conversations it trained on.** The out-of-fold number
   is the only one worth reporting; `--extra-data` now enforces fold discipline
   for generated rows, with tests pinning it.
2. **A local gain is a hypothesis, not a result.** Local +0.018 became platform
   −0.0007. Only the platform closes the question, and it costs one validation.

## Generated in-domain data, 20 September 2026 — it does not help

`generate_indomain.py` picked spans on the real ASR word grid and had the served
27B write the clinical yes/no question each one answers, keeping only spans the
passage answers alone and that the rest of the transcript does not: **342 verified
examples from 2,990 candidates** (11.4%), covering all 39 conversations. Training
rows joined only folds already holding their conversation.

Out-of-fold on the same three folds, against the current build's 0.718340 tIoU:

| Extractor training data | Mean tIoU | Raw | vs current build |
| --- | ---: | ---: | ---: |
| External pretraining only | **0.7270** | 0.8362 | +0.0052 |
| + generated, spans <= 12 words (221 rows) | 0.7096 | 0.8258 | −0.0052 |
| + generated, all spans (342 rows) | 0.6786 | 0.8072 | −0.0238 |

**Generated data hurt, and span length explains most of it.** Simply dropping the
rows longer than 12 words recovered +0.031 tIoU. A question written *from* a span
teaches the model to return that whole span, so generated gold ran to a median of
11 words against real gold's 8. This is the third time in one session that
training data with the wrong span length cost accuracy, after MASH-QA (median 55)
and SIMORD (median 43).

The residue is more interesting than the headline. Short-span generated data gave
the **best boundary precision of any configuration measured** — 113 questions at
IoU >= 0.9, against 109 for pretraining alone and 105 for the current build —
while producing the **worst localization**, 29 zero-overlap against 22 and 21.

That is the uniqueness filter showing through. Keeping only spans that are the
sole answer to their question means every generated example is an easy
localization and a precise boundary, so the corpus sharpens boundaries and
dilutes the 195 real examples that carry the occurrence convention. Generating
more of it cannot fix that: the preference between two true passages is recorded
nowhere except those 195 spans.

## Span ranker and the three-producer vote — 20 September 2026

The extractor scores a start and an end independently, so nothing in it can express a
property of the span as a whole -- and "minimal self-contained span" is exactly such a
property. `tools/evidence_training/ranker.py` adds a listwise ranker that scores each
enumerated candidate as a unit from its start, end, mean and width representations, trained
directly on temporal IoU.

**The pool.** Sentence runs of one to four sentences plus comma-delimited clause spans,
capped at 40 words. On the 195 public positives its oracle is **0.9243 mean tIoU (0.9546
raw)**, so the pool is not the constraint; selection inside it is. Adding clause cuts at
"and"/"so" would raise the oracle to 0.9389, and was left out rather than chosen by its
effect on the folds.

**Pretraining.** 30,000 CoQA and MASH-QA rationale questions in the same listwise form,
one epoch (10 minutes on the RTX PRO 6000), starting from the extraction-pretrained
encoder. External development word-span IoU over the pool reached 0.5584.

**Alone it does not beat stage B.** Out of fold, three seeds:

| Seed | Mean tIoU | Raw | vs control |
| --- | ---: | ---: | ---: |
| 17 | 0.718385 | 0.831031 | +0.001322 |
| 18 | 0.693673 | 0.816204 | −0.013505 |
| 19 | 0.707205 | 0.824323 | −0.005386 |

**As a third voter it is worth an order of magnitude more.** The parameter-free medoid of
stage B, the trained extractor and the ranker -- keep the span with the greatest total
overlap with the others:

| Voters | Raw | Gain | Conversation bootstrap 95% | Above zero |
| --- | ---: | ---: | --- | ---: |
| stage B + extractor + ranker (seed 17) | 0.850184 | **+0.020476** | [+0.007924, +0.033806] | 100% |
| stage B + extractor + ranker (seed 19) | 0.849098 | +0.019389 | [+0.001777, +0.036941] | 98% |
| stage B + extractor + ranker (seed 18) | 0.842495 | +0.012787 | [−0.004420, +0.029491] | 93% |
| + retrieved variant as a fourth voter | 0.845287 | +0.015579 | [+0.004981, +0.026464] | 100% |
| + no-draft variant as a fourth voter | 0.842753 | +0.013044 | [+0.004608, +0.021715] | 100% |
| three seeds' scores averaged into one voter | 0.841321 | +0.011612 | [−0.004194, +0.026170] | 93% |
| three seeds as three separate voters | 0.835514 | +0.005805 | [−0.016480, +0.027635] | 69% |

Mean over seeds: **+0.0176 raw**. Leave-one-conversation-out selection over six rule
families (ranker argmax, argmax restricted to candidates overlapping stage B, two local
windows, the medoid, and a score-gated fallback) picks the medoid on all 39 splits, so the
rule is not a family chosen by looking at the answer. Zero-overlap questions fall 21 -> 20
and IoU >= 0.9 rises 105 -> 116.

Two results worth keeping:

- **Averaging seeds is the wrong ensemble here.** A score-averaged ranker agrees with the
  other producers more, and the vote pays for disagreement: +0.0116 against +0.0176 for a
  single seed. Diversity, not accuracy, is what the third voter contributes.
- **More voters is not better.** Four and five producers both score below three.

### A second encoder family: independence delivered, and it still lost

The redundancy measurement said the extractor and the DeBERTa ranker are the weak link, so the
same listwise recipe was built on `deepset/roberta-large-squad2` — a different encoder family,
different pretraining corpus, different tokenizer — to replace one of them.

**The independence was real.** The RoBERTa ranker agrees with the extractor within 0.9 on
61.0-65.6% of questions, against the DeBERTa ranker's 70.3-72.3%, and the four-way oracle rises
to 0.8957-0.9013 raw from the three-way 0.8913. It does find spans the others miss.

**It still lost, on every seed and every combination:**

| Producer set | Mean gain (3 seeds) | Worst seed |
| --- | ---: | ---: |
| **stage B + extractor + DeBERTa ranker (deployed)** | **+0.017551** | +0.012787 |
| All four | +0.013683 | +0.009063 |
| stage B + DeBERTa ranker + RoBERTa ranker | +0.009424 | +0.003750 |
| stage B + extractor + RoBERTa ranker | +0.008332 | +0.006570 |

The reason is quality, not independence: standalone out-of-fold mean tIoU 0.6628 against the
DeBERTa ranker's 0.7064 and the extractor's 0.7270. **A 0.044 quality deficit outweighed a 6-9
point independence gain.** With BM25 (0.4038 standalone) the same trade was catastrophic, so
the exchange rate is now bracketed from both sides: at roughly 0.72 a voter pays, at 0.66 it
does not, and at 0.40 it is destructive.

One more instance of the external proxy lying: RoBERTa scored **better** on the external
development metric (0.5658 against DeBERTa's 0.5584) and worse on the actual task by 0.044.
That metric has now mispredicted transfer three times — SIMORD, the second DeBERTa epoch, and
here.

**Conclusion: the vote is finished as a source of gains.** Both structurally motivated moves
have been tried and measured — a better third voter (platform: −0.0007) and a more independent
third voter (out of fold: −0.008) — and the whole cheap rule space around them is closed. The
remaining 13.4 points of selection headroom need a chooser trained on more than 98
disagreement examples, and the 14 unreachable questions need a producer with a different view
of the audio. Both are the data program, not a rule.

### The offline sweep after 0.8307888: what is left and what is closed

With the three producers' out-of-fold spans on disk, a lot of ideas cost nothing to test.
Almost all of them are closed. Structure first — the medoid's error is not spread evenly:

| Group | Questions | Medoid tIoU | Oracle over the three | Points left |
| --- | ---: | ---: | ---: | ---: |
| All three producers agree (within 0.9) | 97 | 0.8182 | 0.8182 | 0.0 |
| They disagree | 98 | 0.6831 | 0.8194 | **13.4** |

**Agreement is a correctness signal**: where the three agree the medoid already equals the
oracle, and where they disagree it captures about a third of what is there. On the 98
disagreements the sole-best producer is nearly uniform — stage B 20, extractor 10, ranker 11,
57 ties — which is exactly why picking the middle one leaves 13.4 points (+0.041 raw) behind.
Separately, **14 questions have no producer in the right region at all**, 29% of the
remaining leak, and no selection rule can reach those.

Producer redundancy, measured pairwise (seed 17):

| Pair | Agree within 0.9 | Disjoint | Pair oracle |
| --- | ---: | ---: | ---: |
| stage B + extractor | 63.1% | 7.7% | 0.802914 |
| stage B + ranker | 57.9% | 5.6% | 0.791963 |
| **extractor + ranker** | **70.3%** | 6.2% | **0.790308** |

The extractor and the ranker are the redundant pair — the same base encoder and the same
external corpus — and the three-way oracle is 0.8188 tIoU, **0.8913 raw**. So the room is in
replacing one of those two with something genuinely independent.

What was tested and did not clear the deployed rule (all out of fold, three seeds, mean gain):

| Idea | Mean gain | Verdict |
| --- | ---: | --- |
| Deployed: medoid of the three | +0.0176 | baseline for this table |
| Componentwise median of start and end | +0.0198 | better on 3/3 seeds, by +0.002 — the same margin the platform rejected in round two |
| Pair union / pair intersection / pair medoid | +0.0157 to +0.0183 | indistinguishable |
| Componentwise **mean** | −0.0202 | one outlier drags the span; never average spans |
| Fourth voter (retrieved variant) everywhere | +0.0148 | dilutes |
| Fourth voter only where the three disagree | +0.0149 | **targeting changes nothing**: a fourth span rarely moves a medoid the other three already agree on, so the dilution is on the disagreements themselves |
| Five voters on the disagreements | +0.0182 | within noise, more machinery |
| Non-neural lexical producer (BM25 over the pool) | −0.0002 to +0.0162 | **too weak**: 0.4038 tIoU standalone against the neural producers' 0.72 |
| Ranker score as a comparator between producers | −0.0054 | confidence is not a comparator |

Two refinements to the diversity rule come out of this:

- **A voter must be independent *and* comparable in quality.** The lexical producer is the
  most independent thing available — it agrees with stage B on 33.8% of questions against the
  ranker's 57.9% — and it still hurts, because at 0.40 tIoU it pulls the medoid off good
  spans. Independence is necessary, not sufficient.
- **Nested selection between near-identical rules costs more than it gains.** Choosing among
  these six families leave-one-conversation-out landed at +0.0218, +0.0124 and +0.0125 across
  seeds, below simply fixing the median rule (+0.0198). When candidate rules differ by less
  than the seed spread, pick one on principle and stop measuring.

### A two-conversation development set cannot select a checkpoint

The first CV run read **−0.0153 raw** and the reason was entirely procedural: folds 1 and 2
selected epoch 1, because two development conversations saturate immediately (fold 1's dev
tIoU was 0.9371 at the first evaluation and never beaten). Fold 0 trained seven epochs and
gained +0.0103. `--select final` trains a fixed eight epochs and uses those two
conversations for training instead, which turned fold 1 from −0.0333 to +0.0111.

Anything selected on two conversations in this project should be treated the same way.

### It transferred: 0.8307888 on the platform

Validated 2026-09-20 04:00 UTC, zero errors, 4m49s for the 19 hidden conversations.

| Build | Platform score | Against the locked build |
| --- | ---: | ---: |
| Locked (perq v3, rescue 0.24) | 0.8167711209 | — |
| Evidence vote over prompt variants (reverted) | 0.8160730408 | −0.0006981 |
| **stage B + extractor + ranker medoid** | **0.8307887829** | **+0.0140177** |

Accuracy on the hidden set is 1.0000, so hidden mean tIoU moved **0.69462 -> 0.71798**,
+0.0234. The out-of-fold estimate for the deployed seed was +0.0205 and the mean over seeds
+0.0176, so this transferred at about 70-80% of its local estimate -- unlike the earlier
prompt-variant vote, whose +0.0138 became −0.0007. Round trips on the hidden set were
12-17 s against the 60 s budget, and the vote moved 0-4 spans per conversation.

0.85 needs hidden tIoU 0.7500 and 0.90 needs 0.8333, so this closes about a sixth of the
distance to 0.85 and a fifth of nothing to 0.90; the candidate-pool oracle says the
remaining headroom exists, but capturing it needs a better selector, not a bigger pool.

### Round two: a second pretraining epoch, and what a better third voter is worth

The first ranker was pretrained on 30,000 of the 43,812 available external questions for one
epoch. A second epoch over all of them moved external development word-span IoU only
0.5584 -> 0.5609, but it moved the fold results clearly, and it moved the *worst* seed most:

| | Standalone mean tIoU (3 seeds) | Medoid gain (3 seeds) | Worst seed |
| --- | --- | --- | ---: |
| One epoch, 30k questions | 0.7184 / 0.6937 / 0.7072 | +0.0205 / +0.0128 / +0.0194 | +0.0128 |
| **Two epochs, 43k questions** | **0.7264 / 0.7186 / 0.7118** | **+0.0225 / +0.0194 / +0.0191** | **+0.0191** |

Mean medoid gain +0.0176 -> **+0.0203**, and the spread across seeds narrows from 0.0077 to
0.0034. The standalone ranker is still no better than stage B on two of three seeds, so the
vote remains the only way to collect it.

That a nearly flat external metric (+0.0025) produced a visible fold improvement (+0.013
standalone, +0.003 in the vote) is worth remembering: the external development score is a
proxy for the transfer, not a measure of it, and it saturates well before the transfer does.

**And the platform said no.** Served and validated 2026-09-20 04:29 UTC, zero errors:

| Third voter | Out-of-fold gain (3 seeds) | Platform |
| --- | ---: | ---: |
| One epoch, 30k questions (`rank-fit-001`) | +0.0176 mean, +0.0205 seed 17 | **0.8307887829** |
| Two epochs, 43k questions (`rank2-fit-001`) | +0.0203 mean, +0.0225 seed 17 | 0.8301335567 |

**−0.0006552.** The build with the better ranker *and* the better out-of-fold estimate lost,
and the platform resolves differences an order of magnitude smaller than this, so it is a
real if small loss. `rank-fit-001` was restored.

This is the third measurement of the same mechanism, and together they are the clearest
statement of what a vote rewards:

| Change to the third voter | Effect on the vote |
| --- | ---: |
| Average three seeds' scores (most accurate single voter) | +0.0116 |
| Two epochs of pretraining (more accurate) | −0.0007 on the platform |
| One epoch of pretraining (less accurate, more independent) | +0.0140 on the platform |

**Making the third voter better makes the vote worse**, because accuracy buys agreement and
the medoid is paid in disagreement. Do not tune a voter on its own score. The corollary for
anyone extending this: the next gain is a *fourth kind* of producer, not a better third one,
and four producers already measured below three, so it has to replace one.

### The comparator rule fails, which sharpens why the medoid works

Scoring all three producers' spans with the ranker and taking its argmax scores +0.0013,
−0.0136 and −0.0054 across the three seeds -- indistinguishable from simply trusting the
ranker, because its argmax usually *is* the highest-scoring of the three. A model's score
over its own hypothesis space is not a comparator between systems; this is the same result
the decode-margin chooser gave. **Agreement between producers is the signal; confidence is
not.**

## Second experiment — external rationale pretraining, 20 September 2026

Intermediate training on 43,812 checked question/evidence pairs from CoQA and
MASH-QA (66,279 windows; see [EXTERNAL_DATA.md](EXTERNAL_DATA.md)), then the same
conversation-disjoint folds and settings as the pilot.

Pretraining: two epochs, effective batch 16 as 8 x accumulation 2, learning rate
1e-5 with 6% warmup and linear decay, 57.1 minutes on one RTX PRO 6000. External
development macro word-span IoU rose **0.2699 -> 0.7059** (CoQA 0.6678, MASH-QA
0.7440). The selected checkpoint is the epoch-2 mid-point, chosen by an
evaluation inside the epoch; the end-of-epoch checkpoint scored 0.7054.

| Comparison | Mean tIoU | Raw | Change vs control | Conversation bootstrap 95% |
| --- | ---: | ---: | ---: | --- |
| Frozen control | 0.7161810760 | 0.8297086456 | — | — |
| Pilot, original data only | 0.6662128340 | 0.7997277004 | −0.0299809452 | [−0.0537, −0.0065] |
| **Pretrained initialization** | **0.7269130554** | **0.8361478333** | **+0.0064391876** | [−0.0204, +0.0339] |
| Pretrained + SIMORD dialogue | 0.6926750577 | 0.8156050346 | −0.0141036110 | [−0.0361, +0.0082] |

Per-fold change for the pretrained run was +0.013210, +0.008309 and −0.002213,
against the pilot's −0.000473, −0.049319 and −0.038481. The repair is consistent
across folds, not one lucky fold.

### What actually changed, and what did not

| Evidence quality bucket | Control | Pretrained extractor |
| --- | ---: | ---: |
| Zero overlap | 21 | 22 |
| 0 < IoU < 0.5 | 32 | 26 |
| 0.5 <= IoU < 0.9 | 37 | 38 |
| IoU >= 0.9 | 105 | 109 |

**Boundaries improved; occurrence selection did not.** On the 167 questions where
both systems land somewhere real, mean tIoU rises 0.8024 -> 0.8198. But the
extractor fixes 6 of the control's 21 zero-overlap misses while creating 7 new
ones — a wash. The trained model is the better boundary model and is not yet a
better occurrence finder, which is exactly the split that decides how it should
be used.

### Adding SIMORD dialogue supervision hurt

The 16 medical-dialogue training questions cost **−0.0342 mean tIoU** and broke
fold consistency (+0.020142, −0.022401, −0.038999). Their sentence-level
provenance has a median span of 43 words against the competition's median of 8,
so the only in-domain dialogue data available teaches the wrong span length. Long
external rationales are harmful here even when they are in-domain; this is why
the preparation filter caps evidence at 48 words.

### Throughput

Measured on the real mixture, effective batch 16 throughout:

| Micro-batch x accumulation | Seconds per update | Peak CUDA | Minutes per epoch |
| --- | ---: | ---: | ---: |
| 4 x 4 (pilot shape) | 0.421 | 13.9 GiB | 29.0 |
| **8 x 2** | **0.351** | 20.6 GiB | **24.3** |
| 16 x 1 | 0.370 | 32.3 GiB | 25.5 |

Total GPU session was about 1h20m including benchmarks, both CV runs and the
vLLM restart, roughly $2 of instance time.

## First GPU pilot — 19 September 2026

### Experiment

- Model: `deepset/deberta-v3-large-squad2`, approximately 435M parameters.
- Hardware: one RTX PRO 6000 Blackwell Max-Q, 96 GB.
- Data: 390 questions / 39 conversations / 195 positive spans; no external or
  generated examples.
- Three conversation-disjoint outer folds, each with 24 training, two development
  and 13 test conversations. Best epoch selected using inner-development tIoU.
- Three epochs maximum, learning rate 1e-5, batch four, accumulation four,
  BF16 autocast with FP32 weights, 512-token windows and stride 192.
- Boolean answers and ASR timestamps frozen to the saved runtime-faithful control.
  The 1.0 answer accuracy below belongs to that control, not a newly trained
  classifier. These public examples were previously inspected; this is an
  exploratory comparison, not hidden-platform validation.

### Measured comparison

The completed remote `runs/cv-002/report.json` was read during the session:

| Metric | Existing control | Trained extractor |
| --- | ---: | ---: |
| Questions scored | 390 | 390 |
| Frozen answer accuracy | 1.000000 | 1.000000 |
| Mean tIoU, all 195 positives | 0.7161810760 | 0.6662128340 |
| Raw score (0.4 accuracy + 0.6 tIoU) | 0.8297086456 | 0.7997277004 |

Raw change: **−0.0299809452**. The paired conversation-bootstrap 95% interval
(2,000 resamples) was **[−0.0537246546, −0.0065001795]**. There were 24 positive
questions with zero overlap. All three outer folds completed in about 97 seconds.

| Outer fold | Selected epoch | Raw score | Change versus control |
| --- | ---: | ---: | ---: |
| 0 | 2 | 0.8614257661 | −0.0004734085 |
| 1 | 3 | 0.7629403461 | −0.0493188538 |
| 2 | 2 | 0.7780802309 | −0.0384808989 |

The first five-update GPU benchmark used about 13.92 GiB peak allocated memory
and averaged 0.465 seconds per optimizer update after warmup. This benchmark
preceded the tokenizer correction described below. A repeat was requested, but
its completion has not been recovered; do not treat it as verified.

### Validation and correction

Eleven local tests passed, including real tiny-model CPU training, checkpoint
reload and complete three-fold scoring. Dataset export reproduces the existing
control score exactly. Gold-to-word mapping has mean attainable tIoU 0.985899.

The first GPU CV attempt failed before training because the tokenizer's pair
overflow path truncated the longest consultation. The corrected implementation
constructs windows from the full context and checks that every transcript word
remains reachable. A GPU preflight covered all 390 questions in 460 windows, and
the corrected three-fold run completed. The failed attempt remains on disk.

## Artifact and instance status

- Branch: `codex/evidence-extractor-training`.
- Local preparation and test artifacts: `runs/evidence-training-20260919/`.
- Remote workspace: `/workspace/medical-evidence-training-codex`.
- Completed remote evaluation: `runs/cv-002/`, including reports, manifests, logs
  and three selected checkpoints. The detailed report has **not been copied back
  locally**; the numeric observations above preserve the session's readout.
- **Recovered 20 September**: the pilot reports, manifests and logs are now at
  `runs/evidence-external-20260919/gpu-artifacts/pilot-runs/`, and the second
  experiment's artifacts at `runs/evidence-external-20260919/gpu-artifacts/external-run/`.
  The pretrained checkpoint (1.7 GB) stays on the box at
  `/workspace/medical-evidence-training-codex/runs/pretrain-mix-001/model`.
- vLLM was stopped for exclusive GPU use during the second experiment and
  restarted afterwards; it was verified serving `qwen` before the instance was
  stopped. No serving configuration was changed.
- GPU compute billing is zero while stopped. Retained instance storage was quoted
  at approximately $0.0292/hour. The instance was not destroyed.
- Only the explicitly authorized Qwen3.5-122B-A10B-GPTQ-Int4 model directory was
  deleted. The existing Qwen3.8-27B model was preserved. The vLLM server was paused
  for training; a restoration command was issued after the final benchmark, but
  its completion was not confirmed before the instance stopped.

When this same instance can next start, recover `runs/cv-002` and its metadata
before doing further experiments, and verify serving status if serving is needed.
The checkpoints can remain remote; copying reports and logs does not require
downloading model weights. Stop the instance again after retrieval.

## Implication

The pilot's failure was data scarcity, not the wrong architecture. Adding 43,812
external rationale examples moved the same recipe from −0.0300 to +0.0064 raw, a
swing of +0.0364, and made it consistent across folds. That validates supervised
span extraction as a direction while falling short of the gate that would justify
touching a locked, validated build.

The measured next step is selection rather than more pretraining: the two systems
already in hand have an oracle combination worth +0.0520 raw, and the extractor's
advantage is specifically in boundaries, where a local-window refinement cannot
introduce new occurrence errors. Both routes are specified in
[SCORE_IMPROVEMENT_PLAN.md](../../SCORE_IMPROVEMENT_PLAN.md).
