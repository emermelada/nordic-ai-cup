# Evidence extractor experiments — 19–20 September 2026

**Decision: retain the locked serving build.** Two GPU experiments were run. The
first, on the original examples alone, was a clear regression. The second added
external rationale supervision and **reversed that regression into a small gain
of +0.0064 raw**, which is below the pre-registered +0.03 substantial-gain gate
and inside its own confidence interval. Nothing was deployed.

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
