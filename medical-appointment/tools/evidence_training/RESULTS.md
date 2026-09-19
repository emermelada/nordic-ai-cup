# First GPU pilot — 19 September 2026

**Decision: retain the existing system.** The QA-pretrained encoder trained on the
original examples did not improve evidence localization. A longer run of the same
recipe is not supported by this result; no final all-data fit or serving integration
was performed.

## Experiment

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

## Measured comparison

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

## Validation and correction

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
- The instance became unavailable before final artifact retrieval. A subsequent
  start request reported that resources were unavailable. That request was
  followed by an explicit stop; Vast confirmed `actual_status=exited`,
  `cur_state=stopped`, `intended_status=stopped` at approximately 21:15 UTC.
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

The available examples are enough to run a cheap, falsifiable pilot. They did not
make this extractor competitive. This does not establish that external training
data cannot help, but it rules out treating this exact original-data recipe as a
demonstrated path to 0.9. The next training experiment should change the evidence
available to the model (for example, relevant externally annotated spans) and use
the same conversation-disjoint comparison before any deployment decision.
