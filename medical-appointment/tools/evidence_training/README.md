# Evidence extractor training

An isolated experiment using `deepset/deberta-v3-large-squad2`. It learns word-span
positions from the supplied annotations. Existing boolean answers, ASR timestamps,
serving code, models and deployment settings remain unchanged. The trainer makes
no hosted inference calls and never removes other files or models.

The first GPU pilot is complete: **0.799728 raw versus 0.829709 for the frozen
control**. This candidate is not recommended for deployment. See
[the measured results](RESULTS.md).

A second stage adds public rationale supervision: 43,812 checked question/evidence
pairs from CoQA and MASH-QA pretrain the extractor before the competition folds,
and SIMORD's 16 medical-dialogue questions adapt each fold. The corpus, licences,
filters and measured cost are in [EXTERNAL_DATA.md](EXTERNAL_DATA.md).

## Prepare the original data

From the `medical-appointment` directory:

```bash
.venv311/bin/python -m tools.evidence_training.data \
  --output runs/evidence-training-20260919/dataset.json
```

Use a new output path for every preparation/run; existing artifacts are not replaced.
Defaults use `tools/gpu_eval/inputs_base.json` and the saved runtime-faithful
`stageb-on-own-resume-control.json` under `runs/resume-20260919-0325/faithful-results/`.
Override with `--inputs` and `--control` when explicitly choosing a different control.

The export has 390 questions from 39 conversations, including 195 positive spans.
It verifies complete baseline coverage and preserves word occurrence identity,
even when a quotation occurs twice. Gold endpoints maximize temporal overlap on
the original ASR word grid; the real export's mean attainable tIoU is 0.985899.
Gold timestamps are never replaced by these approximate training targets in scoring.

Three outer folds contain 13 conversations each. Two additional conversations are
reserved from each training fold for checkpoint selection; the remaining 24 train
the model. The exported split manifest is authoritative. All questions and windows
from one conversation stay together. These are exploratory folds on previously
inspected public data, not a fresh competition holdout.

## Dedicated NVIDIA environment

Copy this package and the prepared dataset to a **new** directory on the existing
GPU host, for example `/workspace/medical-evidence-training-codex`. Do not overwrite
the deployed project. Reuse its CUDA-compatible PyTorch through an isolated venv:

```bash
python3 -m venv --system-site-packages /workspace/medical-evidence-training-codex/.venv
/workspace/medical-evidence-training-codex/.venv/bin/python -m pip install \
  -r /workspace/medical-evidence-training-codex/tools/evidence_training/requirements.txt
cd /workspace/medical-evidence-training-codex
export HF_HOME=/workspace/medical-evidence-training-codex/hf-cache
export TOKENIZERS_PARALLELISM=false
```

The default model is about 435M parameters. The host's running vLLM server may
reserve most GPU memory; the trainer requires 16 GiB free by default and fails
explicitly rather than stopping another process. Obtain exclusive GPU use before
training and restore any paused serving process afterwards. The new cache contains
only this experiment's downloads. No 122B cleanup is automatic.

## Short benchmark first

```bash
timeout --signal=INT --kill-after=30s 10m .venv/bin/python -m tools.evidence_training.train \
  --data dataset.json --output runs/benchmark-001 --device cuda --mode benchmark \
  --benchmark-steps 5 --batch-size 4 --accumulation 4 --max-seconds 540
```

This performs actual forward/backward/optimizer updates. `benchmark.json` reports
step time, peak allocated CUDA memory and an estimated training duration per fold.
The estimate excludes download, evaluation and checkpoint writing. No benchmark
checkpoint is promoted or evaluated as a trained candidate.

## External rationale data

`fetch_external.py` downloads the raw corpora (text only, hashes recorded) and
`prepare_external.py` converts, decontaminates and fully tokenizes them into one
bundle. Both are described, with counts and commands, in
[EXTERNAL_DATA.md](EXTERNAL_DATA.md). Preparation needs no GPU; do not start the
billed instance for it.

## Intermediate pretraining

```bash
.venv/bin/python -m tools.evidence_training.train \
  --data dataset.json --external-data external.json --output runs/pretrain-mix-001 \
  --mode pretrain --epochs 2 --evals-per-epoch 2 --patience 3 \
  --batch-size 8 --accumulation 2 --eval-batch-size 32 --max-seconds 6000 --min-free-gb 60
```

`--external-data` is refused unless the bundle was decontaminated against this
exact `dataset.json`. Pretraining trains on CoQA and MASH-QA only; epochs run for
tens of minutes, so `--evals-per-epoch` selects inside them. The selection metric
is the **unweighted mean of per-source positive word-span IoU** on the external
development split, so the larger general corpus cannot hide a collapse on the
medical one. It is not temporal IoU and not the competition score. The learning
rate warms up over 6% of planned updates and decays linearly. `model/` holds the
best checkpoint, including the untrained epoch zero if nothing beats it.

## Original-data pilot

```bash
timeout --signal=INT --kill-after=30s 45m .venv/bin/python -m tools.evidence_training.train \
  --data dataset.json --output runs/cv-001 --device cuda --mode cv \
  --epochs 3 --batch-size 4 --accumulation 4 --max-seconds 2400
```

Pass `--model runs/pretrain-mix-001/model` to start the folds from a pretrained
checkpoint, and `--external-data external.json` to add SIMORD's medical-dialogue
questions to each training fold. Held-out conversations are never changed by
either option, and no other external source enters fold training.

Use `--fold 0` for one preliminary fold. Use all three folds for the comparison.
The default learning rate is 1e-5. The model sees overlapping 512-token windows
with stride 192, learns valid word endpoints, and uses CLS for negative/empty
windows. Across windows, candidate scores are normalized against the local CLS
score. No word cap is tuned on outer-fold performance. The default cap of 128
words exceeds the observed longest gold span of 33 words.

The best checkpoint is selected only by inner-development temporal IoU, including
the unmodified QA checkpoint as epoch zero. Outer-fold predictions are computed
after that selection. The combined report preserves all scored questions, counts
false-negative answers as zero evidence, and includes per-conversation changes,
selected epochs and a paired conversation-bootstrap confidence interval.

Useful outputs:

- `manifest.json`: exact arguments, source hashes, resolved model revision, library
  versions, device, precision, scope and completion status.
- `events.jsonl`: progress and losses, usable without repeatedly asking a coding agent.
- `fold_N/split.json`, `history.json`, `best/`, `report.json`: each independent fold.
- `report.json`: combined held-out predictions and comparison against the frozen control.

The training loop checks a wall-time budget before every batch and evaluation
batch. The outer `timeout` also bounds startup/download stalls. Budget exhaustion
is reported as incomplete (exit 124), not success. Completed-fold artifacts remain.
The runner does not stop the billed Vast instance: **stop it explicitly when the
command finishes**, including on errors. Model downloads and startup also count
towards billed instance time.

At $1.50/hour, a 45-minute compute cap is approximately $1.125, plus startup/setup
time and any provider storage fees. Avoid starting the instance during local data
preparation. Do not run platform validation alongside training.

For the existing Vast instance, use the already authenticated local CLI; never put
an API key in this repository, a command argument, or a run manifest:

```bash
.venv311/bin/vastai start instance 51489967
.venv311/bin/vastai ssh-url 51489967
# Run and retrieve the bounded job using the current address returned above.
.venv311/bin/vastai stop instance 51489967
```

Check the final state after stopping. A failed or queued start should also be
followed by `stop instance` to cancel an unintended later start. Stopping preserves
the instance disk and checkpoints; storage fees continue. Do not destroy the
instance. The supplied relay address was not usable during this session; the
provider's direct SSH address worked with the existing `~/.ssh/id_vastai` key.

## All-data fit after a useful pilot

```bash
.venv/bin/python -m tools.evidence_training.train \
  --data dataset.json --output runs/final-fit-001 --device cuda --mode fit --epochs 3
```

Choose fixed training settings using the earlier inner-development results. This
exports `model/` but does not produce a held-out score or integrate into serving.
Additional external-data ingestion and annotation are intentionally deferred; this
first run tests the existing samples without API generation costs.

## Checks

```bash
.venv311/bin/python -m unittest tests.test_evidence_training tests.test_evidence_external -v
```

Tests cover the actual 390-row control, gold-to-word mapping against a brute-force
oracle, fold leakage, duplicate transcripts, repeated phrase occurrence, overflow
windows, impossible spans, missing predictions, and a real tiny CPU training /
checkpoint reload / three-fold scoring run. The tiny model tests mechanics only.

The external tests cover CoQA history isolation and the no/unknown distinction,
character rationales expanded to whole words, MASH-QA offsets against the
published archive, SIMORD noncontiguous provenance, the lexical quarantine index,
the dataset-hash guard, train/dev leakage, and real CPU pretraining over a mixed
source bundle followed by fold training from that checkpoint. Tests needing the
downloaded archives or `TextGrid` skip when those are absent.

Optional Mac settings: `--device mps --precision fp32 --batch-size 1 --accumulation 8`.
No quantized LLM or MLX dependency is required by this package.
