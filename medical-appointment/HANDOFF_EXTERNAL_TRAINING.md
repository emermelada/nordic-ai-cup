# Handoff: finish preparing external evidence training data

> **Superseded 2026-09-20.** The work below was completed: the expanded bundle is
> `runs/evidence-external-20260919/prepared-v3/`, and the mixed-source GPU run was
> executed. Read [tools/evidence_training/EXTERNAL_DATA.md](tools/evidence_training/EXTERNAL_DATA.md)
> and [tools/evidence_training/RESULTS.md](tools/evidence_training/RESULTS.md) for the
> current state; this file is kept as the record of the starting point.

Saved 2026-09-19, 21:49 UTC / 23:49 Copenhagen. Intended for Claude Code or another coding agent with local filesystem access.

## User request and stopping point

The user wants a substantial improvement in the medical-appointment competition score, ideally toward 0.9. They authorized a new branch and an external-GPU training implementation, then asked:

> Prepare more data. How much data for approximately 1.5–2 hours of quality training?

They also asked whether one hour versus two hours would improve quality. Answer given: the gain is unknown; use validation and a two-hour ceiling rather than forcing the model to train for the full duration. The first external dataset has been prepared, and an additional medical corpus was just downloaded. **The expanded data pipeline is unfinished and has not been fully tested. No expanded-data GPU training has started.**

The user's latest request was to stop and create this handoff because their Codex credits are low. Resume only when the user supplies this handoff to the next agent.

## Workspace and version control

- Working directory: `/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment`
- Git root: `/Users/chinese/AICUP/Nordic-AI-Cup-2026`
- Branch: `codex/evidence-extractor-training`
- Latest commit: `9212693` — `Add bounded GPU evidence extractor training and evaluated pilot`
- Competition deadline, according to the local root README: September 20, 2026, 16:00 CEST.

The initial trainer is committed. The external-data work below is **uncommitted**:

```
modified: tools/evidence_training/train.py
new: tools/evidence_training/external.py
new: tools/evidence_training/fetch_external.py
new: tools/evidence_training/prepare_external.py
new: tools/evidence_training/pretrain.py
new: tools/evidence_training/requirements-data.txt
new: HANDOFF_EXTERNAL_TRAINING.md
```

Unrelated existing untracked items include the root `.gitignore`, root `README.md`, `drone-flyby/`, `images/`, `survival-simulator/`, and `medical-appointment/research/`. **Do not add, delete, or overwrite those.** Run artifacts under `runs/` are ignored. No production serving code was changed by this task.

## What already worked

The committed standalone trainer uses `deepset/deberta-v3-large-squad2` (about 435M parameters), BF16 CUDA autocast with FP32 weights, batch four / gradient accumulation four, 512-token windows, stride 192, learning rate 1e-5. It supports benchmark, conversation-disjoint CV, and all-data fit. It has checkpoint selection, wall-time limits, manifests, and strict span/coverage checks.

Original data: **39 conversations, 390 questions, 195 positive evidence spans**. Three outer folds: 24 training conversations, two inner-development conversations, 13 held-out test conversations per fold. Boolean answers and ASR timestamps are frozen to the saved control. The 1.0 answer accuracy belongs to the control, not to a newly trained classifier.

Completed original-only GPU experiment:

| Metric | Saved control | New extractor |
| --- | ---: | ---: |
| Mean temporal IoU | 0.7161810760 | 0.6662128340 |
| Raw score | 0.8297086456 | 0.7997277004 |

Raw change −0.0299809452; paired conversation-bootstrap 95% interval [−0.0537246546, −0.0065001795]. All three folds completed in about 97 seconds. **This was a regression, not an accuracy success.** Do not deploy it or promise 0.9.

Five-update GPU benchmark: approximately **0.465 seconds per optimizer update after warmup**, effective batch 16, peak allocated GPU memory 13.92 GiB. This is a short benchmark, not a precise estimate for larger text distributions.

Eleven tests passed for the original trainer, including a real tiny-model CPU training/checkpoint/CV test. These tests have **not been rerun after the external-data modifications**, and new external-data tests still need writing.

Important tokenizer fix already in the committed code: some Transformers/DeBERTa pair-overflow behavior silently truncated a long context before creating windows. `features.py` now encodes the complete context with the tokenizer backend before truncating into windows, and checks every word is reachable. Do not revert this.

Read `tools/evidence_training/README.md` and `RESULTS.md` for the initial experiment. Those documents do not yet describe the new external-data work.

## Current prepared data and downloads

All paths below are relative to the working directory.

### Original competition bundle

`runs/evidence-training-20260919/dataset.json` (21 MB)

Also present: `local-tests.log`, `source.tar.gz`, `source-final.tar.gz`. These archives are for the earlier committed trainer and **do not contain the new external-data work**.

### First external preparation — complete artifact, not yet training-tested

`runs/evidence-external-20260919/prepared/external.json` (76 MB)

Supporting files: `summary.json`, `quarantine.json`, `tokenizer/`, `nltk_data/`. Preparation log: `runs/evidence-external-20260919/prepare.log`.

Actual counts:

| Split/source | Questions | Contexts | Token windows |
| --- | ---: | ---: | ---: |
| CoQA training | 35,947 | 3,041 | 36,375 |
| SIMORD training | 29 | 23 | 138 |
| CoQA development | 2,614 | 200 | 2,659 |
| SIMORD diagnostic | 75 | 49 | 373 |

Thus **35,976 training pairs**, and 38,665 total examples across all splits. CoQA training includes 4,683 yes answers, 3,529 no answers, 589 unknown answers, and 27,146 other span answers. A CoQA **no** answer still has a positive evidence rationale; only **unknown** is unanswerable. Preserve that distinction.

This initial artifact predates the latest changes to `prepare_external.py`: it does **not** include MASH-QA, and its SIMORD filter allowed spans up to 128 words. The current code tightens SIMORD to 64 words and changes the question template, so create a **new** preparation directory; do not silently overwrite this artifact.

There were no competition-overlap rejections in the first preparation. Other rejected items included short/underspecified questions, spans shorter than three or longer than 128 words, noncontiguous medical evidence, duplicate/source-overlapping contexts, two overly long histories, and one unreachable span. About half the CoQA character annotations needed expansion to complete word boundaries; that transformation is recorded explicitly.

### Raw sources — downloads finished

`runs/evidence-external-20260919/raw/`

`download-manifest.json` records **95 files**, URLs, sizes and SHA-256 checksums. `download-v2.log` confirms completion. Files include:

- CoQA original training and development JSON, downloaded from Stanford.
- SIMORD training and original development annotations, official Microsoft release.
- ACI-Bench text JSON at commit `b909b2bb9cf1d19de08df15cddde7bd0179665e4`.
- Required PriMock57 doctor/patient TextGrid transcripts at commit `cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5`. **No audio was downloaded.**
- NLTK English sentence segmentation data and attribution/license files.
- **MASH-QA `mashqa_data.zip` (27 MB)**, downloaded from the dataset author's public Google Drive link. It contains original `train`, `val`, and `test` JSON in full and consecutive-span variants. The archive has not yet been converted by the full preparation pipeline.

The MASH-QA archive was inspected. Its consecutive-span JSON has SQuAD-style contexts/questions with `answer_start`, `text`, `answer_span` sentence IDs, and `answer_starts`. Use only genuinely consecutive evidence and verify the exact character slice. No synthetic labels are required.

## Source relevance and terms

- CoQA: https://stanfordnlp.github.io/coqa/ — human conversational questions with highlighted evidence. We use **Wikipedia and Gutenberg only**, CC BY-SA 4.0, excluding RACE/MCTest/CNN. Retain per-document attribution. Two previous QA turns are added to the question to resolve follow-ups; the current/future answer is never included.
- SIMORD: https://huggingface.co/datasets/microsoft/SIMORD and https://github.com/jpcorb20/mediqa-oe — clinician-annotated medical orders with sentence provenance, over ACI-Bench and PriMock57 transcripts. Annotations CDLA-Permissive-2.0, source transcripts CC BY 4.0. Official `dev` is now published `test1`; keep it **diagnostic only**, never use it to train or select checkpoints.
- MASH-QA: https://github.com/mingzhu0527/MASHQA — medical web reading comprehension, **not doctor–patient conversations**. Public archive link: https://drive.google.com/file/d/1RY_gWB4gaUPkW3w9WhIZAwxg5dzNFliK/view . The repository code has Apache-2.0; separate data terms are not stated in the README. Do not falsely claim the underlying WebMD content is Apache-licensed. Retain original WebMD URLs and source documentation.

SIMORD is small and often has overly broad or noncontiguous provenance. Do not inflate it by turning separated sentences into one giant gold span. A manual spot check found reasonable examples but also long turns discussing several orders; the new 64-word cap is intended to reduce this mismatch.

emrQA and RadQA require access agreements; they were not downloaded. PriMock/ACI do not have thousands of ready-made QA spans and overlap SIMORD's source material; do not double-count them as independent medical examples.

## New code: design and incomplete parts

`fetch_external.py` downloads data/text only, verifies or records hashes, supports `--lock`, reconstructs the list of required PriMock files, and handles the public Google Drive size-warning form for MASH-QA. It never uses paid APIs or downloads model weights/audio.

`external.py` provides:

- Text-only contexts with exact word character ranges, no fabricated audio timestamps.
- CoQA rationale conversion with previous-two-turn history.
- SIMORD transcript reconstruction matching the organizer's NLTK sentence numbering and TextGrid conversion, without executing downloaded code.
- Newly added MASH-QA conversion; **not yet tested against the complete archive**.
- Grouping, validation, and lexical overlap detection using 5-/8-word shingles.

`prepare_external.py`:

- Verifies raw hashes and excludes overlaps against **all 39 original conversations**, external development/diagnostic data, and source identities.
- Uses every example's actual tokenizer windows to check gold reachability before acceptance.
- Deduplicates contexts/questions, rejects ambiguous/oversized cases, records quarantine reasons.
- Preserves CoQA dev and MASH-QA val for external checkpoint selection; SIMORD test1 stays diagnostic.
- Newly added MASH-QA test handling uses source identities/contexts for exclusion only; its question labels are not imported for training or selection.
- Newly added `--medical-window-cap 45000` bounds supplemental MASH-QA training windows, deterministically ordered by context hash. This cap has **not yet been exercised**.
- Produces `external.json`, `summary.json`, `quarantine.json`, tokenizer and NLTK resources in a new output directory.

`train.py` modifications:

- New `--external-data` and `--mode pretrain`.
- Hash guard ties external decontamination to the exact original dataset file.
- CV/all-data fit append **SIMORD training only** to original training examples; held-out competition conversations remain unchanged.
- Records code/data hashes in manifests.
- **Incomplete:** external benchmark currently filters only `source == 'coqa'`; update it to reflect the new CoQA + MASH-QA mixture.

`pretrain.py`:

- Implements intermediate text-rationale training, 6% LR warmup/linear decay, per-epoch evaluation, early stopping, best checkpoint (including epoch zero), budgets/logging.
- External metric is word-span IoU, explicitly **not temporal IoU or competition score**.
- **Incomplete:** currently trains/selects on **CoQA only**, hard-coded in several places. Update it to include MASH-QA training/validation before calling the expanded dataset ready. Consider reporting and selecting a macro-average of CoQA/MASH-QA source metrics so the general corpus does not drown out the medical one. Clearly label this selection metric.
- It needs meaningful tiny-model tests before a paid run. There is no optimizer-state resume implementation.

The last compile check passed **before** the latest MASH-QA additions. No external training run, GPU or CPU, has yet validated these new paths.

## Suggested next actions

1. Read the uncommitted changes; finish the CoQA + MASH-QA benchmark/pretraining integration. Keep medical-dialogue adaptation confined to SIMORD's training split.
2. Run preparation into `prepared-v2` using the command below. Inspect exact counts, dropped-label reasons, medical span samples, and true window-based runtime estimates. Do not present the first artifact's counts as the expanded set's counts.
3. Add tests for: exact source offsets/word expansion; CoQA no versus unknown; previous-history isolation; medical noncontiguous-span rejection; MASH-QA offsets; no held-out/source leakage; dataset hash guard; mixed-source pretraining/checkpoint reload; medical-only augmentation of original training folds.
4. Run existing and new tests, including real tiny CPU training. Audit at least a small deterministic sample of each source.
5. Update README and a concise data report with final counts, limitations, source attribution, measured window counts, recommended epochs, and run commands. Preserve source locks and checksums. Make a new transfer archive containing the updated code, original bundle, external bundle, and attribution; no secrets.
6. Commit only this task's files on the existing branch. Leave production serving and unrelated files alone.
7. Report data ready and the realistic runtime estimate to the user. **The current request is data preparation; do not start a paid extended training run merely to fill the requested time.**

Preparation dependencies were installed into the experiment directory, leaving the application environment unchanged:

```bash
cd /Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment
PYTHONPATH=runs/evidence-external-20260919/prep-deps \
HF_HOME=runs/evidence-external-20260919/hf-cache \
TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 \
.venv311/bin/python -m tools.evidence_training.prepare_external \
  --raw runs/evidence-external-20260919/raw \
  --target runs/evidence-training-20260919/dataset.json \
  --output runs/evidence-external-20260919/prepared-v2 \
  --medical-window-cap 45000 \
  > runs/evidence-external-20260919/prepare-v2.log 2>&1
```

`prep-deps` contains nltk 3.9.2 and TextGrid 1.5. The tokenizer is cached under the experiment's `hf-cache`. Raw downloads are already complete; normally no network is needed for the next preparation. Always use a fresh output directory if a prior attempt exists.

## Runtime/cost reasoning

Initial planning target communicated: roughly **40,000–60,000 checked question/evidence pairs** for a 1.5–2 hour session, subject to actual window lengths. Do not pad data or repeat epochs just to consume time.

At the pilot's measured speed, estimate:

```
training_seconds ≈ windows × epochs ÷ 16 × 0.465
```

First CoQA set: 36,375 windows, about 17.6 min/epoch, 52.9 min for three epochs **excluding setup, evaluation and checkpoint I/O**. The planned extra medical budget permits up to 45,000 additional windows: about 81,375 total, two epochs about 79 minutes of updates, leaving room for evaluation/setup inside 1.5–2 hours. Actual counts/speed may differ; benchmark the new mixture before choosing the final cap.

At the user's $1.50/hour, 1.5–2 hours costs approximately $2.25–$3.00 in running GPU time, plus setup/idle time and retained storage. One hour versus two does **not** imply a predictable accuracy gain. Evaluate after each epoch and preserve the best checkpoint; overfitting is possible.

## GPU access and strict cost/deletion boundaries

- Vast instance: **51489967**.
- GPU: RTX PRO 6000 Blackwell Max-Q, 96 GB.
- Last checked: stopped (`actual_status=exited`, `cur_state=stopped`, `intended_status=stopped`). No GPU was started during external-data preparation.
- Existing authenticated CLI: `.venv311/bin/vastai`. The user supplied an API key in the conversation; **do not copy it into source, logs, manifests or this handoff**. The CLI already authenticates.
- Key path: `/Users/chinese/.ssh/id_vastai` (reference the path; never print its contents).
- Previously working direct address: `root@87.236.196.76`, port `40852`, with `IdentitiesOnly=yes` and `BatchMode=yes`.
- The supplied relay `ssh4.vast.ai:19967` did not work; relay port 19966 also denied keys. Query `.venv311/bin/vastai ssh-url 51489967` for the current direct address.
- A previous restart reported resources unavailable; status alone does not guarantee capacity. Cancel queued starts with an explicit stop if abandoning the attempt.

Controls, when an actual run is authorized:

```bash
.venv311/bin/vastai start instance 51489967
.venv311/bin/vastai ssh-url 51489967
.venv311/bin/vastai stop instance 51489967
```

When inspecting `show instance --raw`, filter to necessary status fields; its full JSON contains credentials. **Stop the billed instance on completion/error, not just the training process.** The trainer's time limit does not stop Vast billing. Use SSH connection/keepalive limits and bounded jobs. Do not leave a queued start or idle GPU running. Stopping retains disk, quoted around $0.0292/hour; destroying the instance would lose existing data and is not authorized.

Only deletion ever authorized: the **122B model**. Already deleted the exact directory:
`/workspace/models/models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4`.
The existing `/workspace/models/models--Qwen--Qwen3.8-27B` was preserved. **Delete nothing else.**

Remote isolated training workspace: `/workspace/medical-evidence-training-codex`, with `.venv`, `hf-cache`, original `dataset.json`, and initial trainer. Do not overwrite `/workspace/medical-appointment` or the production environment. Existing CUDA builds work; no driver/system upgrades needed.

Earlier GPU artifacts remain remote in `runs/cv-002/` (complete reports/checkpoints). Their numeric results were read, but detailed reports were not copied back before the instance stopped. Recover those when it next becomes available. An attempted `results.tar.gz` packaging job may or may not have finished; inspect before relying on it.

vLLM was paused with `supervisorctl stop vllm` for exclusive GPU use. A later restoration command was issued but completion was not confirmed before shutdown. Check serving state if serving is required; do not stop unrelated portal/tunnel services. The full remote environment guide is `/etc/vast-agents-guide.md` (also accessible from the remote workspace guidance).

## Completion standard

The immediate deliverable is a checked, reproducible external training dataset and tested ingestion/training path, with honest source counts and a budgeted next-run recipe. No claim of improved competition performance is justified until the augmented model beats the frozen control on the same conversation-disjoint comparison, followed by appropriate end-to-end validation. Do not deploy the regressed pilot or reuse held-out conversations for training.
