# Local MLX implementation

Use native Apple Silicon and Python 3.11. The original `.venv` uses Python 3.14;
keep it separate from this environment.

```bash
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements-mlx.txt
.venv311/bin/python -m unittest discover -s tests -v
.venv311/bin/python local_evaluator.py --oracle
```

These model snapshots must already be complete in the local Hugging Face cache:

- `mlx-community/whisper-large-v3-turbo` (fp16 serving ASR)
- `mlx-community/Qwen3.5-9B-4bit` (primary answers and evidence)
- `mlx-community/Qwen3-8B-4bit` (secondary answer veto and diagnostic evidence)

For the optional 8-bit ASR experiment, mlx-whisper loads `weights.safetensors`, but
that repo ships `model.safetensors`. Stage it into the gitignored `models/`
directory before selecting it; this is not required by the current serving build:

```bash
SNAP=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo-8bit/snapshots/*/)
mkdir -p models/whisper-large-v3-turbo-8bit
cp "$SNAP/config.json" models/whisper-large-v3-turbo-8bit/
cp "$SNAP/model.safetensors" models/whisper-large-v3-turbo-8bit/weights.safetensors
```

The 8-bit ASR scored 0.7558 against 0.7505 for fp16 on the training set at half the
memory. ASR weights are released after each transcription: reloading from the page
cache costs nothing measurable (3.0 s versus 3.1 s warm) and keeps the memory free for
the answering model. The 4-bit ASR was rejected at 0.7274.

Startup resolves cached snapshots with `local_files_only=True`; missing weights
fail startup rather than downloading them. No hosted APIs are used for inference.

## Current solution

Whisper-large-v3-turbo supplies word timestamps. Qwen3.5-9B answers all questions in one
deterministic call with thinking disabled, and Qwen3-8B answers them again as a second
opinion. An explicit secondary-model "no" rejects a primary-model "yes"; a primary
"no" is never promoted. Missing, invalid or duplicate secondary answers inherit the
primary decision rather than a retrieval veto. **Live build `543cf3e` keeps only
primary-model evidence** (`primary_evidence_response`), while preserving that answer
policy. Request capture is active. Platform validation scored **0.743079**, up from
**0.717908** with consensus, despite its lower local training score.

Current public prediction URL (replaced on 2026-09-18):
`https://clusters-jan-chorus-royal.trycloudflare.com/predict`.
The previous `computed-frequencies-lamp-carolina` hostname is no longer usable.

The preceding build `d0a5391` used retrieval to choose between the two models' spans,
with equal overlap scores preferring the earlier occurrence. Its platform score was
0.717908, below the best validated single-9B build. The live candidate isolates this
span selector without changing models, prompts or answer decisions.

Both answering models remain resident in 9.9 GB, with 14-20 s of generation in prior
runs. A reasoning channel, if emitted, is stripped before parsing.
`SECOND_LLM_MODEL = None` disables the second pass,
which is also skipped automatically when the first generation exceeds 25 s. The `compact` prompt
keeps the original answering rules but asks for one-line JSON with unit ids and a
quote per yes, about a third of the original output tokens. Decimal-aware alignment
maps the quote back to words without confusing `2.5` with `25`. Two span rules then
apply to every yes:

- a quote that ends in a question is extended to a reply of at most six words,
  but not another question or across a pause longer than two seconds;
- the start moves to the first 10 ms frame above -45 dBFS within the first selected
  word, because Whisper often starts a word inside the preceding pause while the
  annotations start at speech. A quiet first word is never trimmed away.

The model and prompt are `LLM_MODEL` and `DEFAULT_PROMPT` in
`pipeline/mlx_backend.py` (`legacy`, `focused` and `compact` are available).
Running processes keep their loaded code until restarted.

### Platform validation (19 conversations)

| Build | Training set (39) | Validation |
| --- | ---: | ---: |
| Qwen3-8B, legacy prompt | 0.738 | 0.732 |
| Qwen3-8B, focused prompt | 0.755 | 0.714 |
| Qwen3-8B, legacy + span rules | 0.753 | 0.732 |
| Qwen3-8B, compact + span rules | 0.770 | 0.699 |
| **Qwen3.5-9B, compact + span rules** | 0.751 | **0.743** |
| Qwen3.5-9B, minimal prompt | 0.751 | 0.731 |
| gpt-oss-20b, compact + span rules + 8-bit ASR | 0.760 | 0.727 |
| Qwen3.6-35B-A3B REAP-19B, compact + span rules | 0.756 | 0.732 |
| Qwen3.5-9B + Qwen3-8B consensus, compact | 0.770 | not yet validated |
| Same consensus, earlier-occurrence tie-break + grounding guards | 0.781 | 0.717908 |
| Same consensus + explicit secondary-no veto | 0.783777 | 0.717908 |
| **Primary-only evidence + same veto (live)** | 0.753538 | **0.743079** |

Validation attempt `c1dcda85c624436c85c667de8e68ffc7` completed on 2026-09-18
00:33 CEST against pipeline `aa50c41` (deployment `00ac739`): **0.7179075995**,
with no platform errors. All 19 requests completed without logged fallbacks;
mean/worst server time was **19.65/24.96 seconds**. The result is below the best
validated single-9B build, despite its higher local score. It does not isolate the
earlier-occurrence tie-break from the rest of the two-model build. Platform
accuracy/tIoU components were not supplied. Raw result and attribution:
`runs/platform-validation-c1dcda85/`.

Attempt `12faf62fab674ff4a4c5ba636810ce46` completed at 08:44 CEST on 2026-09-18
against answer-veto pipeline `d0a5391`: **0.7179075995361159**, exactly equal to the
preceding attempt, with no platform errors. The new API/worker processes served
all 19 conversations without logged fallbacks; mean/worst server time was
**18.68/23.85 seconds**. Every conversation's yes count matches the preceding run
(**91/190** overall). This is consistent with the veto having no effect on validation,
but full predictions and veto counters were not logged, so identical outputs are
not established. The three corrected training answers produced no validation-score
gain. Raw result and per-request attribution: `runs/platform-validation-12faf62f/`.

Attempt `17987a1a854248c9a8aae6bf49b8255f` completed at 14:22 CEST on 2026-09-18
against primary-evidence pipeline `543cf3e` (deployment record `a64446b`):
**0.7430787042906362**, a **+0.025171104754520224** gain over the preceding attempt.
This restores the historical single-9B best level; that older score is documented
only as approximately 0.743, so a new all-time record is not established.

All **19 conversations / 190 questions** completed with no platform errors, captured
fallbacks or worker errors. Mean/worst server time was **20.72/26.75 seconds**.
All 19 full request/trace/response captures were verified against the deployed source
hashes and successful platform POSTs; deployment smoke tests and the pre-attempt
request were excluded. Captures remain private and local under `runs/request-captures/`.

The secondary supplied **190 valid answers but triggered zero vetoes**. Both parsed
answer disagreements were primary-no/secondary-yes, which the policy does not promote.
Every returned yes had evidence. Per-conversation yes counts still match the preceding
attempt (**91/190** overall); its full predictions were not saved, so this does not
prove historical per-question equality.

Applying the old consensus selector to **this attempt's captured generations** would
replace six spans, four with disjoint intervals, without changing answers. Together
with the platform gain, this supports retaining primary evidence over this particular
selector. It does not identify which individual spans scored better: no gold labels
or accuracy/tIoU components were supplied, and a counterfactual score cannot be computed.
Result, capture attribution and label-free comparison: `runs/platform-validation-17987a1a/`.

Span consensus across models is the one lever that moved the training score materially:
three-model consensus reached 0.774, two models plus the retrieval referee 0.770-0.775,
while the same model under three different prompts gained nothing (0.750), and free
candidates from one model (its cited units, sentence expansion) were flat or worse.

Qwen3.8-27B (16 GB) scored 0.755 offline but cannot serve here: with Whisper loaded it
evicts its own weights, ASR slows from 9 s to 16 s, and requests miss the deadline. The
resulting swapping also killed the cloudflared tunnel once. Keep the answering model
near 12 GB on this machine.

### What the score is not limited by (measured 2026-09-17)

The best contiguous word span each transcript could give, scored against the gold spans
with onset refinement, is the localisation ceiling:

| Transcripts | Ceiling | Score with the 9B |
| --- | ---: | ---: |
| whisper-large-v3-turbo fp16 (serving) | 0.945 | 0.7505 |
| whisper-large-v3-turbo 8-bit | 0.945 | 0.7558 |
| whisper-large-v3-turbo 4-bit | 0.942 | 0.7274 |
| whisper-large-v3 (full) 8-bit | 0.892 | 0.7196 |

Actual tIoU is ~0.59 against a 0.945 ceiling, so the loss is passage selection, not ASR
timing. The full large-v3 is worse on both counts: coarser word timestamps (3.6 s mean
segments against turbo's 2.3 s) and twice the latency. Prompt wording moved nothing on
the 9B either: quote-the-whole-sentence 0.7355, prefer-the-explicit-statement 0.7467,
prefer-the-first-mention 0.7535, against 0.7505 for the serving prompt. Snapping span
boundaries to sentence ends was flat or worse at every setting tried.

Training-set scores did not predict validation: the prompt rules were written from
Qwen3-8B errors on those conversations. Choose builds by platform validation.
Offline, a LoRA on the training spans regressed on held-out folds and a worked
example in the prompt traded accuracy for spans; neither is in the serving path.

### Grounding fixes (2026-09-17)

A CPU replay of the exact serving consensus path on all 39 cached conversations
scores **0.769543 raw**, **0.626161 mean tIoU**, **384/390 correct**. Inputs, source
hashes, code snapshots and per-question outputs are archived in
`runs/grounding-fixes-20260917/`. Reply guards reject a following question or a
pause longer than two seconds; regression tests pass, and all 390 final outputs
remain identical to the control. Limiting onset refinement to the first selected
word also preserves all 390 outputs while preventing loss of a quiet first word.
These are correctness fixes, not measured score gains. Exact-repeat citation
tie-breaking scored **0.764762** and was removed from serving code; the candidate
and its tests remain archived under `exact-repeat-citations/`. This result does not
exclude using occurrence-aware candidates with a better selector. Blanket shared
span/intersection (**0.765982**) and combined span/union (**0.761582**) rules also
regressed and were removed; both experiments are archived.

**Retained gain:** prefer the earlier model span only when retrieval-overlap scores
are exactly tied. Missing retrieval evidence scores both spans at zero. Non-tied
choices and all primary-model answers stay unchanged. No additional inference is
needed. Full replay results:

| Public-training subset | Original consensus | Earlier-occurrence tie-break |
| --- | ---: | ---: |
| All 39 conversations | 0.769543 | **0.780700** |
| Development, 30 conversations | 0.770360 | **0.781627** |
| Previously inspected holdout, 9 conversations | 0.766820 | **0.777600** |

Accuracy stays **384/390**; mean tIoU rises **0.626161 → 0.644756**; zero-overlap
positives fall **30 → 26**. Seven spans change: five improve, one regresses, one
remains disjoint. A paired conversation bootstrap gives a descriptive 95% raw-gain
interval of **[0.00249, 0.02200]**; it does not correct for selecting candidates on
reused training data. Subsequent platform validation scored 0.717908, below the
best validated 0.743 single-model build; the local gain did not establish a
hidden-validation gain.

All **83 CPU tests** pass; the unchanged scoring oracle returns **1.000**. Replaying
cached worker frames through `Pipeline.predict` matches all **39/390** archived
conversation/question outputs (IPC mocked). No fresh ASR/LLM inference, HTTP latency
benchmark, serving restart or platform submission was performed during that
implementation pass. The API was subsequently restarted under a supervisor and
the user submitted the validation result recorded above.

Artifacts: `runs/grounding-fixes-20260917/earlier-tied-span/` contains per-question
scores, all changed spans and the runtime-parity result. `replay.py` beside it
replays both cached models; ordinary `tools.eval_offline` currently scores only the
primary model. To reproduce the consensus score, choose an unused output name:

```bash
.venv311/bin/python runs/grounding-fixes-20260917/replay.py verification
```

Fallback commits: `1ae5d06` is the original consensus; `eab9557` adds reply guards;
`edcd070` also protects the first word. All three score 0.769543 on this replay.
Optimization stopped after the first measured gain for user-run platform validation.

### Secondary-answer veto and ranker experiments (2026-09-18)

Two CPU-only ridge rankers used five conversation-disjoint folds, with scaling and
weights fitted only on training folds. The broad question-aware candidate ranker
scored **0.762816**, below the **0.780700** control. Restricting candidates to nearby
boundaries of the serving span scored **0.778313**. Neither is deployed. The broad
pool's **0.914520 candidate oracle** measures coverage with gold-based selection,
not achievable ranking performance; selection remains unsolved by these prototypes.
Scripts, fold assignments, weights and predictions: `runs/span-ranker-20260918/`.
These results do not rule out stronger question-conditioned selectors.

Inspecting the six wrong primary answers led to a smaller candidate: veto a primary
"yes" only when the secondary provides a valid "no" for that question. An omitted,
invalid or duplicate entry keeps the primary answer; a skipped second pass also
keeps the primary. No model, prompt, generation budget or additional pass changes.

| Public-training subset | Before veto | With veto |
| --- | ---: | ---: |
| All 39 conversations | 0.780700 | **0.783777** |
| Development, 30 conversations | 0.781627 | **0.784294** |
| Previously inspected holdout, 9 conversations | 0.777600 | **0.782044** |

Accuracy rises **384/390 → 387/390**. All **195 gold-positive outputs are identical**;
mean tIoU stays **0.644756**, with **26** zero-overlap positives. Only three false
positives change: abnormal examination, fever present, and the officially negative
but semantically ambiguous hobby question. Remaining errors are the implied
stethoscope examination, "molluscs" versus "molluscum," and diabetes complications
that both models incorrectly affirm despite "no complications."

Using the secondary's answers wholesale also gets 387 correct and scores 0.784204,
but introduces a new false positive about blood-test results while recovering the
stethoscope answer. The narrower veto avoids that new error on these data. This is
a three-question gain on reused public data, not evidence of hidden-set superiority;
a secondary false negative could hurt both accuracy and evidence credit there.

All **86 CPU tests** pass. The official scoring oracle remains **1.000**. Cached
frames through the actual `Pipeline.predict` match all **39/390** replay outputs
(IPC mocked; no fresh ASR/LLM inference or HTTP benchmark). Replay and code snapshots:
`runs/grounding-fixes-20260917/secondary-veto/`; answer audit and runtime-parity check:
`runs/answer-veto-20260918/`. The API and tunnel were not restarted during that
implementation pass; `9222c1c` is the pre-veto fallback commit.

### Primary-evidence ablation and request capture (2026-09-18)

The candidate preserves both model calls and the valid-secondary-no veto, but retained
yes answers keep the primary's evidence. It changes **26 spans and no answers** across
all 390 cached questions. The extracted consensus control exactly reproduces the
previous committed responses.

| Cached public-training replay | Consensus + veto | Primary evidence + veto |
| --- | ---: | ---: |
| Correct answers | 387/390 | 387/390 |
| Mean tIoU | 0.644756 | 0.594358 |
| Raw score | 0.783777 | 0.753538 |
| Zero-overlap positives | 26 | 31 |

This was tested as a controlled ablation despite its local regression, motivated by
the historical single-9B validation advantage. Subsequent platform validation improved
from **0.717908 to 0.743079**; the local score remains lower. Both model calls and the
answer-veto policy were preserved to isolate evidence selection. Disabling the secondary
model would also remove the veto and is a different experiment.

The deployed API captures requests that reached the worker under gitignored
`runs/request-captures/`. Each private JSON file contains the full base64 audio request,
ordered questions, transcript/word timings, both raw generations when available,
parsed primary/secondary/referee outputs, final response, outcome, process identity
and source hashes frozen at import. These are received inputs, **not hidden gold labels**;
failed inference can leave a partial trace. Requests rejected before dispatch are not saved.

Writes run after the HTTP response, publish complete files without overwriting existing
ones, and log storage failures without changing the response. New directories/files
use permissions 0700/0600. Capture stops at **512 MiB**; old files are not deleted.
Capture is enabled by default in this candidate; set `MEDICAL_CAPTURE_REQUESTS=0`
when launching the supervisor or server to disable it. Nothing is uploaded. The project
README's rules require local inference and do not prohibit local diagnostic logging.
Once validation inputs are inspected or used for tuning, treat them as development
information, not an untouched holdout.

**98 tests pass**, including capture failure, storage bounds, request isolation,
post-response ordering and worker recovery. The scoring oracle remains 1.000. All
39 real training request bodies also passed through FastAPI's ASGI path with cached
worker frames: exact replay predictions and 39 verified complete captures, totaling
106,008,309 bytes. Capture work after the response averaged 7.0 ms, worst 12.1 ms in
that run. This is not fresh inference or a network-latency benchmark.

Artifacts: `runs/grounding-fixes-20260917/primary-evidence/` and
`consensus-control-after-extraction/` in the same directory; API/capture verification:
`runs/primary-evidence-20260918/cached-api-and-capture/`. Reproduce with unused output names:

```bash
.venv311/bin/python runs/grounding-fixes-20260917/replay.py primary-rerun primary
.venv311/bin/python runs/primary-evidence-20260918/check.py api-rerun
```

### Rehearsal of the serving path (2026-09-18, consensus build)

All 39 training conversations through the running server, one request at a time, as the
evaluator calls it:

- Score **0.779**: accuracy 0.985 (384/390), mean tIoU **0.642**.
- **0 failed conversations, 0 timeouts, 0 fallback responses**; 2 yes answers without a span.
- Round trip 19.3 s mean, **33.9 s worst** against the 60 s budget, so 57% used at worst.

Fresh ASR scored above the cached-transcript replay (0.7696), so the replay estimate is
if anything conservative. Rerun with
`.venv311/bin/python local_evaluator.py --url http://127.0.0.1:9054/predict`.

### Deploy behind the tunnel

Build `d0a5391` was deployed on 2026-09-18 at 08:30 CEST by restarting only Uvicorn.
The existing supervisor started API PID 96371 and inference worker 96374. Public
health passed at the unchanged URL:
`https://computed-frequencies-lamp-carolina.trycloudflare.com/predict`.
A fresh local HTTP test of `sample_33` completed without fallback in **17.01 seconds**,
with **10/10 answers correct**, including the answer-veto regression case. This is
a smoke test, not platform validation. Record: `runs/deploy-d0a5391-20260918/smoke-test.json`;
server log: `runs/serve-9054/server-20260918-083032.log`.

Build **`543cf3e` was deployed at 14:02 CEST on 2026-09-18** by restarting Uvicorn
under the existing supervisor. API PID **98962**, inference worker **98964**;
server log: `runs/serve-9054/server-20260918-140226.log`. Fresh local `sample_33`
returned HTTP 200, **10/10 correct in 19.89 seconds**, without fallback. Its capture
confirms the source hashes, primary-only evidence and an active secondary-no veto.

The old tunnel had already failed before this restart: its log reported
`Unauthorized: Tunnel not found` from 12:37 CEST, readiness was HTTP 503 with zero
connections, and both local DNS and Cloudflare's 1.1.1.1 resolver returned NXDOMAIN.
The old process was stopped gracefully and replaced at 14:10 CEST. New tunnel PID
**99355**, log `runs/cloudflared-20260918-141056.log`, metrics still on 127.0.0.1:20241.
The unrelated tunnel to port 9053 was not touched.

**Use the new URL:** `https://clusters-jan-chorus-royal.trycloudflare.com/predict`.
Public health and a fresh HTTPS audio request both returned HTTP 200; the public
smoke test scored **10/10 in 26.91 seconds**, without fallback, and produced a
verified capture. Local and public smoke outputs were identical. These are smoke
tests, not platform validation. Records: `runs/deploy-543cf3e-20260918/`.

The cloudflared quick tunnel forwards to 127.0.0.1:9054; restarting a healthy tunnel
changes the public hostname. A running process and `/quicktunnel`'s reported hostname
do not prove it is usable: also check `/ready` and public DNS/health. Keep the Mac
awake and online during validation. With `tools/serve.sh` running, check the latest
`runs/serve-9054/` log and active connections for in-flight requests, then restart
only Uvicorn. The supervisor starts its replacement; do not launch a second server.

```bash
kill -INT $(lsof -tiTCP:9054 -sTCP:LISTEN)
curl -s http://127.0.0.1:20241/quicktunnel   # public hostname; submit https://<host>/predict
```

## Serve and score

```bash
.venv311/bin/python -m uvicorn api:app --host 127.0.0.1 --port 9054 --workers 1
```

Wait for `Application startup complete`, then in another terminal:

```bash
.venv311/bin/python local_evaluator.py --url http://127.0.0.1:9054/predict --verbose
```

Use one worker, without reload. Startup warms Whisper and Qwen in a persistent
spawned child. The parent serializes requests and owns a 52-second inference
budget, including lock waits and IPC, plus at most two seconds of process cleanup.
On timeout/crash it returns the completed ASR-based retrieval fallback, or all
`true` with null spans if ASR never completed, and restarts the child separately.
Requests during recovery receive the floor response rather than waiting for
model loading. Ctrl-C shuts down the server and inference child.

This command binds only to loopback. It does not expose the endpoint, validate on
the competition platform, or spend the single evaluation attempt.

## Iterate offline

Existing probe artifacts were preserved in gitignored `transcripts/probe-turbo/`
and `runs/reference/`. Replay uses no model inference:

```bash
.venv311/bin/python -m tools.eval_offline \
  --transcripts transcripts/probe-turbo \
  --replay-llm runs/reference/llm_dump_qwen8b.json
```

For a new ASR cache and fresh Qwen answers:

```bash
.venv311/bin/python -m tools.transcribe_all --output transcripts/whisper-turbo
.venv311/bin/python -m tools.eval_offline --transcripts transcripts/whisper-turbo
```

Both tools accept `--limit N`. Transcription resumes compatible complete entries;
unknown/legacy cache files are never overwritten. Evaluation saves `questions.csv`,
`raw_outputs.json`, and `summary.json` under a new `runs/offline/` directory.
Use `--output DIR` to choose an unused destination. Missing conversations or
mismatched replay questions fail explicitly instead of shrinking the denominator.

`--retrieval-only` evaluates the actual fallback. `--start-offset 0.1` compares
moving model-aligned/cited evidence starts **later** by 0.1 seconds; it does not
change serving defaults or retrieval spans. Baseline serving uses zero offset.

### Compare prompts and reproduce the split

```bash
# Old prompt with the old matcher, using fresh generation on cached ASR:
.venv311/bin/python -m tools.eval_offline --transcripts transcripts/probe-turbo \
  --prompt legacy --alignment legacy --output runs/new-legacy-comparison

# Current solution, with a conversation-level split:
.venv311/bin/python -m tools.eval_offline --transcripts transcripts/probe-turbo \
  --subset holdout --output runs/new-focused-holdout

# Replay the saved complete candidate without loading models:
.venv311/bin/python -m tools.eval_offline \
  --replay-llm runs/evidence-focused-all/raw_outputs.json

# Development ablations with per-conversation resumable generation:
.venv311/bin/python -m tools.compare_evidence --subset dev --prompt focused \
  --output runs/new-focused-experiment
```

`--subset dev` selects 30 conversations; `--subset holdout` selects the nine IDs
with the lowest SHA-256 hashes. Selection happens before `--limit`. Question labels,
spans and IDs are never supplied to the answering model. The holdout was excluded
from candidate selection, but is still previously inspected public training data,
not an unseen competition validation set.

Replay honors its saved `alignment` metadata; dumps without it retain the legacy
matcher. `--alignment numeric` explicitly tests the new matcher on an older dump.
Fresh runs default to the serving prompt and numeric matcher. Outputs are never
overwritten. Choose an unused directory for each scoring run.

## Earlier focused-prompt study (2026-09-17, Qwen3-8B)

All CPU tests pass (75 as of the compact build), including worker hangs/crashes, recovery, partial IPC,
concurrent requests, parser failures, decimal tokens, replay metadata and deadlines.
The unchanged official scoring oracle returns 1.000.

Cached-ASR comparison (score = 0.4 × accuracy + 0.6 × mean tIoU):

| Dataset | Baseline score | Focused score | Baseline correct | Focused correct |
| --- | ---: | ---: | ---: | ---: |
| Development, 30 conversations | 0.7258 | **0.7550** | 298/300 | 297/300 |
| Holdout, 9 conversations | **0.7818** | 0.7567 | 88/90 | 89/90 |
| All 39 conversations | 0.7384 | **0.7554** | 386/390 | 386/390 |

The full candidate's mean tIoU rises from **0.5709 to 0.5991**. This is a boundary
selection gain, not improved passage recall: zero-overlap positives rise from
30 to 31, including three missing spans instead of two. The candidate exceeds
0.734 on both splits, but **regresses against the baseline on the holdout**;
there is no evidence here that it will beat the baseline on unseen competition
conversations. Platform validation is still needed before the single final attempt.

Development ablations scored 0.7453 for a separate evidence pass and 0.7325 for
quote-before-answer generation. Neither was adopted. Citation-priority matching
also regressed; numeric-safe matching retained the winning development score.
No timing offset was fitted or applied.

Artifacts: `runs/evidence-baseline-{dev,holdout}/`,
`runs/evidence-focused-dev/scores/`, `runs/evidence-focused-holdout/`,
`runs/evidence-focused-all/`, and `runs/evidence-{refine,grounded}-dev/`.
Each scoring directory contains the raw output replay, per-question CSV and summary.

Fresh audio-to-response HTTP evaluation of the focused solution completed all
**39 conversations / 390 questions** on an isolated loopback server:

- Raw score: **0.756**, versus the previous baseline HTTP score of **0.736**.
- Accuracy: **386/390 (0.98974)**; mean tIoU: **0.600**.
- Failed requests / timeouts: **0 / 0**.
- Mean / worst round trip: **14.920 / 21.551 seconds** (60-second limit).
- Report: `runs/evidence-http/evaluation.log`; server timings: `server.log` in that
  directory. The temporary server on port 9055 was shut down cleanly afterward.

The one-off JSON export failed after scoring because `dataclasses.asdict` could
not serialize a `defaultdict`. `runs/evidence-http/summary.json` was recovered from
the intact official report and explicitly labels its rounded precision: score and
tIoU to three decimals, latency to milliseconds. No inference requests failed.

The previous baseline HTTP run had mean tIoU 0.567 and mean/worst latency
**16.495 / 27.224 seconds** (`runs/http-evaluation.log`). Cached replay is not fresh
ASR or HTTP latency. The existing server on 9054 and its tunnel were not restarted,
and no competition validation or evaluation attempt was submitted.
