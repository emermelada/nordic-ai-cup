# Medical appointment serving state

## FINAL BUILD (locked 2026-09-19) — read this first

Best validated **0.8167711209** (attempt `b524ca41`), from 0.7431 at the start of the day.
Five validations of this configuration returned 0.81658-0.81677, a spread of 0.0002, so it
is deterministic and the number is real. **No further experiments are planned**; distillation
was costed and declined (expected +0.005 with a 30% chance of a loss, see the closing
section). What remains is the single evaluation attempt, before **2026-09-20 16:00 CEST**.

**The configuration, exactly:**

| Piece | Value |
| --- | --- |
| Host | Vast instance **51489967**, RTX PRO 6000, stopped between sessions |
| Model | `Qwen/Qwen3.8-27B` bf16 via vLLM, supervisor-managed, auto-starts |
| ASR | `MEDICAL_ASR_MODE=base` — faster-whisper `base`/int8/CPU, the annotators' coordinates |
| Answers | `MEDICAL_ANSWER_PROMPT=named` (compact rules plus the phonetic-name rule) |
| Evidence | `MEDICAL_EVIDENCE_MODE=perq`, `MEDICAL_EVIDENCE_PROMPT=v3` |
| Rescue | `MEDICAL_RESCUE_PHONETIC=1`, `MEDICAL_RESCUE_READING=turbo`, `MEDICAL_RESCUE_THRESHOLD=0.24` |
| Exposure | Vast port 3000 -> **http://87.236.196.76:40724/predict** (no Cloudflare) |

**To run the final evaluation:**

```bash
.venv311/bin/vastai start instance 51489967          # ~2 min, then ~4 min for vLLM to load
ssh vast '/workspace/serve_direct.sh 0.24'           # binds 0.0.0.0:3000, prints the public URL
curl -s -m 20 http://87.236.196.76:40724/api         # expect a JSON uptime line
ssh vast 'cd /workspace/medical-appointment && python3 local_evaluator.py --url http://127.0.0.1:3000/predict'
.venv311/bin/vastai stop instance 51489967           # afterwards; $1.52/h running, $0.029/h stopped
```

Expect the verification to print **accuracy 1.000, tIoU 0.719-0.720, raw 0.831-0.832**, zero
fallbacks, zero null spans, worst round trip about 23 s against the 60 s budget. If it does
not, something drifted: do not submit. `tools/serving/serve_direct.sh` is the same script
kept in the repo in case the instance is rebuilt; `GPU_SETUP.md` rebuilds the box from
scratch in about ten minutes.

**Two rules that cost us real points today.** Keep the GPU exclusive during any scored
attempt: an offline experiment running alongside a validation cost 0.016 on an otherwise
identical build. And never let the endpoint be reachable while the evidence stage is being
restarted; verify `/api` before handing over a URL.


**Deployment paused on 2026-09-19:** the user confirmed the Vast credits ran out
and instructed **no further SSH connection attempts**. The SSH recovery watch was
stopped; do not resume connection checks or deployment until the user says to.
The last successful health check was about 02:08 UTC; at 02:19 UTC SSH refused
connections and the public URL returned Cloudflare 1033 (HTTP 530). No deployment
files or processes had been changed. The retrieved-example candidate is prepared
locally; it is **not deployed and not ready for validation**. Its integration passed
216 unit tests before the newer `locate` experiment was added. Test execution and
model/validation runs for that newer code are deferred at the user's request.
A local recovery archive of the baseline, with all 11 original pipeline-module
hashes verified, is at `runs/resume-20260919-0325/baseline-source.tar.gz`.

## The gap to 0.82 is missed positives, not span quality (2026-09-19)

Every route to better spans is now exhausted: prompt rewrites (`match`, `exchange`,
`annot`), no-draft and retrieved-example ablations, candidate verification, probability
re-ranking, medoid ensembling, reasoning, a larger model (122B), the turbo-text hybrid, and
post-hoc trimming rules all scored at or below the serving build. Meanwhile the answer pass
was treated as solved because it is 390/390 on the training conversations.

It is not solved on unseen ones. Both platform sets are exactly half yes, 95 of 190, and
the captured hidden-set attempts returned **91, 91 and 92** yes answers. At least three or
four positives were therefore being missed on conversations the prompt was not tuned on.

That single fact explains the whole local-to-platform gap. A missed positive loses its
accuracy mark *and* its tIoU share, since the denominator is every annotated yes:

| Missed positives | Accuracy | Mean tIoU | Predicted raw |
| ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.7200 | 0.8320 |
| 2 | 0.9895 | 0.7048 | 0.8187 |
| 3 | 0.9842 | 0.6973 | 0.8120 |
| 4 | 0.9789 | 0.6897 | 0.8054 |

Local measured 0.832; the platform returned 0.8094, which sits between three and four
misses. Recovering even one or two clears 0.82, and no span improvement has to work.

A miss costs 0.4/190 + 0.6x0.72/95 = **0.0067 raw**; a false positive costs the accuracy
mark alone, **0.0021**. The miss is **3.2x** worse, so the break-even confidence for
answering yes is about **0.24**, not 0.5. Nothing in the pipeline knew that.

### What was built for it

`pipeline/rescue.py` re-asks every question answered no, on its own, against the numbered
transcript, and reads P(yes) from the reply token's logprobs with the scorer already used
by `pipeline/verify.py`. A no flips only above `MEDICAL_RESCUE_THRESHOLD`; a yes is never
revisited, so the decisions that are already right cannot be disturbed. The worker runs it
before the evidence stage, so a rescued question is given a span like any other yes, and
the parent applies the identical flip. A rescued answer is exempt from the secondary veto,
since the asymmetry argument already prices in the uncertainty a second opinion expresses.

The default threshold is **0.5, which changes nothing**. It is opted into only after the
calibration below.

Separately, **every yes now carries a span**. A yes with a null span scores nothing on the
larger half while any span scores at least nothing, so a yes the evidence stage could not
place falls back to the sentence sharing the most content words with the question
(`stage_b.lexical_span`). Earlier local runs returned two such null spans per attempt.

### Calibrating it (the first thing to run when the GPU is back)

`tools/gpu_eval/rescue_calibration.py` sweeps the threshold offline once the probabilities
are cached. Point it at the **`compact`** answers, not the deployed `named` ones: the
serving build is perfect on training, so a rescue there can only add false positives and
the benefit would be invisible, whereas `compact` misses exactly two positives (Airomir
heard as "Aromere", molluscum as "molluscs") and both sides of the trade become measurable.

```bash
MEDICAL_BACKEND=vllm VLLM_URL=http://127.0.0.1:18000/v1 EVAL_INPUTS=inputs_base.json \
  python3 tools/gpu_eval/rescue_calibration.py --source tools/gpu_eval/results/answers-base27b.json
```

Serve the lowest threshold whose false-positive column stays small, via
`MEDICAL_RESCUE_THRESHOLD`. Cost is one short call per no answer, about five per
conversation, run in parallel; the request budget is 33% used at worst today.

### Calibration result (2026-09-19, 197 re-asked "no" answers)

A re-ask that reads the same garbled text **repeats the same mistake**: the two positives
the `compact` prompt misses scored P(yes) **0.031 and 0.138**, below any usable threshold,
while the 195 true negatives sat at a median of 0.000 with nothing in [0.2, 0.5). The
scorer is confident and confidently wrong on exactly the cases it should catch, so
threshold tuning alone could never have worked. Independence had to come from the prompt.

| Rescue variant, threshold 0.24 | Recovered of 2 | False positives of 195 |
| --- | ---: | ---: |
| same text, plain | 0 | 0 |
| **same text + phonetic-name rule (deployed)** | **1** | **0** |
| accurate turbo text, plain | 1 | 0 |
| accurate turbo text + phonetic rule | 1 | 0 |

The phonetic rule lifts one miss from 0.031 to 0.991. Reading the accurate turbo transcript
matches it but needs a second ASR pass per request, so the same-text variant is served:
`MEDICAL_RESCUE_PHONETIC=1 MEDICAL_RESCUE_THRESHOLD=0.24`. Its one residual risk is a single
true negative scoring in [0.2, 0.24); the turbo variant has an empty band and is the fallback
if false positives ever appear.

**Deployed 2026-09-19.** The 39-conversation verification is unchanged at **1.000 accuracy,
0.719 tIoU, 0.832 raw**, with **0 answers rescued** - the expected and required result, since
the `named` prompt is already perfect on training, so this measures only that the rescue adds
no false positives. 0 fallbacks, 0 null spans, 13.3 s mean and 21.7 s worst (36% of budget).
Its value is entirely on unseen conversations, where 3-4 positives were being missed.

## Selection is a closed question: seven mechanisms, all below generation (2026-09-19)

The candidate pool around our own pick has an oracle of 0.862 tIoU (raw 0.915) and the
deployed spans reach 0.719, so roughly 0.08 raw sits in choosing better among candidates we
already produce. Seven independent mechanisms have now been measured against it, and every
one lost to simply keeping what the model generated first:

| Mechanism | tIoU |
| --- | ---: |
| **generation, as deployed** | **0.718** |
| ensembling (medoid over 5 prompt variants) | 0.7185 (oracle over them 0.835) |
| reasoning before answering | 0.659 |
| candidate verification, shortest sufficient | 0.653 |
| question likelihood, length-corrected | 0.613 |
| question likelihood, raw | 0.548 |
| multiple choice over lettered candidates | 0.503 |
| probability re-ranking | 0.404 |
| locate: sentences, then explicit word IDs | 0.671 |
| rule-based post-processing | +/-0.00 |

The last of these was the most principled: the questions were written *from* the gold
passage, so P(question | passage) is literally the annotation's generative story, and a
likelihood has no position bias and no opinion to be talked out of. Scored with vLLM prompt
logprobs it still lost. Raw, it prefers passages a median of 12 words against a gold median
of 8, because a longer passage makes any question more predictable; the best length penalty
recovers it only to 0.613. `tools/gpu_eval/likelihood_eval.py` keeps the implementation.

`locate` was worth re-testing because it generates rather than selects, and its parser was
rejecting the bare `[31, 34]` pair the model actually returns instead of the named object the
prompt asks for; fixing that took valid outputs from 18/195 to 175/195 and 0.614 to 0.671,
still below 0.718. Mutual exclusivity across a conversation's ten questions is also a dead
end: gold spans overlap on 7.0% of question pairs and ours on 5.6%, so we are already less
repetitive than the annotator.

Treat "select among candidates" as closed. The only untried family with a real mechanism
left is supervised fine-tuning with **enough** data - the single LoRA attempt used 137
quotes at rank 8 and was negative out of fold, which is what that quantity buys, not a
refutation. Doing it properly means rejection-sampling the 27B's own best spans on the
training folds and needs one to two hours of exclusive GPU.

## Current primary endpoint (last verified 2026-09-19)

The primary deployment is **Qwen3.8-27B bf16 on the Vast RTX PRO 6000**, not the
historical Mac builds below. It uses `MEDICAL_BACKEND=vllm`,
`VLLM_URL=http://127.0.0.1:18000/v1`, `MEDICAL_ASR_MODE=base`,
`MEDICAL_ANSWER_PROMPT=named`, `MEDICAL_EVIDENCE_MODE=perq`,
`MEDICAL_EVIDENCE_PROMPT=v3`, and `ASR_CPU_THREADS=16`.

Public prediction URL (reissued when the tunnel restarts):
`https://flooring-vbulletin-capital-blink.trycloudflare.com/predict`.
Health/info is `/api`, not `/health`. API and tunnel are the remote `api` and `tunnel`
tmux sessions; vLLM is supervisor-managed. Preserve the tunnel and model process
when deploying API changes. See `GPU_SETUP.md` for the machine layout.

Best documented **platform validation remains 0.8093943511**, attempt
`c89957bd05be475dbb6d1eb5115ba9c7`. The 122B replacement scored **0.7918187304**
(attempt `a6f87ab589c84e0eb1aa7e34b5c7c8ea`) and was reverted. After restoration,
the 27B completed all 39 training conversations: **390/390 answers, 0.720 tIoU,
0.832 raw**, no failures/timeouts/fallbacks, **12.150 s mean / 19.122 s worst**.
The local training score does **not** establish the requested >0.82 platform score.

The turbo-text/base-timestamp hybrid also lost: named-prompt cached raw **0.8204**
versus the historical base-only cached **0.8310**; compact hybrid **0.8134**.
These old cached comparisons used the old evaluator and are not interchangeable
with runtime-faithful replay. Recovery logs/results are preserved under
`runs/resume-20260919-0325/`. Branch `feat/medical-resume-verification` preserves
the inherited uncommitted implementation.

Keep the GPU exclusive during every platform attempt: offline contention already
cost 0.016 on an otherwise identical build. Never launch the single final evaluation
automatically. The sections below preserve earlier experiments, not current deployment
instructions.

## Resume work (2026-09-19)

`tools/gpu_eval/eval.py` now reparses saved answer-pass raw output, passes its
original `draft_quotes` to evidence generation, and disables reply extension for
explicit base/exact coordinates. Supply `--asr-mode base` for the legacy base input
bundle; a missing energy envelope is not used to infer timing policy. Source/raw
answer disagreement fails rather than silently changing fixed decisions.

Generation caches use versioned prompt/input/model/decoding/timing fingerprints;
`--model-id` must name the physical model revision, not just `qwen`. `--replay`
constructs no backend and fails on missing/incompatible generations. Historical
unfingerprinted Stage-B caches cannot be reused as the corrected control. Existing
result files are preserved, whole parallel batches are timed, and missing/unmatched
evidence is reported explicitly. The local suite passes **204 tests**.

The evidence ablations are offline-only in `tools/gpu_eval/evidence_variants.py`:
`control` preserves the existing builder, `no-draft` removes both draft text and its
keep/fix instruction, and `retrieved` replaces the fixed examples with two
question-BM25 demonstrations from distinct donor conversations. Sorted conversation
IDs are assigned round-robin to three 13-conversation folds; the entire scored fold
is excluded from its donor bank. Greeting-only annotations remain in scoring but
are excluded from demonstrations. These previously explored training conversations
are not a pristine held-out validation set.

At 01:49 UTC, all **197 / 536,174,728 bytes** of remote request captures were moved,
without deletion, to `runs/request-captures-pre-resume-20260919/`. The API recreates
its normal capture directory on the next prediction. No API/model/tunnel restart
was needed. Nineteen subsequent incoming requests wrote private, complete worker
traces with no reported evidence-stage errors/skips. The local capture fingerprint
now also covers `vllm_backend.py` and `base_asr.py`; this fingerprint change is not
yet deployed.

### Bounded experiment results, 02:03–02:07 UTC

A traffic guard stopped the first launch before any inference when incoming
predictions started. The experiment ran only after that 19-request batch finished
and the API stayed quiet for 60 seconds. All three runs used the same 27B revision,
base transcript bundle and fixed named-prompt answers; no model/API/tunnel restart
or serving-code change occurred. The 585 evidence completions took **216.8 s** total.

| Evidence variant | Correct | tIoU | Raw | Δ raw vs control | Zero overlap | Batch seconds mean / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime-faithful control | 390/390 | 0.7162 | 0.8297 | — | 21 | 1.48 / 2.84 |
| No draft | 390/390 | 0.6892 | 0.8135 | -0.01619 | 24 | 1.52 / 2.70 |
| Retrieved examples | 390/390 | 0.7056 | 0.8233 | -0.00637 | 22 | 2.31 / 3.84 |

No-draft fold raw deltas were **-0.0117 / -0.0216 / -0.0149**; retrieved-example
deltas were **+0.0078 / -0.0252 / -0.0003**. Paired conversation-bootstrap 95%
intervals for raw gain were **[-0.03468, +0.00048]** and **[-0.02304, +0.00936]**,
respectively. No-draft improved/lost/tied 19/25/151 positive spans; retrieved
examples 13/19/163. Every expected evidence response was present, but 3/3/4 outputs
were unusable for alignment in control/no-draft/retrieved; runtime-equivalent
fallback preserved the primary spans. These were not missing batches or timeouts.

Neither candidate met the predeclared promotion gate. **The live build is unchanged;
no >0.82 platform improvement is established.** This is a local result, not proof
that the variants cannot improve platform validation. Retrieved examples remain the
more plausible candidate for a controlled platform comparison if that is pursued;
the recent incoming batch tested neither variant.

All three complete results were reproduced locally through strict CLI replay with
backend construction and network calls blocked. One final code-review pass found no
actionable issues. Full results, raw generation caches, fold/conversation comparisons,
bootstrap intervals and provenance are in `runs/resume-20260919-0325/`, especially
`paired-comparison.json`, `faithful-results/`, `execution.json`, and
`strict-replay.log`. Remote staging is `runs/evidence-resume-20260919/`; the model
queues were empty and the original public `/api` was healthy after completion.
No candidate was deployed, so no candidate-specific fresh-audio or platform attempt
was launched. The final evaluation was not used.

### Prepared locally; further testing deferred

The retrieved-example candidate is integrated behind
`MEDICAL_EVIDENCE_MODE=perq MEDICAL_EVIDENCE_PROMPT=retrieved`. Its frozen bank in
`pipeline/span_examples.json` contains 193 curated public-training demonstrations.
Matching all ten questions excludes a known training conversation's entire fold,
independent of fresh ASR; unseen question sets use the full public-training bank.
All 195 saved positive prompts and all 39 backend batches matched the pre-integration
offline implementation. This integration passed 216 tests and its final review
before the subsequent experiment below was added. Defaults remain unchanged.

The newer **`MEDICAL_EVIDENCE_MODE=locate`** experiment is implemented but unmeasured:
1. Locate a contiguous range of numbered sentences from the full transcript, without
   the answer-pass draft anchoring the choice of occurrence.
2. Show that passage plus one neighboring sentence on each side as context; ask for
   inclusive, absolute word IDs defining the final span. Context is not output padding.

Word IDs distinguish repeated identical quotations and avoid fuzzy quote alignment.
Out-of-range, reversed, duplicated or otherwise malformed selections preserve the
primary span; no boolean answer can be changed. Both model batches share the existing
request budget and the parent watchdog is unchanged. The existing timestamp policy
still applies: base coordinates do not receive reply extension or onset adjustment.

The evaluator's `stageb --evidence-mode locate` path exercises the same implementation,
using fixed saved answers supplied with `--source` and the usual model/input flags. It records both batches' actual prompts, outputs, timings,
selected windows and completeness. The complete chain is cached with prompt templates,
input/configuration and source fingerprints, so strict replay preserves budget-skipped
second stages rather than trying to generate them. Three-fold reporting remains
available. A 39-conversation run needs at most 390 evidence completions.

Regression cases for the new parser, repeated occurrences, window restrictions,
shared budget, answer preservation, complete-chain replay and label independence
have been written in `tests/test_locate.py` and `tests/test_gpu_eval.py` but **not run**.
No model comparison, fresh-audio test or platform validation of `locate` has been run;
no score improvement is claimed. The prior 216-test result does not certify this newer
code. Resume testing only when requested; do not reconnect to the GPU automatically.

## Historical local MLX setup

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

## Earlier local solutions (historical)

Whisper-large-v3-turbo supplies word timestamps. Qwen3.5-9B answers all questions in one
deterministic call with thinking disabled, and Qwen3-8B answers them again as a second
opinion. An explicit secondary-model "no" rejects a primary-model "yes"; a primary
"no" is never promoted. Missing, invalid or duplicate secondary answers inherit the
primary decision rather than a retrieval veto. Evidence keeps only the primary model's
passage (`primary_evidence_response`), while preserving that answer policy. Request
capture is active. Build `543cf3e`, which stops there, scored **0.743079** on platform
validation, up from **0.717908** with consensus, despite its lower local training score.

**The Mac build deployed on 2026-09-18 at 20:33 CEST was the REAP-19B single-model pipeline
with the sentence-table evidence stage** (uncommitted working tree on top of `c02a72a`):
`LLM_MODEL = 'mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit-REAP-19B'`, `SECOND_LLM_MODEL = None`,
`EVIDENCE_MODEL = LLM_MODEL`. It validated **0.779780** (attempt `d2804175`). The 9B plus
9B stage B validated 0.754572 before it, and the committed `543cf3e` build 0.743079. See
[sentence-table evidence stage](#sentence-table-evidence-stage-2026-09-18) and
[larger evidence models](#larger-evidence-models-2026-09-18-late-evening); deploy records
are under `runs/deploy-stageb-20260918/`. To serve an earlier build, set the three model
constants in `pipeline/mlx_backend.py` back (9B, 8B, `LLM_MODEL` for the 9B stage-B build;
plus `EVIDENCE_MODEL = None` for `543cf3e`) and restart only Uvicorn.

Local training score is not a reliable guide here. Every build that beat `543cf3e`
locally has matched or lost to it on the platform: consensus scored 0.783777 locally
and 0.717908 on validation. The boundary selector's +0.011 is out of fold but still on
conversations the ranker was fit on, so **it remains an untested upgrade path, not a
disproven one** — it needs a validation attempt, not a better local number.

Earlier Mac prediction URL (not the primary GPU endpoint):
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
| Same + learned boundary selector (deployed 16:46, rolled back 17:05) | 0.762963 out of fold | not yet validated |
| **Same + sentence-table evidence stage (live since 18:10)** | 0.7739 cached / 0.769 fresh | **0.754572** |
| **REAP-19B for answers and the evidence stage, no second model (live since 20:33)** | 0.7877 cached | **0.779780** |
| **Vast.ai RTX PRO 6000: Qwen3.8-27B bf16 via vLLM for both passes (see GPU_SETUP.md)** | 0.8008 cached / 0.804 fresh | **0.792957** |
| **Same, annotators' coordinates + per-question evidence (live since 02:05 CEST)** | 0.8289 cached / 0.825 fresh | **0.801968** |
| **Same + phonetic-name answer prompt (live since 03:35 CEST)** | 0.8310 cached / 0.831 fresh | **0.809394** |

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

### Evidence-only extraction and nonlinear ranking (2026-09-18)

Three fresh 9B evidence-only pilots kept the cached answer decisions fixed. The same
six hash-selected development conversations (60 questions, 34 positives) were used:

| Evidence policy | Pilot raw score |
| --- | ---: |
| Current primary evidence | 0.801846 |
| Explicit word pointers | 0.737062 |
| Word pointers + retrieved training examples | 0.696515 |
| Exact quotes + retrieved training examples | 0.777716 |

These configurations are not promoted. Examples came from other conversation folds;
removing each target fold's labels left all 78 prompts unchanged across the two formats.
Seven extractor safeguard tests pass. Full prompts, raw generations, source snapshots,
timings and comparisons: `runs/evidence-pointer-20260918/`. No validation captures or
labels were used in these pilots.

A CPU-only histogram-boosted ranker then used the existing question/boundary features
and word-span candidates. Its 150 shallow trees used fixed settings, equal total weight
per positive question and within-question-centered tIoU targets. Five conversation-disjoint
outer folds used no held-out early stopping. All **390 answer decisions remain unchanged**:

| Cached public-training comparison (39 conversations) | Raw score | Mean tIoU | Zero overlap |
| --- | ---: | ---: | ---: |
| Current primary evidence | 0.753538 | 0.594358 | 31 |
| Broad linear ranker, out of fold | 0.765893 | 0.614949 | 30 |
| Broad nonlinear ranker, out of fold | 0.762057 | 0.608557 | 32 |
| **Primary-boundary nonlinear ranker, out of fold** | **0.764041** | **0.611864** | **30** |

The boundary-only version retains overlap with the primary passage and restricts both
endpoints to within two seconds of the primary endpoints. Its **+0.010504 raw** gain
is positive in all five folds. The descriptive paired-conversation bootstrap interval
is approximately **[0.000003, 0.021396]**; this is a small exploratory signal on reused
training data, not proof of hidden-validation improvement. The all-data fit scores
0.774568 on its training data and is not the out-of-fold result.

This became the integration candidate below; it is **not deployed or ready to submit yet**.
Its boundary-pool gold oracle is only 0.845590 raw, so it cannot reach 0.90 without additional
passage-selection improvements. Artifacts, fold models and the all-data model:
`runs/span-boost-20260918/primary-boundaries/`; broad comparison: `broad/` alongside it.
Scikit-learn and its helper dependencies were installed only under that experiment's
`dependencies/` directory; the serving environment, API and tunnel were not changed.

### Learned-boundary integration (2026-09-18)

`pipeline/boundary.py` now applies the frozen ranker after the existing secondary-no veto.
Its 150 shallow trees are exported as a **66 kB JSON artifact**, without scikit-learn,
joblib, a new model call, or runtime access to training data. Primary overlap and the
two-second endpoint limits remain mandatory. The optional stage checks a 250 ms budget
within the request deadline, with text, word and candidate-work limits; failures retain
the completed primary/veto response without restarting the worker. Secondary features
are identical with capture enabled or disabled, and selected spans are not refined again.
Captures fingerprint both the selector code and the weight file.

| Matched comparison, 39 public-training conversations | Primary raw | Boundary raw, out of fold | Gain |
| --- | ---: | ---: | ---: |
| Original frozen inputs | 0.753538 | 0.764041 | +0.010504 |
| Fresh ASR/model outputs from the unchanged server | 0.751809 | 0.762963 | +0.011154 |

The fresh comparison improves all five folds, changes **none of the 390 answers**
(387 correct), increases mean tIoU **0.591476 → 0.610067**, and reduces zero-overlap
positives **31 → 30**. The descriptive conversation-bootstrap gain interval is
[0.000826, 0.022243]. These are reused training conversations, not a new holdout or
platform validation. The all-data selector scores 0.772381 on these fresh outputs;
that optimistic training-conversation result is **not** the out-of-fold measurement.

The 39 fresh audio requests used the existing live server: zero failures/timeouts,
19.79 s mean and 34.76 s worst HTTP round trip. Their captured worker frames were then
replayed through the candidate's actual FastAPI/runtime path, not another GPU stack.
Final candidate adjustment time was **17.5 ms mean, 31.5 ms worst** on those frames,
with zero skipped adjustments. This is **not a full candidate HTTP latency benchmark**.

All six exported models exactly match the research features, scores and selected spans
on all **17,335 cached candidates**. Both cached and fresh API replays verify all
390 decisions, capture-on/off equivalence and 39 complete captures each. **117 CPU
tests pass**, and the official scoring oracle remains **1.000**. The one final review
identified unbounded text preparation; early text limits and per-term deadline checks
were added, then the complete verification was rerun without changing either score.

Artifacts: `runs/boundary-integration-20260918/` — `model-parity-final.json`,
`cached-api-final/`, `fresh/`, `fresh-api-final/`, `tests-final.log`, `oracle-final.log`.
When these results were recorded the live API/worker were still **98962/98964**, with
source **543cf3e** and validated score **0.743079**. This build has since been deployed;
see [deploy behind the tunnel](#deploy-behind-the-tunnel). A genuine fresh-audio
candidate HTTP rehearsal over all 39 conversations is still required before
recommending a platform submission.

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

Build **`dc734d8` was deployed at 16:46 CEST on 2026-09-18** by restarting Uvicorn under
the existing supervisor, adding the learned boundary selector to the served path. API PID
**4092**, inference worker **4094**; server log: `runs/serve-9054/server-20260918-164600.log`.
The tunnel was not restarted, so the public URL is unchanged; it had reconnected on its own
at 14:40 UTC after a transient `network is unreachable`, and readiness reported one
connection.

A public HTTPS smoke request for `sample_33` returned HTTP 200, **10/10 correct in 21.09
seconds**, without fallback. Its capture confirms the verified source hashes for all six
tracked files, `evidence_policy = primary-with-secondary-veto-and-learned-boundaries`,
and `boundary.status = completed` in **27.9 ms**. The selector changed **no span** on this
conversation, so its score equals the matched primary counterfactual (0.723140); one
conversation neither confirms nor contradicts the out-of-fold gain. Record:
`runs/deploy-dc734d8-20260918/smoke/`, with the check script alongside it.

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

### Rollback to the validated build (2026-09-18)

`dc734d8` was reverted at 17:05 CEST on 2026-09-18, restoring `543cf3e`'s serving code
after less than 20 minutes live. It had no platform validation of its own, and the
deadline allows one final attempt, so the served pipeline was returned to the best score
measured on unseen data. The revert is a commit, not a history rewrite: the selector,
its 117 tests and `boundary_model.json` remain in `dc734d8`.

Only code was reverted. This log keeps the learned-boundary results, because the
experiment is still a live candidate.

Uvicorn was restarted under the existing supervisor; the tunnel was untouched, so the
public URL is unchanged. API PID **5371**, inference worker **5373**; server log:
`runs/serve-9054/server-20260918-170516.log`. The previous API PID was 4092. Nothing
was in flight at shutdown.

Verification via `runs/rollback-543cf3e-20260918/check.py`, a fresh public HTTPS
`sample_33` request:

- `git diff 543cf3e -- . ':!*.md'` is **empty**: every served file is byte-identical
  to the validated build.
- The capture's `evidence_policy` is back to `primary-with-secondary-veto`, the
  `boundary` trace key is **absent** rather than merely inactive, and the fingerprinted
  source list no longer contains `boundary.py` or `boundary_model.json`.
- HTTP 200, **10/10 correct in 18.86 seconds**, no fallback.
- The answers **and all six spans exactly reproduce** the 14:13 CEST public smoke test
  of `543cf3e` (`runs/deploy-543cf3e-20260918/public-smoke-test.json`), which is the
  strongest available evidence that the validated build is serving again.

Record: `runs/rollback-543cf3e-20260918/public-smoke-test.json`. One training
conversation; a smoke test, not platform validation.

### Sentence-table evidence stage (2026-09-18)

Measured against whisper words on the 195 training positives, the gold span is the minimal
self-contained clause: 140 start and end exactly on sentence punctuation, 147 are one
sentence, and a coordinated clause about another fact is cut ("Your blood pressure is
normal," not the whole sentence) while a preceding sentence is kept when the fact
sentence only carries a pronoun ("Your glucose, which is your blood sugar. That was normal
as well."). A doctor question is included only before a bare one- or two-word reply.
Openers are inconsistent: "Overall," is kept in one conversation and dropped in the next.
Where the 9B overlaps the gold it already has the exact start 71% of the time; its
losses are leading acknowledgements, trailing "and ..." clauses, and the wrong one of two
mentions (the gold is usually the mention whose wording the question paraphrases).

Content-aware trimming rules were re-measured on the archived outputs and net under
+0.01: every rule breaks as many spans as it fixes. Candidate oracles show the headroom
instead: best single sentence 0.724 tIoU, one or two sentences 0.807, clause variants of
one to four sentences 0.918, and a local pool of the draft's sentence ±2 plus the three
BM25-closest sentences ±1 reaches 0.859 (raw 0.91 with perfect selection).

`pipeline/stage_b.py` therefore adds a second pass after the answers: the transcript as
numbered sentences, every retained yes with its draft quote, the conventions above with a
synthetic worked example, and one JSON reply of exact quotes. Quotes are aligned with
the numeric matcher, and a multi-sentence quote that skips an intervening sentence is
aligned piecewise so the span bridges it. Only spans of retained yes answers change; the
secondary-no veto still runs afterwards. The stage is skipped once a request is 32 s old
and its output is capped at 320 tokens (median 93; one 9B repetition loop hit 700).

Cached-ASR results on all 39 conversations, answers fixed at 387/390:

| Evidence pass | Mean tIoU | Raw | Zero overlap | Bootstrap gain (39 conversations) |
| --- | ---: | ---: | ---: | --- |
| Primary quote (validated 0.743 build) | 0.5944 | 0.7535 | 31 | |
| Qwen3-8B, local windows, v1 conventions | 0.6114 | 0.7638 | 32 | |
| Qwen3-8B, whole transcript, v2 conventions | 0.6226 | 0.7705 | 27 | +0.028 [0.003, 0.055] |
| Qwen3.5-9B, whole transcript, v2 conventions | 0.6193 | 0.7685 | 31 | +0.025 [-0.017, 0.066] |
| **Qwen3.5-9B, whole transcript, v3 conventions (candidate)** | **0.6282** | **0.7739** | 28 | +0.034 [-0.003, 0.070] |
| Majority vote of draft, 9B v3 and 8B v2 | 0.6346 | 0.7777 | | oracle of the three 0.672 |

Prompt-based refinement saturates near 0.63 tIoU. A Qwen3-8B LoRA on the per-question
window task (137 training quotes, conversation-disjoint fold 0, rank 8, 300 steps at
5e-5) scored 0.597 on its held-out fold against 0.638 for the primary quote and 0.621 for
the unadapted model with the same bare prompt: it learns the conventions but loses to
their inconsistency with this little data, and it is not deployed.

The candidate is `EVIDENCE_MODEL = LLM_MODEL`, `EVIDENCE_MODE = 'refine'` in
`pipeline/mlx_backend.py`; `EVIDENCE_MODEL = None` disables the stage and restores the
validated behaviour exactly. Feeding the cached primary, secondary and evidence generations
through `Pipeline.predict` with a mocked worker reproduces the candidate's 390 outputs with
no per-question mismatch. 115 CPU tests pass, including the evidence frame, budget skip,
adapter loading and a broken stage that keeps the primary answers.

Deployed at 18:10 CEST on 2026-09-18 by restarting Uvicorn under the supervisor; the
tunnel and public URL are unchanged. The fresh 39-conversation rehearsal through the live
server scored **0.769** (accuracy 0.992, mean tIoU 0.620, 387/390 correct) against
0.7518 for the primary build's own fresh run: **0 failed conversations, 0 timeouts**,
round trip 25.7 s mean and **36.0 s worst** (60% of the budget). Stage B took 5.8 s
on average, 8.2 s at most, and was skipped once by the 32-second guard on the longest
conversation, where ASR and the two answer passes alone reached 33 s. It changed 58
of 197 returned yes spans. Record, rehearsal log, experiment scripts and every cached
generation: `runs/deploy-stageb-20260918/`. This is a rehearsal on training audio, not
platform validation; it needs a validation attempt before the final evaluation.

A platform validation attempt ran against this build from 18:35 CEST on 2026-09-18
(19 POSTs from 46.62.240.126, the same conversations as the earlier attempts). All 19
completed with HTTP 200 and no fallback: server time **27.1 s mean, 33.4 s worst**
(`sample_26`). Stage B ran on every request with no skip or error and changed **16 of
91** returned yes spans; the yes count is **91/190**, identical to every earlier attempt,
as answers are untouched. With 95 gold positives each changed span is worth at most
0.0063 raw. Attempt `8a745b9a1afe413aa1f8eebaca875b45` scored **0.7545718384246973**,
**+0.011493** over the primary build and the best platform score so far; captures for
the 19 requests are under `runs/request-captures/` with `server_pid` 8846.

Further selectors tried on the same cached data after that validation, all negative:

| Variant (39 conversations, answers fixed) | Mean tIoU |
| --- | ---: |
| 9B stage B, v3 conventions (deployed) | 0.6282 |
| 8B refining the 9B stage-B output | 0.6272 |
| Multiple choice: 9B picks one of ~8 lettered candidate spans around the 9B draft (oracle 0.827) | 0.5027 |
| Same on the primary draft (oracle 0.817) | 0.4867 |
| 9B stage B with the reasoning channel enabled | unusable: 45 s per call and the 2000-token cap hit before an answer |
| Global start/end shift of the deployed spans | best +0.004 (start -50 ms); no systematic timing offset remains |

The multiple-choice format fails on position bias: the model picks A, B or C in 126 of 194
answers, and in transcript order those are the longest multi-sentence options, so perfect
drafts become three-sentence spans. Deterministic snapping of either draft to its covering
sentence(s) also loses (0.575 / 0.600). The local 9B and 8B are the bottleneck for every
selection formulation, not the candidate pool.

### Larger evidence models (2026-09-18, late evening)

With the server stopped to free memory, the cached `Qwen3.6-35B-A3B` REAP-19B (14 GB MoE)
was run as the stage-B refiner with the same v3 prompt, and as the answering model:

| Configuration (39 conversations, cached ASR) | Correct | Mean tIoU | Raw |
| --- | ---: | ---: | ---: |
| 9B answers + 9B stage B (deployed) | 387 | 0.6282 | 0.7739 |
| **9B answers + REAP stage B** | 387 | **0.6746** | **0.8017** |
| REAP answers, own quotes | 385 | 0.6030 | 0.7567 |
| REAP answers + REAP stage B | 385 | 0.6547 | 0.7877 |

The REAP refinement of the 9B output gains **+0.080 tIoU**, bootstrap [0.032, 0.133] over
conversations, positive in four of five folds, with zero-overlap positives 31 → 23. REAP
generation took 7.0 s mean / 11.3 s max for the answer pass and 9.2 s / 15.2 s for the
refinement. The best row cannot be served on this Mac: REAP (14 GB) beside the 9B (5.6 GB)
leaves no room for ASR and the system; REAP alone for both passes is the serviceable form.

The 27B dense model (`mlx-community/Qwen3.5-27B-4bit`, downloaded 2026-09-18) refined the
9B output on the first 20 conversations from 0.6558 to 0.6925 tIoU, a smaller gain than
REAP, at 24 s mean and 40 s worst per call: unservable within the budget, and not deployed.

**Deployed at 20:33 CEST on 2026-09-18**: `LLM_MODEL` = REAP-19B, `SECOND_LLM_MODEL = None`,
`EVIDENCE_MODEL = LLM_MODEL` (supervisor PID 16243, worker 16249, log
`runs/serve-9054/server-20260918-203255.log`). Smoke on `sample_33`: 10/10 in 16.0 s.
Validation attempt `d2804175c347480fbd9578db8b56b220` scored **0.779780068894168**
(+0.025208 over the 9B stage-B build, +0.036701 over the original 0.743 build) with no
errors; server time per conversation ran 13.9-29.0 s during the attempt.

Variants of the REAP-only pipeline measured afterwards on cached ASR (each needed the
server stopped; the endpoint was down 22:05-22:55 CEST in two windows):

| REAP-only variant (39 conversations) | Correct | Mean tIoU | Raw |
| --- | ---: | ---: | ---: |
| **compact answers + v3 stage B (live)** | 385 | 0.6547 | 0.7877 |
| compact answers + v2 stage B | 385 | 0.6583 | 0.7899 |
| compact answers + v3 stage B without drafts | 385 | 0.6510 | 0.7855 |
| compact answers + v3 stage B on local windows | 385 | 0.6117 | 0.7619 |
| focused answers alone | 385 | 0.6337 | 0.7751 |
| focused answers + v3 stage B | 385 | 0.6497 | 0.7847 |
| Qwen3.5-4B answers alone (2.5 GB, 4.2 s) | 376 | 0.6102 | 0.7517 |
| Qwen3.5-4B answers + REAP v3 stage B | 376 | 0.6654 | 0.7849 |

Every REAP-only variant lands within noise of the live one, so it stays. REAP refines
another model's drafts better than its own (0.675 on the 9B's, 0.665 on the 4B's, 0.655 on
its own): it over-extends its own terse clauses. The 4B's nine extra wrong answers cost
more than that evidence gain returns, and adding a second model beside REAP is a memory
risk on this machine either way.

### Rented GPU (2026-09-19, 00:00-00:45 CEST)

A Vast.ai RTX PRO 6000 Blackwell 96 GB (about $1.2/h) now serves the same pipeline through
`pipeline/vllm_backend.py`: faster-whisper large-v3-turbo on the GPU and `Qwen/Qwen3.8-27B`
in bf16 behind vLLM 0.29 for both the answer pass and the evidence stage (no second model).
Setup, model switch and the three gotchas are in `GPU_SETUP.md`. On the cached transcripts
(`tools/gpu_eval/eval.py`): 27B answers **388/390**; 27B stage B on its own answers
**0.6713 tIoU, raw 0.8008**; on the Mac 9B's drafts **0.6951, raw 0.8140** (gain
[0.052, 0.150] over the drafts). Fresh audio through the box's API, all 39 conversations:
**score 0.804** (accuracy 0.995, mean tIoU 0.677), 0 failures, 0 timeouts, round trip
**16.5 s mean, 34.4 s worst**. Public URL (quick tunnel from the box):
`https://childhood-diana-northeast-defensive.trycloudflare.com/predict`. The Mac keeps
serving the validated REAP build on its own URL as the fallback. Validation attempt
`30ea85ce96294dc8b402ded03651cc0f` (00:04-00:11 CEST) scored **0.792957** with no errors
and 11-15 s per conversation: the best platform score so far.

### The annotators' coordinate system (2026-09-19, from the team's 0.802 branch)

The team's other branch (`medical-appointment-0.802`, readable with the user's GitHub
login) established that the gold spans are **faster-whisper `base`, int8, CPU** word
timestamps (`language='en'`, `word_timestamps=True`, no VAD, decoded from a file on disk):
on their machine all 390 boundaries match bit for bit. `pipeline/base_asr.py` transcribes
that way and marks the transcript `exact_timestamps`, which switches off the speech-onset
shift and the reply extension (both were corrections for Whisper-turbo timings) and makes
stage B split sentences on end punctuation only, the unit the annotators quoted in.

On the rented Threadripper the boundaries are **not** bit-exact: 232/390 exact, 79% within
20 ms. A ctranslate2 sweep over `CT2_FORCE_CPU_ISA` (auto/AVX2/AVX/GENERIC), `CT2_USE_MKL`
and the compute type showed this box's defaults are already its best; forcing MKL is worse.
The drift costs little: the word-span oracle is **0.9859 tIoU** against 0.9260 for the turbo
words, and 65 of the 195 gold spans become exactly reachable.

| Qwen3.8-27B on the 39 training conversations | Correct | Mean tIoU | Raw |
| --- | ---: | ---: | ---: |
| turbo words, answers only | 388 | 0.6059 | 0.7615 |
| turbo words, answers + stage B | 388 | 0.6713 | 0.8008 |
| base words, answers only | 388 | 0.6283 | 0.7749 |
| **base words, answers + stage B (v3 prompt)** | 388 | **0.7129** | **0.8257** |

Both remaining answer errors are missed positives on phonetically mangled names ("Aromere"
for Airomir, "molluscs" for molluscum), so `MEDICAL_ANSWER_PROMPT=named` adds the
transcript's real sound-alike pairs to the answering rules. `MEDICAL_EVIDENCE_PROMPT=annot`
(their measured quote convention, with their quotes as examples) and `EVIDENCE_MODE=perq`
(one call per yes question over the whole transcript, run in parallel against the server)
are the other two candidates under measurement.

Evidence variants, all on base words with the 27B answering (388/390 correct in every row;
the second column leaves out the seven conversations the `annot` prompt quotes examples from):

| Evidence pass | Mean tIoU | Held-out tIoU | Raw |
| --- | ---: | ---: | ---: |
| none (the answer pass's own quotes) | 0.6283 | | 0.7749 |
| batched JSON, `annot` prompt | 0.6723 | 0.6590 | 0.8013 |
| per question, `annot` prompt | 0.6816 | 0.6690 | 0.8069 |
| batched JSON, `v3` prompt | 0.7129 | 0.7140 | 0.8257 |
| **per question, `v3` prompt (live)** | **0.7182** | **0.7187** | **0.8289** |

Transcribing the 0.802 branch's prompt (`annot`) into our pipeline is clearly worse than
our own conventions, batched or not; asking one question at a time over the whole
transcript is worth about +0.005 tIoU either way. **Deployed at 02:05 CEST on 2026-09-19**
(`MEDICAL_ASR_MODE=base MEDICAL_EVIDENCE_MODE=perq MEDICAL_EVIDENCE_PROMPT=v3`): the fresh
39-conversation run scores **0.825** (accuracy 0.995, mean tIoU 0.712), 0 failures,
0 timeouts, round trip **12.0 s mean, 19.1 s worst** (32% of the budget at worst). Validation
attempt `593f9bbca49744969f29b305d5bc7220` (00:46-00:50 CEST) scored **0.801968**, up from
0.792957 for the turbo-coordinate build and the best platform score so far.

Where the remaining tIoU sits, measured on the live configuration's spans: 107 of 195 are
near exact, 35 stop short of the gold, 17 overrun it, 14 are offset, 20 miss the passage
entirely and 2 have no span. Widening every span to whole sentences scores **0.6762**, below
the model's own trimming, so the quotes stay as cut. Nearly every fully-missed span is a
*different valid mention* of the same fact, and in each case the gold is the mention whose
wording the question echoes — the annotator wrote the question while reading it. Two
candidate prompts addressed exactly those buckets: `match` (wording decides between
mentions) and `exchange` (same, plus take the whole contiguous exchange that settles the
fact). Both lose, batched and per question:

| Evidence prompt, 27B on base words | Batched | Per question |
| --- | ---: | ---: |
| **v3 (live)** | 0.7129 | **0.7182** |
| match | 0.6739 | 0.7068 |
| exchange | 0.6745 | 0.6960 |
| annot | 0.6723 | 0.6816 |

### What does not work, measured (2026-09-19)

Three independent attempts to beat the generated span all fail, and they fail the same way.

**Bit-exact timestamps are not the gap.** On the 98 of 193 spans where we already pick
exactly the annotator's words we score **0.9813**, so reproducing their kernels bit for bit
is worth **+0.006 raw**. The drift is symmetric (median 0.000, mostly ±20 ms), so no global
shift helps, and `check_timestamps.py --fingerprint` shows this box matching their reference
on sample_4. The remaining loss is word *selection*: the other 95 spans average 0.46.

**Candidate verification loses badly.** A pool of contiguous passages around the pick
(±2 sentences, up to 3 joined, plus clause cuts; 16 per question) has an oracle of
**0.8619 tIoU, raw 0.9152**, so the right span is almost always in reach. Scoring each
candidate in its own request — "does this passage on its own establish the answer?", P(yes)
from the reply token's logprobs — and keeping the shortest that clears the bar scores
**0.6531** at its best threshold, against 0.7182 for the anchor, and raises zero-overlap
from 22 to 34. Re-ranking the cached scores every other way is worse still: highest P(yes)
collapses to **0.4035**, a length-penalised score peaks at 0.6482, and using the verifier
only as a veto on the anchor peaks at 0.6430. The model's entailment judgement cannot
separate two passages that both contain the fact, which is the same failure as the earlier
lettered multiple-choice attempt. `pipeline/verify.py` keeps the implementation and its 144
tests; it is not in the serving path.

**Reasoning before answering makes it worse.** The Mac could not afford it (45 s a call);
on the server, with the questions in flight together, it is affordable. With
`MEDICAL_EVIDENCE_THINKING=1` on the per-question v3 evidence pass the answers stay perfect
at 390/390 but mean tIoU falls **0.7183 -> 0.6588** (raw 0.8310 -> 0.7953), with zero-overlap
spans rising 20 -> 25. Deliberation moves the model off the passage its first instinct
picked, which is the same failure as verification and re-ranking.

**Ensembling the prompts does not help either.** The medoid span across up to five cached
prompt variants scores 0.7185, against 0.7182 for v3 alone, even though the oracle over
those same variants is **0.8354**. Every unsupervised way of choosing among good candidates
has now failed; only generation picks well.

**The phonetic-name answer prompt is a clean win.** `MEDICAL_ANSWER_PROMPT=named` adds the
transcript's real sound-alike pairs to the answering rules and takes the answer pass to
**390/390**, recovering both missed positives (Airomir heard as "Aromere", molluscum as
"molluscs") with the yes count exactly matching the gold balance. With the per-question v3
evidence stage on top: **390/390 correct, tIoU 0.7183, raw 0.8310** (0.8312 on the held-out
subset). Deployed at 03:35 CEST; the fresh 39-conversation run through the endpoint scores **0.831** with accuracy **1.000**, every returned yes carrying a span, 0 failures, 0 timeouts, round trip 12.2 s mean and 19.6 s worst. Validation attempt `c89957bd05be475dbb6d1eb5115ba9c7` (01:20-01:24 CEST) scored **0.809394**, +0.007427 over the build without the rule and the best so far.

A second attempt on the *same* build minutes later scored 0.793232, and the cause is in the
server log, not the model: an offline experiment was sharing the GPU, six of the nineteen
conversations took 35-46 s instead of 12, and generation alone reached 40 s. Past
`EVIDENCE_BUDGET_SECONDS` the evidence stage is skipped and those spans fall back to the
answer pass's own quotes, which predicts about -0.017; the observed difference was -0.0161.
Three identical requests to the idle endpoint return byte-identical responses in 11.7 s, so
the build itself is deterministic. Two lessons, both now acted on: **never run offline GPU
work while the endpoint can be called**, and the budget was raised 32 s -> 40 s
(`MEDICAL_EVIDENCE_BUDGET`) with a loud warning logged whenever the stage is skipped, so a
silent degradation can no longer be mistaken for a worse model.

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
