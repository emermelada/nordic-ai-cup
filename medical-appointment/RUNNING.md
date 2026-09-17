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

- `mlx-community/whisper-large-v3-turbo-8bit` (serving ASR, staged locally; see below)
- `mlx-community/gpt-oss-20b-MXFP4-Q8` (serving answers)
- `mlx-community/Qwen3.5-9B-4bit`, `mlx-community/Qwen3-8B-4bit`, `whisper-large-v3-turbo` (earlier builds)

mlx-whisper loads `weights.safetensors`, but the 8-bit repo ships `model.safetensors`,
so stage it once into the gitignored `models/` directory:

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

Whisper-large-v3-turbo (8-bit) supplies word timestamps; gpt-oss-20b answers all
questions in one deterministic call at low reasoning effort. Its reasoning channel is
stripped before parsing, and 1200 output tokens cover the 769 it needed at most. The `compact` prompt
keeps the original answering rules but asks for one-line JSON with unit ids and a
quote per yes, about a third of the original output tokens. Decimal-aware alignment
maps the quote back to words without confusing `2.5` with `25`. Two span rules then
apply to every yes:

- a quote that ends in a question is extended to a reply of at most six words;
- the start moves to the first 10 ms frame above -45 dBFS, because Whisper often
  starts a word inside the preceding pause while the annotations start at speech.

The model and prompt are `QWEN_MODEL` and `DEFAULT_PROMPT` in
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
| gpt-oss-20b, compact + span rules + 8-bit ASR | 0.760 | not yet validated |

Qwen3.8-27B (16 GB) scored 0.755 offline but cannot serve here: with Whisper loaded it
evicts its own weights, ASR slows from 9 s to 16 s, and requests miss the deadline. The
resulting swapping also killed the cloudflared tunnel once. Keep the answering model
near 12 GB on this machine.

Training-set scores did not predict validation: the prompt rules were written from
Qwen3-8B errors on those conversations. Choose builds by platform validation.
Offline, a LoRA on the training spans regressed on held-out folds and a worked
example in the prompt traded accuracy for spans; neither is in the serving path.

### Deploy behind the tunnel

The cloudflared quick tunnel forwards to 127.0.0.1:9054; restarting cloudflared
changes the public hostname. To deploy, check the latest `runs/serve-9054/` log for
in-flight platform requests, then restart only uvicorn:

```bash
kill -INT $(lsof -tiTCP:9054 -sTCP:LISTEN)
nohup .venv311/bin/python -m uvicorn api:app --host 127.0.0.1 --port 9054 --workers 1 \
  > runs/serve-9054/server-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null & disown
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
