# Local MLX implementation

Use native Apple Silicon and Python 3.11. The original `.venv` uses Python 3.14;
keep it separate from this environment.

```bash
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements-mlx.txt
.venv311/bin/python -m unittest discover -s tests -v
.venv311/bin/python local_evaluator.py --oracle
```

Both model snapshots must already be complete in the local Hugging Face cache:

- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/Qwen3-8B-4bit`

Startup resolves cached snapshots with `local_files_only=True`; missing weights
fail startup rather than downloading them. No hosted APIs are used for inference.

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

## Verified measurements

The 59 CPU tests pass, including worker hangs/crashes, recovery, partial IPC,
concurrent requests, parser failures, graceful shutdown, and offline-cache
validation. The official scoring oracle returns 1.000. Fresh transcription,
compatible-cache resume, and fresh offline Qwen generation were also exercised.
Real-model warm-up and shutdown completed without leaving the inference worker.

Full HTTP evaluation, 39 conversations / 390 questions:

- Accuracy: **386/390 (0.990)**.
- Mean tIoU over all 195 gold-yes questions: **0.567**.
- Weighted score: **0.736**.
- Failed conversations / timeouts: **0 / 0**.
- Mean / worst round trip: **16.495 / 27.224 seconds**.

The worst request was sample 79 (8.36 seconds ASR, 18.80 seconds generation).
It misses the aspirational 25-second target, but stays well below 60 seconds.
The first smoke request, sample 17, answered 10/10 correctly in 13.46 seconds.
Per-question HTTP results are in `runs/http-evaluation.log`.

Separately, replay of the previously cached Qwen outputs reproduced **386/390**,
**0.571 mean tIoU**, and **0.738 score**, with 30 zero-overlap positives including
two missing spans. Its artifacts are in `runs/baseline-replay/`.

Cached replay is not a fresh ASR or HTTP benchmark. Historical generation timings
exclude transcription. Offline reports separate cached timing, fresh generation,
and CPU postprocessing; only the HTTP evaluator measures full requests.
