# Handoff: medical-appointment (Nordic AI Cup 2026)

Deadline **Sun 2026-09-20 16:00 CEST**, one evaluation attempt. Read `README.md` for the
protocol, then `PLAN.md` sections 0, 4, 7.0, 8 and 12. The plan is grounded in
measurements made on this machine on 2026-09-17; the scripts are in `tools/probes/`.

## Task in one line
POST with an MP3 (base64) + 10 yes/no questions → 10 booleans + a (start, end) span in
seconds for every yes. Score = 0.4 × accuracy + 0.6 × mean tIoU over gold-yes questions.
60 s per request, no cloud APIs at inference, never raise, lists must be length 10.

## Where we are (measured on the 39 training conversations)
- ASR: `mlx-community/whisper-large-v3-turbo` via `mlx-whisper`, word timestamps, 3–9 s per
  conversation, transcripts near-verbatim. Word-span ceiling for tIoU = 0.93.
- Answering: `mlx-community/Qwen3-8B-4bit` via `mlx-lm`, one JSON call for all 10
  questions, thinking off → **386/390 correct**, 0 JSON failures, ~12 s. Prompt is
  `SYSTEM` in `tools/probes/llm_probe.py`.
- Evidence: LLM quote fuzzy-aligned to whisper words → mean tIoU **0.57**, score **0.72**.
  Loss: 30/195 quotes cite a different mention than the annotator; the rest are a
  sentence too long or a clause too short in equal numbers. Post-hoc trimming rules are
  worth < 0.01. See PLAN.md §8 for the fix (evidence prompt with candidate quotes,
  passage re-ranking, per-question evidence pass).
- No pipeline code exists yet: `example.py` is still the all-true baseline.

## Environment
- Use `/usr/local/bin/python3.11 -m venv .venv311` (the checked-in `.venv` is Python 3.14,
  no ML wheels). `pip install -r requirements.txt mlx-whisper mlx-lm "omi-med-stt[mlx]"
  rank_bm25 rapidfuzz json-repair`.
- Apple M4 Pro, 24 GB. ffmpeg at /opt/homebrew/bin. Models already in the HF cache:
  whisper-large-v3-turbo, Qwen3-8B-4bit, omi-health/omi-med-stt-v1-mlx-q8 (medical
  Parakeet, ~1 s per file; load it via `omi_stt.mlx_runtime._load_model`, stock
  parakeet-mlx rejects its weights).
- `transcripts/` and `models/` are gitignored; cache transcripts there.

## Do in this order
1. `pipeline/` per PLAN.md §4.1: audio decode, ASR backend, clause units (§6), answer
   (§7.1 prompt, JSON repair, retrieval fallback §7.3), evidence (fuzzy quote → words,
   +0.1 s start pad), watchdog (soft 45 s / hard 52 s), warm-up at import.
2. `tools/transcribe_all.py` (cache) and `tools/eval_offline.py` (same report format as
   `local_evaluator.py`, plus per-question CSV). Reproduce 0.99 / 0.57 / 0.72 offline.
3. Wire `example.py`, run `python api.py` + `python local_evaluator.py --verbose`:
   39/39 answered, worst latency < 25 s.
4. Expose (`brew install cloudflared`, `cloudflared tunnel --url http://localhost:9054`),
   submit `https://<host>/predict`, run a platform validation attempt, read its accuracy.
5. Improve tIoU (PLAN.md §8.2, in order): new quote instruction + up to 3 candidate
   quotes → BM25 candidates + re-ranker (lexical overlap with question terms, contains
   the question's number/entity, length prior, speaker guess, LLM rank; fit on the 195
   positives with leave-one-conversation-out) → per-question evidence pass for
   ambiguous cases → length calibration. Measure every step offline before shipping.
6. Fix the known answer misses: put unit/name variants in the prompt (mmol/mol =
   millimoles per mole; molluscs = molluscum) and use Parakeet's normalised text for the
   number check.

## Gotchas
- Gold spans are one clause (median 2.9 s); returning whole segments caps tIoU at 0.62.
- Questions use abbreviations (HbA1c, ECG, BMI, TSH, mmol/L) that the audio says in
  full; Danish brand names are transcribed phonetically (Panodil→panadil,
  Airomir→Aromir, Activelle→Activel, Esomeprazole→Isomeprazole): fuzzy-match entities.
- 21 questions are tag questions; 27 positives are phrased as an absence.
- A wrong-length list or an exception costs all 10 questions; five timeouts end the run.
- Don't overfit 39 conversations: few knobs, LOO tuning, trust the platform validation.

## Reproduce the numbers
```bash
python tools/probes/transcribe_whisper.py transcripts/turbo mlx-community/whisper-large-v3-turbo
python tools/probes/analyze_alignment.py transcripts/turbo
python tools/probes/llm_dump.py transcripts/turbo mlx-community/Qwen3-8B-4bit llm_dump.json
python tools/probes/align_experiments.py llm_dump.json --worst
```
