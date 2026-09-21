# Probe scripts behind the numbers in PLAN.md

Run from a Python 3.11 venv with `mlx-whisper`, `omi-med-stt[mlx]`, `mlx-lm`, `rank_bm25`, `rapidfuzz` installed (Apple Silicon).

```bash
python tools/probes/transcribe_whisper.py transcripts/turbo mlx-community/whisper-large-v3-turbo   # word timestamps, all 39 files
python tools/probes/transcribe_omi.py     transcripts/parakeet_med                                   # medical Parakeet, token timestamps
python tools/probes/analyze_alignment.py  transcripts/turbo          # localisation ceilings: segment / merged / word span; boundary offsets
python tools/probes/units_probe.py        transcripts/turbo          # clause-splitting thresholds vs best-unit ceilings
python tools/probes/show_gold_text.py     transcripts/turbo sample_6 # the words inside each gold span
python tools/probes/retrieval_baseline.py transcripts/turbo          # BM25 + number rule, no LLM
python tools/probes/llm_probe.py          transcripts/turbo mlx-community/Qwen3-8B-4bit 39   # one JSON call per conversation
```

Transcript JSON format (one file per conversation): `{"file", "model", "seconds", "text", "segments": [{"start", "end", "text", "words": [{"word", "start", "end", "p"}]}]}`.

```bash
python tools/probes/llm_dump.py         transcripts/turbo mlx-community/Qwen3-8B-4bit llm_dump.json   # raw LLM outputs per conversation
python tools/probes/align_experiments.py llm_dump.json --worst                                        # span strategies + worst cases
```

```bash
python tools/probes/rules_experiment.py llm_dump.json   # too-long/too-short split and post-hoc span rules (trim / extend / pad)
```
