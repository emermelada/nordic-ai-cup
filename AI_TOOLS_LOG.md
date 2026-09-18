# AI tools, models, APIs and datasets used


| Date | Who | Challenge | Tool / model / API / dataset | What for |
|---|---|---|---|---|
| 2026-09-15 | D | infra | Claude Code (claude-opus-5) | Repo scaffold, Dockerfiles, compose, scripts |
| 2026-09-17 | Z | survival-simulator | Hermes Agent (deepseek-v4.1-flash via OpenRouter) | Sim source analysis, controller design + evolutionary parameter search, evaluation harness + determinism fixes, end-game diagnostics, VPS deployment and scoring |
| 2026-09-17 | Z | survival-simulator | Hermes Agent subagents (same model) | Parallel workstreams: parameter/robustness sweeps, imitation-learning pipeline, behaviour-preserving CPU optimisation |
| 2026-09-18 | F | medical-appointment | Claude Code (claude-opus-5) | Found that the evidence labels are faster-whisper `base` int8 word timestamps; exact-timestamp ASR, evidence mapper, offline harness, plan |
| 2026-09-18 | F | medical-appointment | faster-whisper `base` (Systran/faster-whisper-base, int8, CPU) | Served ASR: word timestamps in the annotators' coordinate system. Also compared small, small.en, base.en, medium.en, large-v3 to identify it |
