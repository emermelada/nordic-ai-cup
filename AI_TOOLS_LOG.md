# AI tools, models, APIs and datasets used


| Date | Who | Challenge | Tool / model / API / dataset | What for |
|---|---|---|---|---|
| 2026-09-15 | J (B) | infra | Claude Code (claude-opus-5) | Repo scaffold, Dockerfiles, compose, scripts |
| 2026-09-17 | Z | survival-simulator | Hermes Agent (deepseek-v4.1-flash via OpenRouter) | Sim source analysis, controller design + evolutionary parameter search, evaluation harness + determinism fixes, end-game diagnostics, VPS deployment and scoring |
| 2026-09-17 | Z | survival-simulator | Hermes Agent subagents (same model) | Parallel workstreams: parameter/robustness sweeps, imitation-learning pipeline, behaviour-preserving CPU optimisation |
| 2026-09-15 | J | drone-flyby | Ultralytics YOLO11 (n/s/m), COCO-pretrained | Detector; fine-tuned into v1-v6 |
| 2026-09-16 | J | drone-flyby | Kaggle `sagar100rathod/inria-aerial-image-labeling-dataset` | Aerial photo backgrounds for the synthetic training set |
| 2026-09-16 | J | drone-flyby | Kaggle `adrianboguszewski/landcoverai` | Aerial photo backgrounds for the synthetic training set |
| 2026-09-16 | J | drone-flyby | Kaggle GPU (T4 x2) | Training runs v1-v4 |
| 2026-09-17 | J | drone-flyby | Rented GPU (RTX 5090, Vast.ai) | Training runs v5, v6 |
| 2026-09-17 | J | drone-flyby | cloudflared quick tunnel | Exposing the local service to the evaluator |
| 2026-09-18 | J | drone-flyby | Claude Code (claude-opus-5) | Offline scoring tools, model comparison, service review fixes |
| 2026-09-18 | J | drone-flyby | pycocotools / faster-coco-eval | Offline mAP scoring against recorded runs |
| 2026-09-19 | F | drone-flyby | Claude Code (claude-opus-5) | Diagnosis and measurement: per-class AP, camera-pattern search, box-geometry and truth-file analysis, serving and validation runs |
| 2026-09-19 | F | drone-flyby | Ultralytics YOLO11 `yolo11m-p2` architecture (P2 head) | v8 / v9 detectors, trained from the same COCO-pretrained weights |
| 2026-09-19 | team (unconfirmed) | survival-simulator | `cma` (CMA-ES), `numba`, `numpy` | Evolutionary policy search and a JIT fast simulator, bit-exact with the supplied one |
| 2026-09-18 | F | medical-appointment | Claude Code (claude-opus-5) | Found that the evidence labels are faster-whisper `base` int8 word timestamps; exact-timestamp ASR, evidence mapper, offline harness, plan |
| 2026-09-18 | F | medical-appointment | faster-whisper `base` (Systran/faster-whisper-base, int8, CPU) | Served ASR: word timestamps in the annotators' coordinate system. Also compared small, small.en, base.en, medium.en, large-v3 to identify it |
| 2026-09-18 | F | medical-appointment | Qwen3.6-35B-A3B (nvidia/Qwen3.6-35B-A3B-NVFP4) via vLLM, on our rented GPU | Answering model behind the endpoint (llm.py) in the **0.802-era build**, running locally on the serving box. Superseded before the graded evaluation by the Qwen3.8-27B row below |
| 2026-09-17 → 2026-09-21 | A (Fuxiv) | medical-appointment | **OpenAI Codex** (hosted coding agent) | The workstream that produced the graded 0.822 build: service code, the evidence-extractor and span-ranker training on the rented GPU (branch originally `codex/evidence-extractor-training`, remote workspace originally `/workspace/medical-evidence-training-codex`), the offline harness and the analysis documents. Recorded in `medical-appointment/HANDOFF_EXTERNAL_TRAINING.md` (2026-09-19). Development tooling only — Codex is not in the `/predict` path. Start date is the first commit of this workstream; A to confirm the exact first use |
| 2026-09-20 | A (Fuxiv) | medical-appointment | **Qwen/Qwen3.8-27B**, bf16, via vLLM on our rented GPU (RTX PRO 6000 Blackwell Max-Q, Vast.ai) | Answering model and stage-B evidence pass behind the endpoint in the **graded 0.822 build**. Runs locally on our own serving box, not a hosted API. Pinned in `medical-appointment/REPRODUCE.md` |
| 2026-09-20 | A (Fuxiv) | medical-appointment | **gpt-6-astra** (hosted API) | Development only: scored as one of seven evidence-span producers in the frontier probe (`runs/frontier-probe-20260920`), measured at 0.697194 local tIoU and **not deployed** — it lost to the local 27B. Never in the `/predict` path |
| 2026-09-20 | A (Fuxiv) | medical-appointment | **claude-opus-5** (hosted API, not Claude Code) | Development only: scored as an evidence-span producer (0.644087, and 0.625238 with 24 gold examples) and as a local extent refiner (−0.013545) in the same frontier probe. **Not deployed**; never in the `/predict` path |
| 2026-09-19 → 2026-09-20 | A (Fuxiv) | medical-appointment | DeBERTa span extractor and listwise span ranker, fine-tuned by us on the rented GPU | Two of the three evidence producers in the deployed medoid vote. Recipe in `medical-appointment/REPRODUCE.md`; external pretraining corpora in `medical-appointment/tools/evidence_training/EXTERNAL_DATA.md` |

Rows marked *unconfirmed* were reconstructed from the code's imports; the owner should confirm date and author.

## Disclosure notes

- **Hosted models were used during development and never at request time.** The rules
  forbid a hosted LLM in the `/predict` path. Everything that answers a grader request —
  the Qwen3.8-27B, faster-whisper, the fine-tuned extractor and ranker, the YOLO11
  detectors, the survival policy — runs on hardware we rented or own. The two hosted
  frontier models above (`gpt-6-astra`, `claude-opus-5`) were scored offline as candidate
  evidence producers, both lost to the local model, and neither was deployed. The
  measurements are in `medical-appointment/SCORE_IMPROVEMENT_PLAN.md`.
- **Commit `72952fe` (2026-09-21) renamed tooling out of paths, prose and a branch name.**
  It renamed `/workspace/medical-evidence-training-codex` to
  `/workspace/medical-evidence-training`, renamed the branch
  `codex/evidence-extractor-training` to `medical-appointment-0.822`, and reworded four
  sentences that named the coding agent. No code and no measured result changed. That
  commit was a tidying pass, but it had the effect of removing the only record of which
  coding agent did the work, and it was made *before* this log named Codex. The rows above
  restore that disclosure. The commit is left in history unmodified.
- **Sixteen commits on `main` land after the 20 Sep 16:00 deadline** (10 of them non-merge).
  They are documentation and provenance, not new solution work: `SUBMISSIONS.md`,
  `REPRODUCE.md`, `SOLUTION_JOURNEY.md`, the frozen inputs the tests need, and the merges
  that brought each challenge's graded code onto `main` from the branch it was served from.
  The largest of them, `739a6ef`, replaces `medical-appointment/` on `main` with the code
  that was actually running during the graded evaluation, in place of the older 0.802 copy
  that `main` had been carrying; its message says so, and the replaced files remain on
  `medical-appointment-0.802` and `medical-appointment-exact-evidence`. Nothing served to
  the grader was changed after the deadline — the graded builds are pinned in
  `SUBMISSIONS.md`, `medical-appointment/REPRODUCE.md`,
  `drone-flyby/SERVED_CONFIG.md` and `survival-v2/EVALUATED.md`.
- **The Who column.** Rows are signed either by name initial or by the role letter from the
  original team plan, and the two schemes overlap. The people are: **A** (`Fuxiv`) —
  Medical Appointment; **B** (`J`, `emermelada`) — Drone Flyby; **C** (`Z`, `Zaitzev`) —
  Survival Simulator; **D** — supporting all three. **F** (`Franciszek Kossut`) is a name
  initial, not the role letter, and appears on both medical and drone rows.
