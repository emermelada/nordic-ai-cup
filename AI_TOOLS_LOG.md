# AI tools, models, APIs and datasets used


| Date | Who | Challenge | Tool / model / API / dataset | What for |
|---|---|---|---|---|
| 2026-09-15 | D | infra | Claude Code (claude-opus-5) | Repo scaffold, Dockerfiles, compose, scripts |
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
| 2026-09-19 | team (unconfirmed) | medical-appointment | faster-whisper 1.2.1 (Whisper `base`), ctranslate2 4.8.2, av 18.1.0 | Speech-to-text with word timestamps; versions pinned because that stack reproduces the annotators' timestamps 390/390 |
| 2026-09-19 | team (unconfirmed) | survival-simulator | `cma` (CMA-ES), `numba`, `numpy` | Evolutionary policy search and a JIT fast simulator, bit-exact with the supplied one |

Rows marked *unconfirmed* were reconstructed from the code's imports; the owner should confirm date and author.
