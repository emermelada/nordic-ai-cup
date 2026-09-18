# AI tools, models, APIs and datasets used

Keep this current as we go; we may be asked to disclose it.

| Date | Who | Challenge | Tool / model / API / dataset | What for |
|---|---|---|---|---|
| 2026-09-15 | D | infra | Claude Code (claude-opus-5) | Repo scaffold, Dockerfiles, compose, scripts |
| 2026-09-15 | J | drone-flyby | Ultralytics YOLO11 (n/s/m), COCO-pretrained | Detector; fine-tuned into v1-v6 |
| 2026-09-16 | J | drone-flyby | Kaggle `sagar100rathod/inria-aerial-image-labeling-dataset` | Aerial photo backgrounds for the synthetic training set |
| 2026-09-16 | J | drone-flyby | Kaggle `adrianboguszewski/landcoverai` | Aerial photo backgrounds for the synthetic training set |
| 2026-09-16 | J | drone-flyby | Kaggle GPU (T4 x2) | Training runs v1-v4 |
| 2026-09-17 | J | drone-flyby | Rented GPU (RTX 5090, Vast.ai) | Training runs v5, v6 |
| 2026-09-17 | J | drone-flyby | cloudflared quick tunnel | Exposing the local service to the evaluator |
| 2026-09-18 | J | drone-flyby | Claude Code (claude-opus-5) | Offline scoring tools, model comparison, service review fixes |
| 2026-09-18 | J | drone-flyby | pycocotools / faster-coco-eval | Offline mAP scoring against recorded runs |
