# Nordic AI Cup 2026 — Elysa's Secret

Team **Elysa's Secret**, University of Southern Denmark. Three challenges, three independent
services, one graded evaluation attempt each.

| Use case | Evaluation | Denmark rank | Points | Code | What was graded |
|---|---|---|---|---|---|
| Medical Appointment | **0.8222** | 1st | 25 | [`medical-appointment/`](medical-appointment/) | [`REPRODUCE.md`](medical-appointment/REPRODUCE.md) |
| Survival Simulator | **1405.256** | 4th | 12 | [`survival-v2/`](survival-v2/) | [`EVALUATED.md`](survival-v2/EVALUATED.md) |
| Drone Flyby | **0.2630** | 6th | 8 | [`drone-flyby/`](drone-flyby/) | [`SERVED_CONFIG.md`](drone-flyby/SERVED_CONFIG.md) |

**45 points, 1st in Denmark.** Full results and provenance: [`SUBMISSIONS.md`](SUBMISSIONS.md).

---

## Start here

- **What we submitted and where each graded build lives** → [`SUBMISSIONS.md`](SUBMISSIONS.md)
- **Every AI tool, model, API and dataset we used** → [`AI_TOOLS_LOG.md`](AI_TOOLS_LOG.md)
- **How the Survival solution was reasoned out** → [`SOLUTION_JOURNEY.md`](SOLUTION_JOURNEY.md)

Each challenge folder has its own entry document, listed in the sections below. Read the
"what was graded" file first — it pins the exact configuration behind the score, separately
from the much larger pile of notes about things that did not work.

## How a submission works

The grader calls an HTTP endpoint we host and scores the responses. Nothing is precomputed:
every service runs a real model on the request it is given, on hardware we own or rent. There
is no stored table of validation or evaluation answers anywhere in this repository.

```
 grader ──HTTPS──▶ public URL ──▶ our machine :PORT ──▶ uvicorn ──▶ api.py
 (Ambolt)          (tunnel or VPS)                                  │
                                                     dtos.py validates the request
                                                     predict() runs the model
                                                     dtos.py shapes the response
```

Medical Appointment and Drone Flyby serve `POST /predict` on container port 8000 (host 8001 and
8002). Survival Simulator is different: the organisers' simulator calls it every tick, and it
listens on **9052**.

---

## Medical Appointment — 0.8222, 1st in Denmark

One request carries a consultation's audio and ten yes/no questions. We return ten booleans and,
for each `yes`, the stretch of audio that answers it. Score is
`0.4 × answer accuracy + 0.6 × mean temporal IoU`; hidden-set answer accuracy was **1.0000**, so
the whole remaining margin is span localisation.

The pipeline transcribes on the CPU with faster-whisper `base` — deliberately small, because the
gold spans are in *that recogniser's* coordinate system — answers all ten questions in one
Qwen3.8-27B call, re-asks each `no` alone and flips it above a threshold, then picks each evidence
span as the **medoid of three independently trained producers**: the 27B's own quote, a fine-tuned
DeBERTa span extractor, and a listwise span ranker over an enumerated candidate pool.

- **Reproduce it:** [`medical-appointment/REPRODUCE.md`](medical-appointment/REPRODUCE.md) — exact
  model, vLLM arguments, GPU, ASR configuration, serving commands, and the retraining recipe for
  both checkpoints. It also shows how to verify the reported numbers with no GPU and no weights.
- **Operate it:** [`medical-appointment/RUNNING.md`](medical-appointment/RUNNING.md)
- **What was tried and rejected:**
  [`medical-appointment/SCORE_IMPROVEMENT_PLAN.md`](medical-appointment/SCORE_IMPROVEMENT_PLAN.md) —
  thirteen failed selection mechanisms, and why no selector can work here.
- **Not in the repository:** the two trained span checkpoints, ~1.7 GB each. `REPRODUCE.md`
  rebuilds them.

## Survival Simulator — 1405.256

A colony of agents foraging under predation. Reading the simulator source first showed that the
`score += dt` survival bonus is constant with respect to strategy, so the real learnable margin is
**fruit intake minus predation loss** — a foraging and evasion problem, not a survival one. A PPO
feasibility spike ([`spikes/001-ppo-feasibility/`](spikes/001-ppo-feasibility/)) invalidated
from-scratch RL inside the time budget, so the deployed policy is a shared-map colony controller
tuned by evolutionary search, alongside a numba simulator that matches the official one tick for
tick — positions, energy, score and RNG state.

- **Exactly what was graded:** [`survival-v2/EVALUATED.md`](survival-v2/EVALUATED.md) — file
  hashes, the serving command, the three per-game scores, and the honest limits.
- **Measured results:** [`survival-v2/RESULTS.md`](survival-v2/RESULTS.md)
- **How we got there:** [`SOLUTION_JOURNEY.md`](SOLUTION_JOURNEY.md)

## Drone Flyby — 0.2630

A drone flies over terrain and we report what is on the ground and where, in real time, under a
per-frame deadline. The served build runs **five YOLO11 passes per frame** at mixed scales, a fixed
six-quadrant camera sweep with a whole-frame look after every step, ground motion refitted online
during the flight, and one flat box-growth factor for every class.

Validation mean was **0.5841** over 7 runs and the evaluation **0.2630** on an unseen flight, with
all 249 frames answered. That gap is the honest headline of this challenge: roughly 45 % of the
validation score carried over, because much of the tuning fitted one validation flight.

- **Exactly what was served:** [`drone-flyby/SERVED_CONFIG.md`](drone-flyby/SERVED_CONFIG.md),
  launched by [`drone-flyby/configs/evaluated_serve_env.sh`](drone-flyby/configs/evaluated_serve_env.sh).
  It also lists what was tried on the real grader and **rejected** — per-class box geometry, a sixth
  pass, Level-2 zoom, per-class thresholds — because those were fits to the validation flight.
- **Why not higher:** [`drone-flyby/WHY_NOT_0.7.md`](drone-flyby/WHY_NOT_0.7.md)
- **Operating runbook:** [`drone-flyby/RUNBOOK.md`](drone-flyby/RUNBOOK.md)
- Detector weights **are** committed under `drone-flyby/models/`. The 3.4 GB of recorded validation
  frames used for offline replay are not — they lived on a rented GPU instance that has been shut down.

---

## Running a service

```bash
docker compose up -d --build medical-appointment      # or drone-flyby
docker compose logs -f drone-flyby
scripts/smoke-test.sh                                 # hits /api and /predict
```

GPU host for Drone Flyby:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build drone-flyby
```

Survival Simulator builds from its own directory, not from the root compose file:

```bash
cd survival-v2 && docker build -f deploy/Dockerfile -t surv-v2 .
docker run -d -p 9052:9052 --restart unless-stopped surv-v2
```

> **Known stale reference:** `docker-compose.yml` still has a `survival-simulator` service pointing
> at a `./survival-simulator` directory that no longer exists — the survival work was consolidated
> into `survival-v2/` on 2026-09-20. Building *that one service* from the root compose file fails;
> the other two are unaffected. Use the commands above for survival.

Medical Appointment needs a vLLM server and the two span checkpoints before it will answer; see
`medical-appointment/REPRODUCE.md`. It will fail startup rather than silently downloading or
calling anything hosted.

## Repository map

| Path | What it is |
|---|---|
| `medical-appointment/` | The graded 0.822 build: service, training tools, research notes |
| `drone-flyby/` | Detector service, committed weights, training and diagnosis tools |
| `survival-v2/` | Policy, numba simulator, evaluation harness, deployment |
| `spikes/001-ppo-feasibility/` | The RL feasibility spike that ruled out from-scratch training |
| `scripts/` | `smoke-test.sh`, `check-gpu.sh`, benchmark helpers |
| `docker-compose.yml`, `docker-compose.gpu.yml` | Service definitions; GPU override for Drone Flyby |
| `requirements-dev.txt` | Local experiment environment, not used by the containers |

Older branches are kept rather than deleted: `medical-appointment-0.802` and
`medical-appointment-exact-evidence` hold the previous medical solution,
`BEST-WORKING-VERSION` the frozen best drone configuration.

## Rules and disclosure

- **No hosted LLM is in any `/predict` path.** Every model that answers a grader request runs on
  hardware we own or rent.
- Hosted models *were* used during development — as coding assistants, and scored offline as
  candidate evidence-span producers. Both frontier producers lost to the local 27B and neither was
  deployed. All of it is disclosed, with dates and purposes, in
  [`AI_TOOLS_LOG.md`](AI_TOOLS_LOG.md), including its **Disclosure notes** section.
- Commits after the 20 Sep 16:00 deadline are documentation and provenance, not new solution work.
  What each of them did, and what pins the graded builds, is set out in the same notes section.
- Third-party datasets and pretrained weights are listed in `AI_TOOLS_LOG.md`; external corpora used
  for span pretraining are in `medical-appointment/tools/evidence_training/EXTERNAL_DATA.md`.

## Reading the notes honestly

Most markdown in the challenge folders is a working log written during a four-day competition:
plans, handoffs, and measurements that were later overturned. Where a document contradicts the
"what was graded" file for its challenge, **the graded file wins**. Each one was written after the
evaluation, against the configuration that actually ran.
