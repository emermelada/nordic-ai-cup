# Nordic AI Cup 2026 — team repo

Competition: **17 Sep 10:00 → 20 Sep 16:00 CET**. Registration closes **16 Sep 23:59 CEST**
(https://share.hsforms.com/1wKxsSoSeRSq8NlyPWYGeKAbulxc, all four names on it).

Three challenges, three **independent** services. Each one is its own FastAPI app in its own
Docker container with its own public URL. They never talk to each other.

| Folder | Owner | Host port | Serves from |
|---|---|---|---|
| `medical-appointment/` | A | 8001 | M4 Mac |
| `drone-flyby/` | B | 8002 | RTX 3060 desktop (GPU) |
| `survival-v2/` | C | 8003 | UpCloud VM, public via Caddy on :9052 |

---

## How it works

```
 grader ──HTTPS──▶ public URL ──▶ your machine :800X ──▶ container :8000 ──▶ uvicorn ──▶ api.py
 (Ambolt)          (cloudflared)   (docker-compose        (Dockerfile)                   │
                                    port mapping)                                         ▼
                                                                     dtos.py validates request
                                                                     predict() runs your model
                                                                     dtos.py shapes the response
```

1. **The grader** sends an HTTP request (usually `POST /predict`) with the input data as JSON.
2. **cloudflared** gives your laptop/desktop a public HTTPS address, since home connections
   have no public IP. It forwards traffic to a local port.
3. **docker-compose** maps that host port (8001/8002/8003) to port 8000 inside the container.
4. **The Dockerfile** builds the container: Python 3.14, installs `requirements.txt`, copies the
   service folder in, starts `uvicorn api:app` on port 8000.
5. **`api.py`** receives the request. FastAPI checks it against the request model in
   **`dtos.py`**, calls your `predict()` code, and checks the return value against the response
   model before sending it back.

If the JSON field names or types don't match what the grader expects, the submission scores
zero, often without an error. That's why `dtos.py` must be copied from the official template.

---

## What to touch and what to leave alone

### Inside your service folder

| File | Touch? | What to do |
|---|---|---|
| `api.py` | **Yes — this is your work** | Put your model code in `predict()`. Load the model **once** at the top of the file (module level), never inside `predict()`, or every request reloads it. Change the route path only if the official template uses a different one. Leave `/` and `/api` as they are (health checks). |
| `dtos.py` | **Once, on day 1** | Replace the placeholder classes with the official request/response models from the challenge template. After that, don't edit it. |
| `requirements.txt` | **Yes, when you add a library** | The ML libraries for your challenge are already listed, commented out. Uncomment what you use. Keep `fastapi`, `uvicorn`, `pydantic`. |
| model code (new files) | **Yes** | Add `model.py`, `utils.py`, whatever you need, in the same folder, and import it from `api.py`. |
| weights (`*.pt`, `*.onnx`, …) | **Put them in the folder, never commit them** | Git ignores them. Copy them to the serving machine with `scp`; the Docker build copies them into the image. |
| `Dockerfile` | **Almost never** | Medical Appointment: uncomment the `ffmpeg` line once you decode audio. Otherwise leave it. |
| `.dockerignore` | **No** | |

### Repo root

| File | Touch? | Notes |
|---|---|---|
| `docker-compose.yml` | **No** | Defines the three services, ports and auto-restart. Already done. |
| `docker-compose.gpu.yml` | **No** | Only used on the 3060 machine for Drone Flyby. Ignore it anywhere else. |
| `AI_TOOLS_LOG.md` | **Yes, everyone, as you go** | One line per tool/model/API/dataset used. We may have to disclose this. |
| `requirements-dev.txt` | **Rarely** | Your local notebook/experiment environment. Not used by the containers. |
| `scripts/` | **Run them, don't edit** | See below. |
| `README.md` | Fill the benchmark table | |
| `.gitignore` | **No** | |

### Things you don't need to do anything with

- **`.venv/`** — your local Python environment, never committed.
- **Healthchecks and restart policy** — already in compose; a crashed container restarts itself.
- **GPU setup** — only B / whoever runs the 3060 machine. Everyone else can ignore
  `docker-compose.gpu.yml` and `scripts/check-gpu.sh`.
- **The other two services** — you can build and run only yours.
- **Ports inside the container** — always 8000; the grader never sees it.

---

## Day 1: turning the placeholder into a real endpoint

1. Read the challenge template (official repo, or `emily open <use-case>`).
2. Copy its request/response models into your `dtos.py`. Check the route path (`/predict` or
   something else) and update `api.py` if it differs.
3. Make `predict()` return a **valid** response, even a constant or random one.
4. `docker compose up -d --build <service>` and `scripts/smoke-test.sh`.
5. Tunnel it, queue a validation attempt, see a score. Commit.

**Emily or this skeleton?** Either. Emily generates the same kind of FastAPI project. If you
use Emily's generated folder, copy its `dtos.py` and endpoint into your service here, or replace
your service folder with it and make sure the container still listens on port 8000. If Emily
fights you for more than 30 minutes, use this skeleton.

---

## Setup (once per machine)

Team standard is **Python 3.14**. Everything we need installs on it (torch 2.14, faster-whisper,
pyannote.audio, ultralytics, gymnasium, stable-baselines3). CUDA torch for 3.14 exists only on the
`cu126` and `cu130` indexes.

```bash
git clone git@github.com:emermelada/nordic-ai-cup.git && cd nordic-ai-cup
python3.14 -m venv .venv && source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
```
Mac (MPS): drop the `--extra-index-url`. NVIDIA: use `https://download.pytorch.org/whl/cu126`.
`requirements-dev.txt` has every role's packages; install only your section if you prefer.

Also needed: Docker, `cloudflared`, Emily v3.1.0 (released 11 Sep 2026 — reinstall if older),
Hugging Face token (`hf auth login`, accept pyannote model terms), Kaggle account (phone-verified).

---

## Everyday commands

```bash
docker compose up -d --build drone-flyby           # build + run one service
docker compose logs -f drone-flyby                 # watch its logs
docker compose down                                # stop everything
scripts/smoke-test.sh                              # hit /api and /predict on 8001-8003
CONCURRENT=1 scripts/smoke-test.sh                 # all at once (two services on one machine)
PAYLOAD='{"...": ...}' scripts/smoke-test.sh       # test with a real request body
```

**Public URL:**
```bash
cloudflared tunnel --url http://localhost:8002     # prints https://<random>.trycloudflare.com
scripts/smoke-test.sh https://<random>.trycloudflare.com
```
The URL changes every time cloudflared restarts. Keep it running (tmux) while an attempt is
queued, and re-check the URL before submitting.

**Redeploy on the serving machine:**
```bash
git pull
docker compose up -d --build <service>
# 3060 machine: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build drone-flyby
```

**GPU machine, once:** `scripts/check-gpu.sh` checks driver, container toolkit, GPU inside a
container, `torch.cuda`, and that sleep is disabled.

---

## Readiness test (everyone, before the 17th, < 10 min)

1. `docker compose up -d --build <your service>` (or `emily open` a 2025 use case)
2. `curl localhost:<port>/api` responds
3. `cloudflared tunnel --url http://localhost:<port>`
4. Open `https://…trycloudflare.com/api` on your phone → JSON back

## Benchmarks

```bash
python scripts/bench_yolo.py [video.mp4]           # synthetic frames if no video
python scripts/bench_whisper.py [audio] [size]     # downloads a JFK sample if no audio
```

| Machine | YOLOv8n FPS | Whisper small, × realtime |
|---|---|---|
| ThinkPad T480s (CPU) | | |
| MacBook Pro M4 | | |
| RTX 3060 desktop | | |

---

## Working rules

- Commit your best working version **before** changing it.
- Stuck 2+ hours on the same thing → say so.
- Log AI tools in `AI_TOOLS_LOG.md` as you go.
- Day 4: nothing new after mid-morning.

## How 2025 submissions worked (expect similar, confirm on Discord)

- Submit host URL + team API key on a form.
- **Unlimited validation attempts**, **one final evaluation per use case** on a different dataset.
- The grader calls your API when you queue an attempt, so the service must be up then.
- Top teams handed over training code and models afterwards.
- Compute was provided (UCloud) in 2025.

## Questions for Discord

1. Continuous grading or on-demand attempts? One final evaluation per challenge again?
2. Inference time limits per challenge (Drone Flyby is real-time).
3. Emily licensing for participants.
4. Can the three endpoints be on different hosts? Are trycloudflare URLs accepted?
5. Any provided compute this year?
