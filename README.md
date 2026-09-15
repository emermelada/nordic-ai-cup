# Nordic AI Cup 2026 — team repo

Competition: 17 Sep 10:00 → 20 Sep 16:00 CET. **Registration closes 16 Sep 23:59 CEST**
(https://share.hsforms.com/1wKxsSoSeRSq8NlyPWYGeKAbulxc, all four names on it).

```
medical-appointment/   A   host port 8001   serves from: M4 Mac
drone-flyby/           B   host port 8002   serves from: RTX 3060 desktop (GPU override)
survival-simulator/    C   host port 8003   serves from: 3060 desktop or cheap VM
docker-compose.yml         all three services, CPU
docker-compose.gpu.yml     GPU override for drone-flyby (cu126 torch)
AI_TOOLS_LOG.md            disclosure log — update as you go
scripts/                   smoke test, GPU check, YOLO/Whisper benchmarks
```

Each service is a standalone FastAPI app: `api.py` (endpoints), `dtos.py` (schema),
`requirements.txt`, `Dockerfile` (`python:3.14-slim`). Containers listen on 8000 internally.
The `/predict` endpoints are **placeholders**: on day 1 copy the official DTOs and route from the
use-case template, the grader scores zero on a schema mismatch.

## Python

Team standard is **Python 3.14** (host venv and container base image). Verified 15 Sep: torch 2.14,
torchaudio, ctranslate2/faster-whisper, pyannote.audio, ultralytics, opencv (abi3), gymnasium,
stable-baselines3, numpy/pandas/sklearn all install on 3.14. CUDA torch for 3.14 exists on the
`cu126` and `cu130` indexes only (not cu128).

If an Emily-generated Dockerfile pins an older Python, change its `FROM` to `python:3.14-slim`
or keep it — the container is what gets graded, but be consistent within a service.

```bash
python3.14 -m venv .venv && source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch numpy pandas matplotlib scikit-learn requests
```
Mac: plain `pip install torch` (MPS). NVIDIA: `--index-url https://download.pytorch.org/whl/cu126`.

## Run locally

```bash
docker compose up -d --build                       # all three
docker compose up -d --build drone-flyby           # one
scripts/smoke-test.sh                              # /api + /predict on 8001-8003
CONCURRENT=1 scripts/smoke-test.sh                 # same, all at once (contention)
docker compose logs -f drone-flyby
```

## Public URL (tunnel)

Home machines have no public IP. Cloudflare quick tunnel, no account needed:
```bash
cloudflared tunnel --url http://localhost:8002     # prints https://<random>.trycloudflare.com
scripts/smoke-test.sh https://<random>.trycloudflare.com
```
Quick-tunnel URLs change on every restart. Once we know the submission flow, either keep the
process alive (tmux/systemd) or set up a named tunnel on a domain so the URL is stable.

## GPU host (RTX 3060)

```bash
scripts/check-gpu.sh      # driver, container toolkit, GPU inside container, torch.cuda, sleep masked
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build drone-flyby
docker compose exec drone-flyby python -c "import torch; print(torch.cuda.is_available())"
```
The GPU override is only for this machine; on a host without the NVIDIA toolkit it fails to start.

## Redeploy

```bash
git pull
docker compose up -d --build <service>             # add -f docker-compose.gpu.yml on the GPU host
```
Weights are git-ignored (`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`): `scp` them into the service
folder on the serving machine; they are copied into the image on build.

## Emily

Latest is **v3.1.0 (released 11 Sep 2026)** — reinstall if you have an older copy.
```bash
gh release download Release-v3.1.0 -R amboltio/emily-cli -p linux.zip   # or macos.zip / emily.pkg
unzip linux.zip && ./emily        # interactive: accept terms (ambolt.io/terms-of-use), install path ~/.emily
emily doctor
cd ~/Proyectos/DM-i-AI-2025 && emily open race-car
```
30-minute rule: if Emily fights you, drop it and use the service skeleton here.

## Readiness test (everyone, < 10 min)

1. `emily open` a 2025 use case (or `docker compose up -d --build <service>` here)
2. `curl localhost:<port>/api` responds
3. `cloudflared tunnel --url http://localhost:<port>`
4. Open `https://…trycloudflare.com/api` on your phone → JSON back

## Benchmarks to run this week

```bash
python scripts/bench_yolo.py [video.mp4]           # ThinkPad, M4, 3060
python scripts/bench_whisper.py [audio] [size]     # M4 vs CPU-only
```

| Machine | YOLOv8n FPS | Whisper small, x realtime |
|---|---|---|
| ThinkPad T480s (CPU) | | |
| MacBook Pro M4 | | |
| RTX 3060 desktop | | |

## How 2025 submissions worked (expect similar, confirm on Discord)

- Submit a host URL + team API key on a submission form.
- **Unlimited validation attempts** (validation set, shown on the scoreboard), but **one final
  evaluation attempt per use case** on a different dataset — don't overfit to validation.
- The grader calls the API when you queue an attempt → the service must be up *then*, not
  necessarily 24/7. Confirm for 2026.
- Top teams had to hand over training code and models afterwards.
- Compute was provided (UCloud) in 2025 — ask whether 2026 has any.

## Questions for Discord

1. Continuous grading or on-demand attempts? One final evaluation per challenge again?
2. Inference time limits per challenge (Drone Flyby is real-time).
3. Emily licensing for participants.
4. Can the three endpoints be on different hosts/URLs? Are tunnel URLs (trycloudflare) accepted?
5. Any provided compute this year (UCloud or similar)?
