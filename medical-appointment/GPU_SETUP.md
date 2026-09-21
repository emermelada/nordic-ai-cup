# Serving on a rented GPU (Vast.ai): A100 80 GB or RTX PRO 6000 96 GB

The pipeline runs unchanged on CUDA through `pipeline/vllm_backend.py`: faster-whisper
transcribes on the GPU and every LLM pass goes to a local vLLM server. Select it with
`MEDICAL_BACKEND=vllm`. Nothing in `/predict` leaves the machine, which is what the rules
require; the rented VM is your own machine.

## 1. Rent the instance

1. vast.ai → Account → add credit (about $25 covers a day) and add your SSH public key
   (`cat ~/.ssh/id_ed25519.pub` on the Mac; create one with `ssh-keygen -t ed25519` if missing).
2. Search → filters: GPU **A100 SXM4 80GB** (PCIe 80GB is fine), **On-demand** (not
   interruptible), disk **≥ 150 GB**, internet down **≥ 500 Mbps**, CUDA **≥ 12.4**,
   verified hosts, reliability ≥ 98 %, max duration ≥ 2 days. Sort by price; expect $0.8–1.2/h.
3. Template: **PyTorch (CUDA 12.x)** with launch mode "SSH". Rent.
4. Instance card → copy the SSH command (`ssh -p <PORT> root@<HOST>`). Everything below runs there.

## 2. Install

```bash
apt-get update && apt-get install -y ffmpeg git rsync tmux curl
python3 -m venv /workspace/venv && . /workspace/venv/bin/activate && pip install -U pip
pip install vllm hf_transfer "huggingface_hub[cli]" faster-whisper \
    fastapi uvicorn "pydantic>=2.7,<3" requests rank-bm25 rapidfuzz "json-repair<1" av numpy
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
```

## 3. Copy the project from the Mac

On the Mac (includes `data/` so the local evaluator works on the box):

```bash
rsync -av -e "ssh -p <PORT>" \
  --exclude .venv --exclude .venv311 --exclude runs --exclude transcripts \
  --exclude models --exclude __pycache__ --exclude .idea \
  /Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/ root@<HOST>:/workspace/medical-appointment/
```

## 4. Download the models

Sizes checked on 2026-09-18. `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` is 79 GB on disk and does
**not** fit an 80 GB card with KV cache and Whisper; do not pick it for a single A100.

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.6-27B   --local-dir /workspace/models/qwen36-27b     # 56 GB bf16, evidence + answers
hf download Qwen/Qwen3.5-9B    --local-dir /workspace/models/qwen35-9b      # 19 GB bf16, optional second server for drafts
```

(`huggingface-cli download ...` on older CLIs.) A few minutes each on a datacenter link.

Alternatives, one line each: `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` (dense 72B, 42 GB, two
generations older) if you insist on a 70B; `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (20 GB) is the
unpruned parent of the REAP model serving on the Mac.

## 5. Start vLLM (tmux window 1)

Single server, 27B in bf16 for both passes:

```bash
tmux new -s vllm
. /workspace/venv/bin/activate
vllm serve /workspace/models/qwen36-27b --served-model-name qwen --port 8000 \
  --max-model-len 12288 --gpu-memory-utilization 0.80 --dtype bfloat16
```

Wait for `Application startup complete`, then check: `curl -s localhost:8000/v1/models`.

Two servers (best measured combination: a small model writes the drafts, the big one refines
them; +0.02 tIoU on the Mac). Use the AWQ 27B so both fit:

```bash
hf download cyankiwi/Qwen3.6-27B-AWQ-INT4 --local-dir /workspace/models/qwen36-27b-awq   # 15 GB
vllm serve /workspace/models/qwen36-27b-awq --served-model-name qwen --port 8000 \
  --max-model-len 12288 --gpu-memory-utilization 0.45
# second tmux window:
vllm serve /workspace/models/qwen35-9b --served-model-name qwen9 --port 8001 \
  --max-model-len 12288 --gpu-memory-utilization 0.35
```

## 6. Start the API (tmux window 2)

```bash
tmux new -s api
. /workspace/venv/bin/activate && cd /workspace/medical-appointment
MEDICAL_BACKEND=vllm VLLM_URL=http://127.0.0.1:8000/v1 \
  uvicorn api:app --host 127.0.0.1 --port 9054 --workers 1
```

Add `VLLM_ANSWER_URL=http://127.0.0.1:8001/v1` for the two-server layout. Wait for
`Inference worker ... loaded and warmed` (Whisper downloads on first start).

## 7. Check locally, then expose

```bash
cd /workspace/medical-appointment && . /workspace/venv/bin/activate
python local_evaluator.py --url http://127.0.0.1:9054/predict          # all 39, read worst round trip
```

Expose (tmux window 3) and submit the printed `https://<name>.trycloudflare.com/predict`:

```bash
tmux new -s tunnel
cloudflared tunnel --url http://127.0.0.1:9054
```

Run a platform validation before anything else. Keep the tmux sessions alive; keep the
instance running until the final attempt (stopping it keeps the disk but changes nothing
else, and the tunnel hostname changes on every cloudflared restart).

## RTX PRO 6000 Blackwell 96 GB (Workstation or Max-Q)

Same steps; the extra 16 GB and Blackwell change three things.

- **Host requirements.** Blackwell needs a CUDA 12.8+ driver: filter Vast listings by
  CUDA ≥ 12.8 and pick a PyTorch template built for cu128. `pip install vllm` ships cu128
  wheels; if `vllm serve` complains about the compute capability, upgrade with
  `pip install -U vllm torch`. Workstation and Max-Q have the same memory; the Max-Q is
  power-limited and roughly 15-25 % slower, which does not matter at these request sizes.
  If faster-whisper (CTranslate2) fails to load on Blackwell, install `whisperx` or
  `openai-whisper` instead; the transcript shape in `pipeline/vllm_backend.py` is the only
  place to adapt.
- **Layout 1, no quantisation anywhere (start here):** Qwen3.6-27B bf16 (56 GB) for the
  evidence pass and Qwen3.5-9B bf16 (19 GB) for the answers, both in full precision, plus
  Whisper. This is the combination that measured best on the Mac, now without 4-bit loss.

  ```bash
  vllm serve /workspace/models/qwen36-27b --served-model-name qwen --port 8000 \
    --max-model-len 12288 --gpu-memory-utilization 0.64 --dtype bfloat16
  vllm serve /workspace/models/qwen35-9b  --served-model-name qwen9 --port 8001 \
    --max-model-len 12288 --gpu-memory-utilization 0.24 --dtype bfloat16
  MEDICAL_BACKEND=vllm VLLM_URL=http://127.0.0.1:8000/v1 VLLM_ANSWER_URL=http://127.0.0.1:8001/v1 \
    uvicorn api:app --host 127.0.0.1 --port 9054 --workers 1
  ```

- **Layout 2, the biggest model that fits:** `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (79 GB,
  10 B active so it decodes fast) for both passes, alone.

  ```bash
  hf download Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --local-dir /workspace/models/qwen35-122b
  vllm serve /workspace/models/qwen35-122b --served-model-name qwen --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.90
  ```

  Whisper takes the remaining 6-8 GB. Compare both layouts with `local_evaluator.py` on the
  box (under ten minutes each) and validate the better one on the platform; the 122B has
  not been measured on this task, the 27B family has.

## Cheap test first: score the model on cached transcripts (no Whisper, three minutes)

`tools/gpu_eval/` holds the 39 training conversations' cached whisper words and the Mac 9B's
answers and spans (4 MB). `eval.py` scores any OpenAI-compatible endpoint on them in three
rows: the model's own answers and quotes, the model refining the 9B's drafts (the best
measured layout), and the model refining its own answers (single-model serving). Copy only
what it needs, then run it wherever the model is served:

```bash
# on the Mac: copy the code and the bundle (no audio, no venvs)
rsync -av -e "ssh -p <PORT>" --exclude .venv --exclude .venv311 --exclude runs \
  --exclude transcripts --exclude models --exclude __pycache__ --exclude data \
  /Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/ root@<HOST>:/workspace/medical-appointment/
# on the box, after vLLM is up (torch already there; these are pure Python)
pip install fastapi "pydantic>=2.7,<3" requests rank-bm25 rapidfuzz "json-repair<1" numpy av
cd /workspace/medical-appointment
MEDICAL_BACKEND=vllm VLLM_URL=http://127.0.0.1:8000/v1 VLLM_API_KEY=<token or empty> \
  EVAL_TAG=qwen38 python tools/gpu_eval/eval.py all
```

It prints the three rows with the Mac reference numbers (9B stage B raw 0.7739, REAP-only
0.7877 on the same inputs) and a bootstrap interval for the evidence gain; generations are
cached under `tools/gpu_eval/results/`, so a rerun is free. About 80 requests, three minutes
on a PRO 6000, roughly $0.10 of rental. Stop the instance if the numbers disappoint.

The same command works from the Mac against a hosted development endpoint that serves the
model (allowed during development, never in `/predict`): set `VLLM_URL` to its base URL,
`VLLM_API_KEY` to its key and `MEDICAL_VLLM_MODEL` to the model name it lists, and leave
`MEDICAL_BACKEND=vllm`. Around 120k tokens in total, cents.

## One-click Qwen3.8-27B on Vast's model library (RTX PRO 6000)

Vast's model page (`vast.ai/model/qwen38-27b`, "Deploy Now", vllm) starts a container that
downloads Qwen3.8-27B and serves it with vLLM; Whisper and the API then run inside that
same container. Steps:

1. Account → add credit and your SSH key first, then click **Deploy Now** with `vllm`
   selected (on-demand, dedicated 1× RTX PRO 6000).
2. Console → Instances → wait until the instance log shows vLLM's
   `Application startup complete` (the 56 GB download takes a few minutes). Copy the SSH
   command from the instance card and log in.
3. Find the server and its key inside the container: `env | grep -iE 'vllm|token|api_key|port'`
   and `curl -s localhost:8000/v1/models`. If the reply is an authentication error, the
   token is in one of those variables; export it as `VLLM_API_KEY` for the API below.
   If vLLM listens on another port, use that in `VLLM_URL`.
4. Install the rest inside the container (torch is already there):

   ```bash
   apt-get update && apt-get install -y ffmpeg rsync tmux curl
   pip install faster-whisper fastapi uvicorn "pydantic>=2.7,<3" requests rank-bm25 rapidfuzz "json-repair<1" av numpy
   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
       -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
   ```

5. `rsync` the project from the Mac (section 3), then in tmux:

   ```bash
   cd /workspace/medical-appointment
   MEDICAL_BACKEND=vllm VLLM_URL=http://127.0.0.1:8000/v1 VLLM_API_KEY=<token or empty> \
     uvicorn api:app --host 127.0.0.1 --port 9054 --workers 1
   ```

6. `python local_evaluator.py --url http://127.0.0.1:9054/predict`, then `cloudflared tunnel
   --url http://127.0.0.1:9054` in another tmux window, submit the URL, validate.

The template's vLLM usually claims about 90 % of the card, so a second server for a 9B
draft model does not fit beside it as launched. To try the two-model layout later, stop the
template's vLLM (its process is visible in `ps`; or edit the template's vLLM arguments in
the console to `--gpu-memory-utilization 0.62`) and start the two servers from Layout 1
above with `vllm serve`, which is installed in the image.

## The annotators' coordinate system: `MEDICAL_ASR_MODE=base`

The team's other branch (`medical-appointment-0.802`) found that the gold spans are
faster-whisper **`base`, int8, CPU** word timestamps (`language='en'`,
`word_timestamps=True`, no VAD, decoded from a file). `pipeline/base_asr.py` transcribes
exactly that way and flags the transcript `exact_timestamps`, which switches off the
turbo-era onset shift and reply extension and makes stage B split sentences on end
punctuation only, the unit the annotators quoted. On the Vast Threadripper the boundaries
are not bit-exact (232/390; 79 % within 20 ms) but the word-span oracle is 0.986 tIoU
against 0.926 for turbo words. `MEDICAL_EVIDENCE_PROMPT=annot` selects the evidence
prompt built on their measured convention with their real quotes as examples (the seven
source conversations are reported separately by `tools/gpu_eval/eval.py`).

```bash
MEDICAL_BACKEND=vllm VLLM_URL=http://127.0.0.1:18000/v1 MEDICAL_ASR_MODE=base \
  MEDICAL_EVIDENCE_PROMPT=annot ASR_CPU_THREADS=16 HF_HOME=/workspace/.hf_home \
  python3 -m uvicorn api:app --host 127.0.0.1 --port 9054 --workers 1
```

The base model runs on the CPU (about 4 s per conversation at 16 threads) and needs no
CUDA 12 libraries. `EVAL_INPUTS=inputs_base.json` scores the harness in these coordinates.

## Gotchas met on the Vast vLLM image (2026-09-18)

- **Switching the template's model.** Append `VLLM_MODEL`, `MODEL_NAME` and `VLLM_ARGS` to
  `/etc/environment`, then `supervisorctl stop vllm model-ui`, kill the *old* `vllm serve`
  and `EngineCore` processes yourself (the supervisor restart leaves them running and the
  new server dies with "Address already in use" on 18000), wait until `nvidia-smi` shows
  the memory freed, then `supervisorctl start vllm model-ui`. Use `pgrep -f "vllm [s]erve"`
  patterns so `pkill` does not kill your own shell.
- **The template exports `VLLM_MODEL`.** Our override is therefore named
  `MEDICAL_VLLM_MODEL`; leave it unset on the box so the served name is discovered.
- **CTranslate2 (faster-whisper) needs CUDA 12 libraries** while the image ships CUDA 13 for
  torch. Install them apart from the system packages and expose them only to the API:
  `pip install --target /workspace/cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12`, then run the
  API with `LD_LIBRARY_PATH=/workspace/cu12/nvidia/cublas/lib:/workspace/cu12/nvidia/cudnn/lib`
  and `HF_HOME=/workspace/.hf_home`. Whisper turbo then transcribes a 2-minute file in ~3 s.
- vLLM listens on **127.0.0.1:18000** without a key; port 8000 is the Caddy edge with the
  instance token. Use 18000 from inside the box.

## Knobs that matter on the GPU

- `EVIDENCE_BUDGET_SECONDS` in `pipeline/mlx_backend.py` (32 s) can stay; a 27B on an A100
  answers in about 7 s and refines in about 3 s.
- `EVIDENCE_MAX_TOKENS['refine']` 320 is enough; raise `--max-model-len` only for longer audio.
- Whisper: `WHISPER_CUDA_MODEL=large-v3-turbo` (default) matches the Mac's timing quality;
  `large-v3` was worse on timestamps locally.
