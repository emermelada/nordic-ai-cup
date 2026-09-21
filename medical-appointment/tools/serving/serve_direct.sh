#!/bin/bash
# Serve on Vast's mapped port 3000 (external 40724): no tunnel, no Cloudflare.
cd /workspace/medical-appointment
ENV="MEDICAL_ASR_MODE=base MEDICAL_EVIDENCE_MODE=perq MEDICAL_EVIDENCE_PROMPT=v3 MEDICAL_ANSWER_PROMPT=named MEDICAL_RESCUE_PHONETIC=1 MEDICAL_RESCUE_READING=turbo MEDICAL_RESCUE_THRESHOLD=$1 ASR_CPU_THREADS=16"
tmux kill-session -t api 2>/dev/null || true
for p in $(pgrep -f "[u]vicorn api:app"); do kill $p 2>/dev/null || true; done
sleep 4
: > /workspace/api.log
tmux new -d -s api "cd /workspace/medical-appointment && export HF_HOME=/workspace/.hf_home MEDICAL_BACKEND=vllm VLLM_URL=http://127.0.0.1:18000/v1 LD_LIBRARY_PATH=/workspace/cu12/nvidia/cublas/lib:/workspace/cu12/nvidia/cudnn/lib $ENV && python3 -m uvicorn api:app --host 0.0.0.0 --port 3000 --workers 1 >> /workspace/api.log 2>&1"
for i in $(seq 1 90); do grep -q "Application startup complete\|startup failed\|Error" /workspace/api.log 2>/dev/null && break; sleep 5; done
grep -E "Inference worker|startup complete|Error" /workspace/api.log | tail -2
ss -ltn | grep ":3000" | head -1
curl -s -m 8 -o /dev/null -w "local  /api %{http_code}\n" http://127.0.0.1:3000/api
echo "PUBLIC: http://$(curl -s -m 8 ifconfig.me):40724/predict"
