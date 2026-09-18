#!/usr/bin/env bash
# Set up and run the medical-appointment endpoint on a rented Linux GPU box.
# Written for vast.ai's vLLM template on an RTX 5090, with the template's
# environment set to
#     VLLM_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
#     VLLM_ARGS=--max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.90
# and port 9054 exposed. Run it from medical-appointment/ on this branch:
#
#   bash gpu_setup.sh          # install, timestamp gate, LLM server, offline score
#   bash gpu_setup.sh serve    # the endpoint on :9054 (run it inside tmux)
#
# It uses whatever LLM server is already answering: the template's on :8000, or
# one on LLM_BASE_URL. On a plain CUDA image with neither, it installs vLLM and
# starts its own. Idempotent, so Sunday on a fresh box is the same two commands.
set -euo pipefail
cd "$(dirname "$0")"

LLM_HF_MODEL="${LLM_HF_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
LLM_PORT="${LLM_PORT:-8080}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:${LLM_PORT}/v1}"
export LLM_MODEL="${LLM_MODEL:-local}"
TEMPLATE_URL="http://127.0.0.1:8000/v1"  # where vast's vLLM template serves
# Outside the repo, or under the git-ignored data/, so nothing here gets committed.
APP_VENV="${APP_VENV:-$HOME/venvs/medical}"
LLM_VENV="${LLM_VENV:-$HOME/venvs/vllm}"
LOGS="$PWD/data/logs"
mkdir -p "$LOGS"

step() { printf '\n=== %s\n' "$*"; }

make_venv() {  # make_venv DIR PACKAGES...
  local dir="$1"; shift
  if [ ! -x "$dir/bin/python" ]; then
    mkdir -p "$(dirname "$dir")"
    python3 -m venv "$dir" || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv "$dir"; }
    "$dir/bin/pip" install -q --upgrade pip
  fi
  "$dir/bin/pip" install -q "$@"
}

fetch_data() {
  if [ -f data/question_train.csv ]; then return; fi
  rm -rf /tmp/official
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/amboltio/Nordic-AI-Cup-2026 /tmp/official
  git -C /tmp/official sparse-checkout set medical-appointment/data
  mkdir -p data && cp -r /tmp/official/medical-appointment/data/. data/
}

served_model() {  # served_model URL: print the model id a server answers with, or fail
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys, urllib.request
print(json.load(urllib.request.urlopen(sys.argv[1] + '/models', timeout=3))['data'][0]['id'])
PY
}

find_llm() {  # point LLM_BASE_URL / LLM_MODEL at a server that is already answering
  local url id
  for url in "$LLM_BASE_URL" "$TEMPLATE_URL"; do
    id="$(served_model "$url")" || continue
    export LLM_BASE_URL="$url" LLM_MODEL="$id"
    echo "LLM server: $url (model $id)"
    return 0
  done
  return 1
}

start_own_llm() {
  make_venv "$LLM_VENV" vllm
  nohup "$LLM_VENV/bin/vllm" serve "$LLM_HF_MODEL" \
      --served-model-name "$LLM_MODEL" --port "$LLM_PORT" \
      --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.90 \
      > "$LOGS/vllm.log" 2>&1 &
  local pid=$!
  echo "vLLM starting (the first start downloads ~23 GB): tail -f $LOGS/vllm.log"
  until find_llm; do
    sleep 10
    kill -0 "$pid" 2>/dev/null || { echo "vLLM exited; last lines of $LOGS/vllm.log:"; tail -20 "$LOGS/vllm.log"; exit 1; }
  done
}

ensure_llm() {
  find_llm && return
  if [ -n "${VLLM_MODEL:-}" ]; then
    # The template launches its own server from VLLM_MODEL / VLLM_ARGS. Wait for
    # it rather than start a second one that would fight it for the GPU.
    echo "Waiting for the template's vLLM ($VLLM_MODEL) on :8000; its first start downloads the model."
    local waited=0
    until find_llm; do
      sleep 15
      waited=$((waited + 15))
      if (( waited % 120 == 0 )); then echo "  still waiting (${waited}s); check the template's vLLM log if this drags on"; fi
    done
    return
  fi
  echo "No LLM server running: installing vLLM into $LLM_VENV and starting it."
  start_own_llm
}

case "${1:-setup}" in
  setup)
    step "Endpoint environment ($APP_VENV)"
    make_venv "$APP_VENV" -r requirements.txt
    step "Training data, for the checks"
    fetch_data
    step "Timestamp gate: expect 390/390 (PLAN.md step 1 if not)"
    "$APP_VENV/bin/python" check_timestamps.py || echo "!!! Not exact on this CPU. Still usable (about -1% tIoU); note it."
    step "LLM server"
    ensure_llm
    step "Offline score on the training conversations (no attempt used)"
    "$APP_VENV/bin/python" eval_offline.py cache
    "$APP_VENV/bin/python" eval_offline.py score llm:answer | tee "$LOGS/offline_score.txt"
    ;;
  serve)
    ensure_llm
    step "Endpoint on :9054. Submit http://<public ip>:<port mapped to 9054>/predict"
    ANSWERER=llm:answer exec "$APP_VENV/bin/python" api.py
    ;;
  *)
    echo "usage: bash gpu_setup.sh [setup|serve]"; exit 2 ;;
esac
