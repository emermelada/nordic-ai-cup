#!/bin/bash
# Keep the endpoint up: restart uvicorn if it ever exits, logging each start.
cd "$(dirname "$0")/.."
while true; do
  LOG="runs/serve-9054/server-$(date +%Y%m%d-%H%M%S).log"
  echo "starting uvicorn -> $LOG"
  .venv311/bin/python -m uvicorn api:app --host 127.0.0.1 --port 9054 --workers 1 >> "$LOG" 2>&1
  echo "uvicorn exited with $? at $(date)" >> "$LOG"
  sleep 3
done
