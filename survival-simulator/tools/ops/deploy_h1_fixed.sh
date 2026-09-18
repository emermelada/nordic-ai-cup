#!/bin/bash
# CORRECTED H1 DEPLOY.
# The first attempt failed the build gate because best_controller.sha256 was NOT uploaded (the VPS
# ledger was stale) and, worse, `docker build ... | tail -3` masked the failure so the container was
# swapped onto the OLD image anyway. This version uploads the ledger, runs the build WITHOUT a pipe
# so its real exit code is visible, and only swaps the container if the build actually succeeded.
KEY=/Users/zaitzev/.ssh/vps_hermes
HEL=212.147.239.222
SRC=/Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator
SSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=15 root@$HEL"

echo "=== 1. upload controller + LEDGER + params ==="
scp -i "$KEY" -o BatchMode=yes "$SRC/best_controller.py" \
    "$SRC/best_controller.sha256" "$SRC/best_controller/params.json" root@$HEL:/opt/nac-survival/ || exit 1

echo "=== 2. does the ledger match the uploaded file on the VPS? ==="
$SSH 'cd /opt/nac-survival && cat best_controller.sha256 && sha256sum -c best_controller.sha256'

echo "=== 3. build (no pipe: the real exit code must survive) ==="
if $SSH 'cd /opt/nac-survival && docker build -f Dockerfile.vps -t nac-survival-vps . >/tmp/build.log 2>&1'; then
  echo "BUILD OK"
  $SSH 'tail -3 /tmp/build.log'
else
  echo "BUILD FAILED -- nothing swapped. Last 15 lines:"
  $SSH 'tail -15 /tmp/build.log'
  exit 1
fi

echo "=== 4. swap the container (only reached if the build succeeded) ==="
$SSH 'docker rm -f nac-survival-vps >/dev/null 2>&1; docker run -d --name nac-survival-vps -p 9052:9052 --restart unless-stopped nac-survival-vps >/dev/null && echo container_started'
sleep 10

echo "=== 5. verify WHAT IS ACTUALLY SERVING (must be H1: fruit_weight 1.148) ==="
$SSH 'docker exec nac-survival-vps cat /app/best_controller/params.json | head -4'
$SSH 'curl -s -o /dev/null -w "origin GET / -> %{http_code}\n" localhost:9052/'
curl -s -o /dev/null -w "PUBLIC GET / -> %{http_code} in %{time_total}s\n" --max-time 25 https://survival.zaitzev.com/
echo "--- a real action from the public endpoint ---"
curl -s --max-time 25 -X POST https://survival.zaitzev.com/predict -H 'Content-Type: application/json' \
  -d '{"agents":[{"agent_id":1,"energy":300,"max_energy":500,"age":10,"biome":"forest","hearing_radius":100,"vision_radius":400,"vision_angle":1.57,"speed":10,"sprint_speed":20,"observations":[{"type":"Fruit","distance":100,"angle":0.2}]}]}' | head -c 300; echo