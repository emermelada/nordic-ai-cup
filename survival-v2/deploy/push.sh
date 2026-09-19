#!/usr/bin/env bash
# Copy the controller to the VPS and (re)start the service.
#
#   bash deploy/push.sh <ip> [first]     # "first" also installs Python deps and the systemd unit
#
# Uses the key ~/.ssh/surv_vps. Line endings are normalised to LF on the way (the Windows checkout is CRLF).
set -euo pipefail
IP=$1
MODE=${2:-update}
KEY=~/.ssh/surv_vps
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new root@$IP"
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/deploy"
for f in hive.py server.py; do tr -d '\r' < "$HERE/$f" > "$TMP/$f"; done
for f in setup_vm.sh requirements-server.txt; do tr -d '\r' < "$HERE/deploy/$f" > "$TMP/deploy/$f"; done
$SSH "mkdir -p /opt/surv/deploy"
scp -i $KEY -q "$TMP/hive.py" "$TMP/server.py" root@$IP:/opt/surv/
scp -i $KEY -q "$TMP/deploy/setup_vm.sh" "$TMP/deploy/requirements-server.txt" root@$IP:/opt/surv/deploy/
rm -rf "$TMP"
if [ "$MODE" = "first" ]; then
  $SSH "bash /opt/surv/deploy/setup_vm.sh"
else
  $SSH "systemctl restart surv && sleep 2 && curl -s http://127.0.0.1:9052/api && echo"
fi
echo "endpoint: http://$IP:9052/predict"
