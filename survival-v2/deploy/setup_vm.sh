#!/usr/bin/env bash
# Set up and start the survival agent server on a fresh Ubuntu/Debian VM (run as root).
#
#   scp -r survival-v2/{hive.py,server.py,deploy} root@<ip>:/opt/surv/
#   ssh root@<ip> 'bash /opt/surv/deploy/setup_vm.sh'
#
# Serves http://<ip>:9052/predict (plain HTTP, no tunnel: the grader runs in Hetzner Helsinki and allows only
# 600 s of accumulated wait per game). Restarts on failure and on reboot (systemd unit "surv").
set -euo pipefail
APP=/opt/surv
PORT=${PORT:-9052}

apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip >/dev/null

python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q -r "$APP/deploy/requirements-server.txt"

cat > /etc/systemd/system/surv.service <<UNIT
[Unit]
Description=Survival simulator agent server
After=network-online.target

[Service]
WorkingDirectory=$APP
Environment=SURV_LOG=$APP/server_log.jsonl
EnvironmentFile=-$APP/deploy/hive.env
ExecStart=$APP/.venv/bin/uvicorn server:app --host 0.0.0.0 --port $PORT --loop uvloop --http httptools --no-access-log --log-level warning
Restart=always
RestartSec=1
Nice=-5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now surv
sleep 2
curl -s "http://127.0.0.1:$PORT/api" && echo
if command -v ufw >/dev/null && ufw status | grep -q active; then ufw allow "$PORT"/tcp; fi
echo "Register: http://$(curl -s ifconfig.me):$PORT/predict"
