#!/usr/bin/env bash
# bootstrap_training_box.sh - one-shot setup for the training instance (Hetzner CCX43, hel1).
#
# Run ON the new instance as root:   bash bootstrap_training_box.sh
# Then (from the Mac) I rsync the code and launch jobs with $(nproc) workers.
#
# Why x86-64 and why docker-free: the grader is x86-64 and this simulator is chaotically sensitive to the
# platform (same seed+policy measured 4,472 ticks on macOS/ARM versus 7,816 on Linux/x86), so training and
# every decision-shaped measurement must happen here. No docker needed for training - plain processes are
# simpler and use all cores; the containers exist only to isolate test sweeps on the 4-core serving box.
set -euo pipefail

echo "=== cores / memory / arch (arch MUST be x86_64) ==="
nproc; free -g | head -2; uname -m

apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip rsync git htop

python3 -m venv /opt/nacv
/opt/nacv/bin/pip install -q --upgrade pip
/opt/nacv/bin/pip install -q numpy pygame shapely scipy
# CPU-only torch is enough: the policies here are tiny MLPs and the simulator is the bottleneck.
/opt/nacv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu

mkdir -p /opt/nac
echo "=== ready. cores=$(nproc) ==="
echo "next (from the Mac): rsync the survival-simulator tree to /opt/nac/, then run:"
echo "  /opt/nacv/bin/python experiments/evolve_meta.py --seeds <seeds> --pop 32 --gen-seeds 8 --workers \$(nproc)"
