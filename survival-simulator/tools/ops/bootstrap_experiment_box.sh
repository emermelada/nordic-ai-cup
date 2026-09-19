#!/usr/bin/env bash
# bootstrap_experiment_box.sh - one-shot setup for the 64-core experiment box (212.147.236.122).
#
# Two lessons are baked in here:
#   * plain `pygame` has no wheel for Python 3.14 and FAILS TO BUILD (no SDL dev headers). Use
#     `pygame-ce`, which provides the same `pygame` module name. The simulator imports pygame at module
#     level in src/elements/*.py, so this is load-bearing, not optional.
#   * the experiment box must NOT host serving. Serving stays on its own machine so that a 64-worker
#     sweep cannot add latency to the grader's requests (score ≈ min(ticks, 600s / latency)).
#
# Run ON the new box as root:  bash bootstrap_experiment_box.sh
set -euo pipefail

echo "=== spec check (arch MUST be x86_64: the grader is x86 and macOS/ARM diverges 4,472 vs 7,816) ==="
uname -m; nproc; free -g | head -2; df -h / | tail -1

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip rsync tmux git htop

python3 -m venv /opt/nacv
/opt/nacv/bin/pip install -q --upgrade pip
/opt/nacv/bin/pip install -q numpy scipy shapely pygame-ce gymnasium pydantic fastapi uvicorn requests
# CPU-only torch: policies here are tiny MLPs and the simulator is the bottleneck. A GPU would idle.
/opt/nacv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu

mkdir -p /opt/nac
/opt/nacv/bin/python - <<'PY'
import numpy, scipy, shapely, pygame, gymnasium, pydantic, fastapi, torch
print(f"imports OK: numpy {numpy.__version__} scipy {scipy.__version__} pygame {pygame.__version__} "
      f"torch {torch.__version__}")
PY
echo "=== ready. cores=$(nproc) ==="
echo "next (from the Mac): tar the survival-simulator tree to /opt/nac/, then launch lanes with $(nproc)-ish workers"
