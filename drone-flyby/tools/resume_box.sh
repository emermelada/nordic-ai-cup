#!/usr/bin/env bash
# Bring the stopped box back and serve the ship config. One command.
#
#   tools/resume_box.sh -p <ssh-port> root@<ip>
#
# The box keeps its disk across a STOP (not a destroy), so this resumes the
# partial upload rather than re-sending 103 MB over a 170 kB/s uplink.
set -u
PORT=''; while getopts 'p:' o; do case $o in p) PORT=$OPTARG;; esac; done
shift $((OPTIND-1)); TARGET=${1:?usage: tools/resume_box.sh -p <port> root@<ip>}
cd "$(dirname "$0")/.."
SSH="ssh -p $PORT -o StrictHostKeyChecking=no"

echo "== resuming upload (only what is missing)"
rsync -a --info=progress2 --partial \
  --exclude 'data/' --exclude 'src/' --exclude '__pycache__' --exclude '.git/' \
  --exclude '.venv/' --exclude 'models/drone-yolo11m-v5.pt' \
  --exclude 'models/drone-yolo11s-v7.pt' --exclude 'models/drone-yolo11n-v3.pt' \
  -e "$SSH" ./ "$TARGET:/workspace/drone-flyby/" || exit 1

echo "== verifying the four models arrived intact"
$SSH "$TARGET" 'cd /workspace/drone-flyby && python3 - <<PY
import hashlib, sys
from pathlib import Path
need = ["drone-yolo11n-v4.pt","drone-yolo11s-v6.pt","drone-yolo11m-v8.pt","drone-yolo11m-p2-v9.pt"]
bad = [n for n in need if not (Path("models")/n).exists()]
if bad: sys.exit("MISSING: " + ", ".join(bad))
for n in need:
    print(f"   {n:28s} {(Path('models')/n).stat().st_size:>10,} bytes")
PY' || exit 1

echo "== deps (--no-deps ultralytics protects the NGC torch build)"
$SSH "$TARGET" 'pip install -q --no-deps ultralytics==8.4.152 2>/dev/null; \
  pip install -q opencv-python-headless py-cpuinfo ultralytics-thop polars faster_coco_eval 2>/dev/null; \
  python3 -c "import torch,ultralytics,cv2; print(\"   torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"'

echo
echo "== now start the arm (internal 6006; check the mapped public port):"
echo "   $SSH $TARGET"
echo "   cd /workspace/drone-flyby && PYTHON=python3 PORT=6006 tools/arm.sh SHIP \\"
echo "     \$(cat SHIP_ARGS)"
