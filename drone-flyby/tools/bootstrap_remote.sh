#!/usr/bin/env bash
# Put this repo on a fresh rented box and start a serving arm, in one command.
#
#   tools/bootstrap_remote.sh -p 41333 root@1.2.3.4
#   ARM_ARGS="DRONE_SET=BOTH_MODELS=1,AGREEMENT_WEIGHT=0.7" tools/bootstrap_remote.sh -p ... root@...
#
# Everything here was learned the hard way on 19 Sep; the comments say what
# each step is defending against.
set -eu
cd "$(dirname "$0")/.."
SSH_ARGS="$*"
REMOTE=/workspace/drone-flyby
RUN=${RUN:-5ace53648dd5429bbf95332494a69ac0}      # recording preflight replays
ARM=${ARM:-FINAL}
ARM_ARGS=${ARM_ARGS:-"DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10"}

say() { printf '\n== %s\n' "$*"; }

say "reachable?"
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 $SSH_ARGS 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'

say "GPU actually computes (a real matmul, not a version string)"
ssh $SSH_ARGS 'python3 -c "
import torch
a=torch.randn(2048,2048,device=\"cuda\"); c=a@a; torch.cuda.synchronize()
assert torch.isfinite(c).all(), \"matmul produced non-finite values\"
print(torch.cuda.get_device_name(0), \"capability\", torch.cuda.get_device_capability(0), \"OK\")"'

say "LATENCY TO THE EVALUATOR -- the gate that matters most"
# The evaluator is Hetzner Helsinki (46.62.240.126 in our serve logs). A camera
# command has a TIGHTER deadline than a frame: it must arrive before the next
# frame is rendered, not merely within the 333 ms budget. Measured 19 Sep, a
# Bulgarian box at 58-89 ms RTT stalled the sweep on 5.7-7.8 % of steps and
# scored 0.42-0.46, where a low-latency box stalled 0.4-3.6 % and scored
# 0.50-0.51. No frames were lost in either case -- the loss is pure coverage.
ssh $SSH_ARGS 'python3 -c "
import socket, time, statistics, sys  # spaced: 100 rapid connects trip Hetzner rate limiting and fake a jitter failure
lat=[]
for _ in range(30):
    s=socket.socket(); s.settimeout(2); t=time.perf_counter()
    try: s.connect((\"hel1-speed.hetzner.com\", 443))
    except Exception: continue
    lat.append((time.perf_counter()-t)*1000); s.close(); time.sleep(0.25)
lat.sort()
med=statistics.median(lat); p95=lat[int(.95*len(lat))]
print(f\"   Helsinki RTT: median {med:.1f} ms  p95 {p95:.1f} ms  min {lat[0]:.1f}  max {lat[-1]:.1f}  n={len(lat)}\")
if med > 35: sys.exit(\"   REJECT: median over 35 ms -- destroy this box and rent closer to Helsinki\")
if p95 > 2.5*med: sys.exit(\"   REJECT: p95 is more than 2.5x the median -- jittery host, camera commands will miss deadlines\")
print(\"   ACCEPT\")"'

say "which ports does this box actually expose?"
ssh $SSH_ARGS 'env | grep ^VAST_TCP_PORT_ | sort'
echo "   pick a FREE one below; 9053 is usually NOT mapped and Jupyter usually holds 8080."
ssh $SSH_ARGS 'python3 -c "
import os, socket
for k,v in sorted(os.environ.items()):
    if k.startswith(\"VAST_TCP_PORT_\"):
        p=int(k.rsplit(\"_\",1)[1]); s=socket.socket(); s.settimeout(1)
        print(f\"   internal {p:5d} -> public {v:>6s}  {\"HELD\" if s.connect_ex((\"127.0.0.1\",p))==0 else \"free\"}\")"'

say "uploading the repo (models included, training inputs and frames excluded)"
rsync -a --info=stats2 -e "ssh $(echo "$SSH_ARGS" | sed 's/[^ ]*@.*//')" \
  --exclude 'src/' --exclude 'data/' --exclude '__pycache__/' --exclude '.venv/' \
  --exclude '*.log' --exclude 'yolo11m.pt' --exclude 'training/patches_val/' \
  --exclude 'training/backgrounds_real/' \
  ./ "$(echo "$SSH_ARGS" | grep -o '[^ ]*@[^ ]*')":$REMOTE/ | tail -3

say "uploading 45 frames of one recording, so preflight has something real to replay"
ssh $SSH_ARGS "mkdir -p $REMOTE/data/recordings/$RUN"
ls "data/recordings/$RUN" | sort | head -90 > /tmp/.bootstrap_frames
rsync -a -e "ssh $(echo "$SSH_ARGS" | sed 's/[^ ]*@.*//')" --files-from=/tmp/.bootstrap_frames \
  "data/recordings/$RUN/" "$(echo "$SSH_ARGS" | grep -o '[^ ]*@[^ ]*')":$REMOTE/data/recordings/$RUN/

say "weights intact?"
LOCAL=$(sha1sum models/drone-yolo11n-v4.pt models/drone-yolo11s-v6.pt models/drone-yolo11m-v8.pt | cut -c1-16)
THERE=$(ssh $SSH_ARGS "cd $REMOTE && sha1sum models/drone-yolo11n-v4.pt models/drone-yolo11s-v6.pt models/drone-yolo11m-v8.pt | cut -c1-16")
[ "$LOCAL" = "$THERE" ] && echo "   three served weights match" || { echo "   MISMATCH - do not serve this"; exit 1; }

say "dependencies, WITHOUT disturbing the NGC torch"
# The box ships torch 2.14.0a0 / torchvision 0.29.0a0. Ultralytics declares
# torch>=1.8.0 and pip EXCLUDES pre-releases from that, so a plain
# `pip install -r requirements.txt` decides torch is unsatisfied, installs a
# stable build over the top and breaks CUDA silently. Hence --no-deps.
ssh $SSH_ARGS 'pip install -q --no-deps ultralytics==8.4.152 && \
  pip install -q opencv-python-headless py-cpuinfo ultralytics-thop polars \
      "ultralytics-platform>=0.1.32" fastapi "uvicorn[standard]" faster-coco-eval 2>&1 | tail -2; true'

say "CUDA still works after the install (this is the check that catches it)"
ssh $SSH_ARGS 'python3 -c "
import importlib.metadata as md, torch, torchvision
print(\"torch\", md.version(\"torch\"), \"torchvision\", md.version(\"torchvision\"))
a=torch.randn(512,512,device=\"cuda\"); assert torch.isfinite(a@a).all()
n=torchvision.ops.nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]],device=\"cuda\"),
                      torch.tensor([0.9,0.8],device=\"cuda\"),0.5)
print(\"cuda matmul OK, torchvision.ops.nms OK ->\", n.tolist())"'

echo
echo "Ready. Start the arm with the free internal port from the list above, e.g.:"
echo "  ssh $SSH_ARGS 'cd $REMOTE && PYTHON=python3 PORT=<free> DRONE_DEVICE=cuda bash tools/arm.sh $ARM $ARM_ARGS'"
echo "Then submit http://<public ip>:<mapped public port>/predict"
