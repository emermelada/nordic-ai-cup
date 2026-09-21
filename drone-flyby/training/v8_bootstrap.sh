#!/usr/bin/env bash
# Paste this on the Vast box after `ssh`-ing in. Survives a dropped connection.
set -euo pipefail

export KAGGLE_API_TOKEN=REPLACE_ME          # kaggle.com > Settings > API

apt-get update -qq && apt-get install -y -qq tmux git >/dev/null

cd /workspace
[ -d nordic-ai-cup ] || git clone -b drone-flyby-real-backgrounds \
    https://github.com/emermelada/nordic-ai-cup.git
cd nordic-ai-cup/drone-flyby

# Fail fast on the Blackwell trap before spending 20 minutes on the dataset.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda);
assert torch.cuda.is_available(), 'no GPU visible';
print('device', torch.cuda.get_device_name(0))"

tmux new-session -d -s v8 "
  MODEL=yolo11m.pt IMGSZ=1280 NAME=drone-yolo11m-v8 BATCH=12 \
  WEIGHTS=tank=2.0,helicopter=1.8,small_plane=1.6,jammer=1.8,spacecraft=1.5,mine_roller=1.3,condor=1.2,ta-ta=1.2,medium_plane=1.2,medium_launcher=1.2,small_launcher=0.9,small_tower=0.7,large_launcher=0.7,large_tower=0.5,jet_plane=0.5,hangar=0.4 \
  bash training/train_remote.sh 2>&1 | tee /workspace/v8.log
"
echo
echo "started in tmux session 'v8'"
echo "  watch it:     tmux attach -t v8       (detach with ctrl-b then d)"
echo "  or tail it:   tail -f /workspace/v8.log"
echo "  weights land: /workspace/drone-yolo11m-v8.pt"
