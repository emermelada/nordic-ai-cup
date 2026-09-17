#!/usr/bin/env bash
# Train a Drone Flyby model on a rented GPU box (Vast.ai, Runpod, ...).
#
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...        # or copy ~/.kaggle/kaggle.json
#   bash training/train_remote.sh                    # yolo11m, 40 epochs
#   MODEL=yolo11l.pt EPOCHS=50 bash training/train_remote.sh
#   WITH_INRIA=0 bash training/train_remote.sh       # skip the 22 GB city photos
#
# Run it from a clone of this repo. It installs what it needs, fetches the
# background photos and the official Helsinki frames, builds the synthetic
# dataset and trains. The finished weights land in $WORK/<name>.pt.
#
# The RTX 5090 is Blackwell: it needs PyTorch with CUDA 12.8 or newer, or every
# run dies with "no kernel image is available for execution on the device".
# The script checks that before spending time on the dataset.
set -euo pipefail

MODEL=${MODEL:-yolo11m.pt}          # yolo11s/m/l, or a .pt of ours to fine-tune
EPOCHS=${EPOCHS:-40}
SCENES=${SCENES:-1600}              # 6 views each: 1600 -> 9600 images
BATCH=${BATCH:-16}                  # 32 GB VRAM at 960 px fits 32-48 for m
IMGSZ=${IMGSZ:-960}
WORK=${WORK:-/workspace}
NAME=${NAME:-drone-$(basename "$MODEL" .pt)-v5}
# Background variety was the single biggest gain (v1 -> v2), so both photo sets
# are used by default; Inria is a 22 GB download.
WITH_INRIA=${WITH_INRIA:-1}
# Weak classes pasted more often; v4 used up to 2.5x and lost tank/jammer/
# spacecraft, so keep it gentle.
WEIGHTS=${WEIGHTS:-small_plane=1.5,large_tower=1.5,medium_plane=1.5,medium_launcher=1.5,jammer=1.3,large_launcher=1.3,small_launcher=1.3,ta-ta=1.3,condor=1.3}

REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$WORK"

echo "== packages"
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq git unzip libgl1 libglib2.0-0 >/dev/null
fi
pip install -q "ultralytics==8.4.152" kaggle

echo "== GPU"
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'no GPU visible'
name = torch.cuda.get_device_name(0)
print('device', name)
major = torch.cuda.get_device_capability(0)[0]
if major >= 12 and tuple(int(p) for p in torch.__version__.split('.')[:2]) < (2, 7):
    raise SystemExit('Blackwell GPU needs torch >= 2.7 built for CUDA 12.8: '
                     'pip install torch --index-url https://download.pytorch.org/whl/cu128')
PY

echo "== official repo (Helsinki frames, utils.py, dtos.py)"
if [ ! -d "$WORK/official" ]; then
    git clone --depth 1 https://github.com/amboltio/Nordic-AI-Cup-2026.git "$WORK/official"
fi
FLYBY="$WORK/official/drone-flyby"
cp -r "$REPO/training" "$FLYBY/"

echo "== background photos"
mkdir -p "$WORK/backgrounds"
if [ ! -d "$WORK/backgrounds/landcoverai" ]; then
    kaggle datasets download -d adrianboguszewski/landcoverai -p "$WORK/backgrounds/landcoverai" --unzip
fi
if [ "$WITH_INRIA" = 1 ] && [ ! -d "$WORK/backgrounds/inria" ]; then
    kaggle datasets download -d sagar100rathod/inria-aerial-image-labeling-dataset -p "$WORK/backgrounds/inria" --unzip
fi

echo "== dataset"
cd "$FLYBY"
export OPENCV_LOG_LEVEL=ERROR
python training/extract_patches.py
python training/make_dataset.py \
    --scenes "$SCENES" --out "$WORK/yolo" \
    --backgrounds "$WORK/backgrounds" \
    --extra-patches training/patches_val \
    --helsinki-share 0.1 \
    --class-weights "$WEIGHTS"

echo "== train $MODEL, $EPOCHS epochs"
MODEL="$MODEL" EPOCHS="$EPOCHS" BATCH="$BATCH" IMGSZ="$IMGSZ" WORK="$WORK" NAME="$NAME" python - <<'PY'
import os, shutil, psutil
from ultralytics import YOLO

work, name = os.environ['WORK'], os.environ['NAME']
# Caching the images in RAM keeps a fast GPU from waiting on PNG decoding;
# ~1.6 GB per 1000 images at 960x540.
images = len(list((__import__('pathlib').Path(work) / 'yolo/images/train').glob('*')))
free_gb = psutil.virtual_memory().available / 1e9
cache = 'ram' if free_gb > images * 0.0017 + 8 else 'disk'
print(f'{images} training images, {free_gb:.0f} GB RAM free -> cache={cache}')

model = YOLO(os.environ['MODEL'])
model.train(
    data=f'{work}/yolo/data.yaml',
    imgsz=int(os.environ['IMGSZ']),
    epochs=int(os.environ['EPOCHS']),
    batch=int(os.environ['BATCH']),
    device=0,
    cache=cache,
    workers=min(16, os.cpu_count() or 8),
    # The drone flies at a fixed altitude: sizes vary little, and the pasting
    # already rotates and recolours each object.
    scale=0.1,
    degrees=0.0,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.03,
    project=f'{work}/runs',
    name=name,
    plots=True,
)
shutil.copy(f'{work}/runs/{name}/weights/best.pt', f'{work}/{name}.pt')
print(f'\nDone: {work}/{name}.pt')
PY

echo
echo "Copy it off the box before destroying the instance, e.g.:"
echo "  scp -P <port> root@<host>:$WORK/$NAME.pt ."
