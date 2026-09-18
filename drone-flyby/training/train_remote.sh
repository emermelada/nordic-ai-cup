#!/usr/bin/env bash
# Train a Drone Flyby model on a rented GPU box (Vast.ai, Runpod, ...).
#
#   export KAGGLE_API_TOKEN=KGAT_...                 # kaggle.com > Settings > API
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...        # older token style, also fine
#   bash training/train_remote.sh                    # yolo11m, 40 epochs
#   MODEL=yolo11l.pt EPOCHS=50 bash training/train_remote.sh
#   WITH_INRIA=0 bash training/train_remote.sh       # skip the 22 GB city photos
#   REAL_SHARE=0.8 bash training/train_remote.sh     # lean harder on real terrain
#
# v6, trained to COMPLEMENT v4 rather than to beat it (see DRONE_MODEL_ALT in
# flyby.py: two models take alternate frames and meet in the object memory, so
# what matters is what the pair covers, not what either scores alone):
#
#   MODEL=yolo11s.pt IMGSZ=1280 NAME=drone-yolo11s-v6 \
#   WEIGHTS=spacecraft=3,small_launcher=3,tank=2.5,small_plane=2,mine_roller=1.5,jammer=1.5,condor=1.3,ta-ta=1.3,medium_plane=1.3,medium_launcher=1.3,helicopter=0.7,small_tower=0.8,large_launcher=0.8,jet_plane=0.5,hangar=0.4,large_tower=0.4 \
#   bash training/train_remote.sh
#
# Why those numbers, measured on the recorded flight with the v4+v5 pair:
# spacecraft 5 %, small_launcher 6 %, tank 25 %, small_plane 35 % are what the
# pair still misses, while hangar 89 %, large_tower 88 % and jet_plane 64 % are
# already covered by v4 - so v6 buys nothing by learning them again and the
# paste budget is better spent elsewhere. They stay in the mix at a low weight
# rather than being dropped, so v6 still learns not to call a hangar a tank.
# IMGSZ=1280 because running v4 at 1280 instead of 960 moved tank 10 % -> 27 %
# and mine_roller 3 % -> 28 % with no retraining at all; training there also
# fixes the box regression, which is what got worse when only inference was
# upscaled (bad boxes 12 % -> 17 %).
#
# Serve the pair with:  DRONE_MODEL=models/drone-yolo11n-v4.pt \
#                       DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt
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
NAME=${NAME:-drone-$(basename "$MODEL" .pt)-v6}
# Backgrounds cut from the recorded validation flight (training/make_real_backgrounds.py).
# Measured on 2026-09-18: v4 scores median IoU 0.93 and 100% correct class on
# Helsinki but 31% per-frame recall on validation, so the background domain is
# the gap. These are the only backgrounds we have from the real thing.
REAL_BACKGROUNDS=${REAL_BACKGROUNDS:-training/backgrounds_real}
REAL_SHARE=${REAL_SHARE:-0.6}
# Background variety was the single biggest gain (v1 -> v2), so both photo sets
# are used by default; Inria is a 22 GB download.
WITH_INRIA=${WITH_INRIA:-1}
# Weak classes pasted more often; v4 used up to 2.5x and lost tank/jammer/
# spacecraft, so nothing here goes above 2.0. Set from per-class AP on the
# recorded flight (tools/score_offline.py), re-measured 18 Sep after fixing the
# ground truth, which had been carried with the tracker's own motion:
#   tank 0.021 and 167 of its 219 object-frames missed outright - the most
#   common object in the flight and our single biggest loss;
#   spacecraft 0.007 (52 of 64 missed), mine_roller 0.000, large_tower 0.001.
#   small_launcher 0.000 but it measures 18px in the flight, ~9px at Level 1,
#   at or below YOLO's finest stride - likely a resolution floor, so it gets a
#   small bump rather than a big one.
#   jammer 0.228 and helicopter 0.295 are NOT dead - weights back to 1.0.
#   small_plane 0.071 is a box problem, not a firing problem (39 bad boxes
#   against 31 hits), so pasting it more often will not help; the mask and
#   paste-scale fixes are what it needs.
# The four classes never confirmed in validation stay up, since the evaluation
# flight is a different one and may contain them.
WEIGHTS=${WEIGHTS:-tank=2.0,spacecraft=1.8,mine_roller=1.6,large_tower=1.6,small_launcher=1.3,condor=1.3,ta-ta=1.3,medium_plane=1.3,medium_launcher=1.3}

REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$WORK"

echo "== packages"
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq git unzip libgl1 libglib2.0-0 >/dev/null
fi
pip install -q "ultralytics==8.4.152" "kaggle>=1.7"

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

if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
    # Newer Kaggle token style: the client reads it from this file.
    mkdir -p ~/.kaggle && printf %s "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token
fi
if [ ! -s ~/.kaggle/access_token ] && [ ! -s ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_KEY:-}" ]; then
    echo "No Kaggle credentials: set KAGGLE_API_TOKEN (or KAGGLE_USERNAME/KAGGLE_KEY)." >&2
    exit 1
fi

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
REAL_ARGS=()
if [ -d "$REAL_BACKGROUNDS" ]; then
    echo "== real backgrounds: $(ls "$REAL_BACKGROUNDS" | wc -l) frames of the flight itself"
    REAL_ARGS=(--real-backgrounds "$REAL_BACKGROUNDS" --real-share "$REAL_SHARE")
else
    echo "!! $REAL_BACKGROUNDS missing - training on stock photos only, which is what v1-v5 did" >&2
fi
python training/make_dataset.py \
    --scenes "$SCENES" --out "$WORK/yolo" \
    --backgrounds "$WORK/backgrounds" \
    "${REAL_ARGS[@]}" \
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
