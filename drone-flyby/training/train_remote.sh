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
# ---------------------------------------------------------------------------
# v8 RECIPE (19 Sep) - the one to run next. Read this before the v6 notes below.
#
#   MODEL=yolo11m.pt IMGSZ=1280 NAME=drone-yolo11m-v8 BATCH=12 \
#   WEIGHTS=tank=2.0,helicopter=1.8,small_plane=1.6,jammer=1.8,spacecraft=1.5,mine_roller=1.3,condor=1.2,ta-ta=1.2,medium_plane=1.2,medium_launcher=1.2,small_launcher=0.9,small_tower=0.7,large_launcher=0.7,large_tower=0.5,jet_plane=0.5,hangar=0.4 \
#   bash training/train_remote.sh
#
# v8 is the STRONG HALF OF THE PAIR, meant to replace v6@1280 beside v4@960.
# Judge it that way - paired, not alone - with:
#
#   python tools/score_offline.py --replay \
#       --model models/drone-yolo11n-v4.pt:960 --model-alt models/drone-yolo11m-v8.pt:1280 \
#       --set BOTH_MODELS=1
#
#   The served pair scores 0.455 on that command as of 19 Sep. Beat it there
#   before spending a validation attempt; the offline scorer now ranks six
#   real-scored configurations at Spearman +0.94, so it can be trusted for this.
#
# WHY THESE WEIGHTS. They come from per-class AP of the served pair under the
# corrected scorer (19 Sep), read together with how many scored object-frames
# each class actually has - which is the part the old weights ignored:
#
#   class            AP     object-frames    share
#   tank            0.354       219          24 %   <- biggest single loss
#   helicopter      0.249       100          11 %
#   small_plane     0.225        99          11 %
#   jammer          0.201        99          11 %
#   spacecraft      0.003        64           7 %   <- dead, but 12.2 px at L1
#   small_tower     0.756        78
#   hangar          0.871        70
#   jet_plane       0.958        66
#   mine_roller     0.082        33
#   large_tower     0.831        33
#   small_launcher  0.000        33                 <- 6.4 px at L1: hopeless
#   large_launcher  0.682        17
#
# The four mid-scoring, high-volume classes - tank, helicopter, small_plane,
# jammer - are 57 % of everything scored and all sit at 0.20-0.35. That is where
# the points are. This is a CHANGE OF TARGET from v6, which was aimed at the
# tiny dead classes on the old scorer's reading that they were 42 % of the
# score; corrected, the genuinely dead classes (spacecraft, small_launcher) are
# 11 %, and large_tower turned out to be 0.831 rather than 0.001.
#
# small_launcher stays under 1.0 deliberately: 6.4 px at Level 1 against YOLO's
# 8 px stride. No amount of pasting fixes a resolution floor.
# small_plane is held at 1.6 rather than 1.8 because its failure was measured as
# box quality, not firing rate - pasting it more often is not the fix.
# Nothing exceeds 2.0: v4 used 2.5x and lost tank, jammer and spacecraft.
#
# WHY yolo11m. Untested on this recipe. Against it: v5 IS yolo11m and scores
# 0.322 against the served pair's 0.455, and v7 (yolo11s at 1280) came out 0.023
# worse than v6. For it: v5 predates both the real-flight backgrounds that made
# v6 work and the corrected Helsinki paste scale. If v8 disappoints, re-run this
# exact recipe with MODEL=yolo11s.pt before concluding the data is at fault -
# that isolates backbone from recipe.
#
# WHY IMGSZ=1280 AND NOT MORE. Measured 19 Sep: v6 alone scores 0.288 at 960,
# 0.398 at 1280, 0.370 at 1600. 1280 is the peak, not a floor still being
# climbed. Do not raise it.
#
# The cut-outs in training/patches_val were re-harvested on 19 Sep after three
# mined objects were promoted into validation_objects.json: large_tower 7 -> 22
# patches, mine_roller 10 -> 19, 263 -> 288 total.
# ---------------------------------------------------------------------------
#
# TWO WAYS TO TRAIN v6. The default below is a STANDALONE v6 meant to replace
# v4. The alternative is a v6 trained to COMPLEMENT v4 as a pair (DRONE_MODEL_ALT
# in flyby.py: two models take alternate frames and meet in the object memory, so
# what matters is what the pair covers, not what either scores alone):
#
#   MODEL=yolo11s.pt IMGSZ=1280 NAME=drone-yolo11s-v6 \
#   WEIGHTS=spacecraft=3,small_launcher=3,tank=2.5,small_plane=2,mine_roller=1.5,jammer=1.5,condor=1.3,ta-ta=1.3,medium_plane=1.3,medium_launcher=1.3,helicopter=0.7,small_tower=0.8,large_launcher=0.8,jet_plane=0.5,hangar=0.4,large_tower=0.4 \
#   bash training/train_remote.sh
#
#   Serve the pair with:  DRONE_MODEL=models/drone-yolo11n-v4.pt \
#                         DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt
#
# Pick that one ONLY if you are certain you will serve both models. It holds
# hangar at 0.4, jet_plane at 0.5 and large_tower at 0.4 on the grounds that v4
# already covers them - which is true of the pair and false of v6 alone. Those
# are three of our best classes (AP 0.871, 0.833, and large_tower is the one v4
# does badly at 0.001), so a v6 trained this way and then served on its own
# would regress them. The pair itself is so far measured only offline.
#
# Its other two ideas are worth taking either way. IMGSZ=1280: running v4 at 1280
# instead of 960 moved tank 10 % -> 27 % and mine_roller 3 % -> 28 % with no
# retraining at all, and training there also fixes the box regression that
# appeared when only inference was upscaled (bad boxes 12 % -> 17 %). Decide it
# by measuring latency on the machine that will actually serve. And a low weight
# rather than zero on the strong classes, so v6 still learns not to call a hangar
# a tank.
#
# Note the per-class figures quoted in that recipe (spacecraft 5 %, tank 25 %,
# hangar 89 %) predate the ground-truth motion fix in this same branch; see the
# WEIGHTS comment below for the corrected table.
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
