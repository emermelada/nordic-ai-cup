#!/usr/bin/env bash
# Sweep serving settings through the REAL predict path, offline, on the box.
#
#   tools/sweep_remote.sh <run-id> [<run-id> ...]
#
# score_offline --replay re-runs flyby.predict against cached detections, so
# unlike rescaling stored answers this exercises tracking, memory carrying and
# box growth exactly as served. On a 5090 a run is seconds, so a sweep that
# would otherwise cost a day of validation attempts costs minutes and spends
# none of them.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
GROW='condor=1.30x1.30,hangar=1.30x1.30,helicopter=1.17x1.87,jammer=0.78x1.09,jet_plane=1.37x1.37,large_launcher=1.17x1.40,large_tower=1.23x1.23,medium_launcher=1.30x1.30,medium_plane=1.30x1.30,mine_roller=1.10x1.66,small_launcher=1.30x1.30,small_plane=1.04x1.25,small_tower=1.04x1.14,spacecraft=1.30x1.30,ta-ta=1.30x1.30,tank=1.30x1.43'
MODELS=(--model models/drone-yolo11n-v4.pt:960
        --model-alt models/drone-yolo11s-v6.pt:1280
        --model-alt models/drone-yolo11m-v8.pt:1280
        --model-alt models/drone-yolo11m-v8.pt:2560
        --model-alt models/drone-yolo11m-p2-v9.pt:1280)

RUNS="$*"
[ -z "$RUNS" ] && { echo "usage: tools/sweep_remote.sh <run-id> [...]" >&2; exit 1; }

printf '%-30s' 'arm'
for r in $RUNS; do printf '%10s' "${r:0:8}"; done; printf '%10s\n' 'mean'

one () {  # one <label> <grow-env...> -- <SET=v ...>
    local label=$1; shift
    local envs=() sets=()
    while [ "$1" != '--' ]; do envs+=("$1"); shift; done; shift
    for s in "$@"; do sets+=(--set "$s"); done
    local sum=0 n=0 line=''
    for run in $RUNS; do
        local v
        v=$(env "${envs[@]}" DRONE_CAMERA=row0 DRONE_LEVEL0_WEIGHT=1.5 DRONE_DEVICE=cuda \
            "$PY" tools/score_offline.py --replay "${MODELS[@]}" "${sets[@]}" \
            --run "$run" 2>/dev/null | grep -oP 'mAP@0\.50.*= \K[0-9.]+')
        [ -z "$v" ] && v=0
        line="$line$(printf '%10s' "$v")"
        sum=$(awk -v a="$sum" -v b="$v" 'BEGIN{print a+b}'); n=$((n+1))
    done
    printf '%-30s%s%10s\n' "$label" "$line" \
        "$(awk -v s="$sum" -v n="$n" 'BEGIN{printf "%.4f", s/n}')"
}

OLD=(DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3)
NEW=(DRONE_BOX_GROW="$GROW" DRONE_BOX_GROW_CAP=1.9 DRONE_BOX_GROW_WH=1)
B=(BOTH_MODELS=1 NEW_TRACK_CONFIDENCE=0.10)

one 'A current 1.3'        "${OLD[@]}" -- "${B[@]}" MAX_MISSES=12
one 'D new geometry'       "${NEW[@]}" -- "${B[@]}" MAX_MISSES=12
one 'D + MAX_MISSES=20'    "${NEW[@]}" -- "${B[@]}" MAX_MISSES=20
one 'D + MAX_MISSES=40'    "${NEW[@]}" -- "${B[@]}" MAX_MISSES=40
one 'D + UNSEEN_DECAY=1.0' "${NEW[@]}" -- "${B[@]}" MAX_MISSES=12 UNSEEN_DECAY=1.0
one 'D + track_conf 0.05'  "${NEW[@]}" -- BOTH_MODELS=1 NEW_TRACK_CONFIDENCE=0.05 MAX_MISSES=12
