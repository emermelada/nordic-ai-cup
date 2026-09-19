#!/usr/bin/env bash
# run_discovery_analyse.sh - merge the parallel generation output and run the decision-boundary analysis.
set -u
cd /opt/nac/experiments
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy
OUT=/opt/nac/discovery

echo "=== merge ==="
cat "$OUT"/rows_*.jsonl > "$OUT"/all_rows.jsonl
echo "  rows: $(wc -l < "$OUT"/all_rows.jsonl)"
echo "  size: $(du -h "$OUT"/all_rows.jsonl | cut -f1)"

echo
echo "=== decision-boundary analysis (depth 3, min leaf 25) ==="
/opt/nacv/bin/python discover.py analyse --inputs "$OUT/all_rows.jsonl" --depth 3 --min-leaf 25 2>&1 | tail -45
