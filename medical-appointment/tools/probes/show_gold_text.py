"""For each positive question: the words inside the gold span, and the whisper segment(s) containing it."""
from pathlib import Path
import csv, json, sys, os
OUT = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
CSV = str(Path(__file__).resolve().parents[2] / 'data' / 'question_train.csv')
rows = [r for r in csv.DictReader(open(CSV)) if r['evidence_start']]
for r in rows:
    if only and r['transcript_id'] != only: continue
    f = f"{OUT}/conversation_{r['transcript_id']}.json"
    if not os.path.exists(f): continue
    d = json.load(open(f)); segs = d['segments']
    g0, g1 = float(r['evidence_start']), float(r['evidence_end'])
    words = [w for s in segs for w in s['words'] if min(w['end'], g1) - max(w['start'], g0) > 0]
    segtxt = [f"[{s['start']:.2f}-{s['end']:.2f}]{s['text']}" for s in segs if min(s['end'], g1) - max(s['start'], g0) > 0]
    print(f"{r['transcript_id']:<10} gold {g0:.2f}-{g1:.2f} ({g1-g0:.2f}s)  Q: {r['question']}")
    print(f"    GOLD WORDS: {''.join(w['word'] for w in words).strip()}")
    for s in segtxt: print(f"    SEG: {s}")
