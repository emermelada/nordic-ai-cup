"""Upper-bound analysis: how well could segment-level vs word-level spans match the gold spans?"""
from pathlib import Path
import csv, json, sys, os, statistics, glob
OUT = sys.argv[1]
CSV = str(Path(__file__).resolve().parents[2] / 'data' / 'question_train.csv')
rows = [r for r in csv.DictReader(open(CSV)) if r['evidence_start']]

def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

def best_segment(gold, segs):
    return max((iou(gold, (s['start'], s['end'])) for s in segs), default=0.0)

def best_word_span(gold, words):
    # best contiguous run of words (any i<=j) -> O(n^2) but n ~ 400 fine
    best = 0.0
    n = len(words)
    for i in range(n):
        if words[i]['start'] > gold[1] + 5: break
        for j in range(i, n):
            if words[j]['start'] > gold[1] + 5: break
            v = iou(gold, (words[i]['start'], words[j]['end']))
            if v > best: best = v
    return best

def best_merged_segments(gold, segs, k=3):
    best = 0.0
    for i in range(len(segs)):
        for j in range(i, min(i + k, len(segs))):
            v = iou(gold, (segs[i]['start'], segs[j]['end']))
            if v > best: best = v
    return best

seg_scores, word_scores, merged_scores, seg_lens = [], [], [], []
snap_start, snap_end = [], []
missing = 0
per_file = {}
for r in rows:
    f = f"{OUT}/conversation_{r['transcript_id']}.json"
    if not os.path.exists(f):
        missing += 1; continue
    d = json.load(open(f))
    segs = d['segments']
    words = [w for s in segs for w in s['words']]
    gold = (float(r['evidence_start']), float(r['evidence_end']))
    s = best_segment(gold, segs); w = best_word_span(gold, words); m = best_merged_segments(gold, segs)
    seg_scores.append(s); word_scores.append(w); merged_scores.append(m)
    # distance of gold boundaries to nearest word boundary
    if words:
        snap_start.append(min(abs(gold[0] - x['start']) for x in words))
        snap_end.append(min(abs(gold[1] - x['end']) for x in words))
    per_file.setdefault(r['transcript_id'], []).append((r['question'], round(s,3), round(w,3)))
for f in glob.glob(f'{OUT}/conversation_*.json'):
    d = json.load(open(f)); seg_lens += [s['end'] - s['start'] for s in d['segments']]

def summ(name, xs):
    print(f'{name:<38} mean {statistics.mean(xs):.3f}  median {statistics.median(xs):.3f}  n={len(xs)}')
print(f'gold rows scored: {len(seg_scores)}  (missing transcripts for {missing})')
summ('UPPER BOUND tIoU: best single segment', seg_scores)
summ('UPPER BOUND tIoU: best 1-3 merged segs', merged_scores)
summ('UPPER BOUND tIoU: best word span', word_scores)
summ('segment length (s)', seg_lens)
summ('gold start -> nearest word start (s)', snap_start)
summ('gold end   -> nearest word end (s)', snap_end)
print('word-span upper bound < 0.5 count:', sum(1 for x in word_scores if x < 0.5))
print('segment upper bound < 0.5 count:', sum(1 for x in seg_scores if x < 0.5))
if '--verbose' in sys.argv:
    for tid, items in per_file.items():
        for q, s, w in items: print(f'{tid:<10} seg={s:.3f} word={w:.3f}  {q}')
tim = f'{OUT}/_timings.json'
if os.path.exists(tim):
    t = json.load(open(tim)); vals = list(t.values())
    print(f'ASR time per file: mean {statistics.mean(vals):.1f}s  max {max(vals):.1f}s  total {sum(vals):.0f}s')
