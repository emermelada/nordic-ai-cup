"""Simulate the clause-splitting rule and measure best-unit / best-2-adjacent-unit ceilings."""
from pathlib import Path
import csv, json, sys, statistics as st, itertools
OUT=sys.argv[1]
CSV=str(Path(__file__).resolve().parents[2] / 'data' / 'question_train.csv')
rows=[r for r in csv.DictReader(open(CSV)) if r['evidence_start']]
def iou(a,b):
    i=max(0,min(a[1],b[1])-max(a[0],b[0])); u=max(a[1],b[1])-min(a[0],b[0]); return i/u if u>0 else 0
def split_units(words, pause=0.5, comma_min=6, max_words=18):
    units=[]; cur=[]
    for i,w in enumerate(words):
        if cur:
            prev=words[i-1]; t=prev['word'].strip()
            gap=w['start']-prev['end']
            if t.endswith(('.', '?', '!')) or (t.endswith((',',';',':')) and len(cur)>=comma_min) or gap>=pause or len(cur)>=max_words:
                units.append(cur); cur=[]
        cur.append(w)
    if cur: units.append(cur)
    # merge tiny units forward
    merged=[]
    for u in units:
        if merged and len(merged[-1])<3 and (u[0]['start']-merged[-1][-1]['end'])<1.0:
            merged[-1]=merged[-1]+u
        else: merged.append(u)
    return [(u[0]['start'],u[-1]['end']) for u in merged]
def ceilings(pause, comma_min, max_words):
    s1,s2,lens=[],[],[]
    for r in rows:
        d=json.load(open(f"{OUT}/conversation_{r['transcript_id']}.json"))
        words=[w for s in d['segments'] for w in s['words']]
        units=split_units(words,pause,comma_min,max_words)
        lens += [b-a for a,b in units]
        g=(float(r['evidence_start']),float(r['evidence_end']))
        s1.append(max(iou(g,u) for u in units))
        s2.append(max(max(iou(g,(units[i][0],units[j][1])) for j in range(i,min(i+2,len(units)))) for i in range(len(units))))
    return st.mean(s1), st.mean(s2), st.median(lens)
print(f"{'pause':>6} {'comma':>6} {'maxw':>5} | best-unit  best-2-adjacent  median-unit-len")
for pause, cm, mw in [(0.5,6,18),(0.4,6,18),(0.6,6,18),(0.5,4,18),(0.5,8,18),(0.5,99,18),(0.5,6,12),(0.3,4,12),(9,99,999)]:
    a,b,c=ceilings(pause,cm,mw); print(f'{pause:>6} {cm:>6} {mw:>5} | {a:.3f}      {b:.3f}            {c:.2f}s')
