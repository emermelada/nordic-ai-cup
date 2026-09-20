"""Split each class's failure into MISSED / MISNAMED / RANKED.

These need completely different fixes, and right now we do not know which one
any of the five dead classes has.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT = Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import score_offline as S
import flyby

run = sys.argv[1]
objects = json.loads(S.OBJECTS.read_text())['objects']
frames = list(range(1, 250))
motion = S.fit_truth_motion(objects)[0]
f = S.box_factors()
if f:
    objects = S.loosen(objects, f); motion = S.fit_truth_motion(objects)[0]
preds = S.from_recording(run)

def iou(a, b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1]); x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    w,h = max(0,x2-x1), max(0,y2-y1); i = w*h
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/ua if ua>0 else 0.0

stat = collections.defaultdict(collections.Counter)
conf = collections.defaultdict(collections.Counter)
rank = collections.defaultdict(list)
size = collections.defaultdict(list)
for fr in frames:
    P = preds.get(fr, [])
    P = sorted(P, key=lambda p: -p[1])
    for name, box in S.known_boxes(objects, fr, motion):
        size[name].append(np.sqrt((box[2]-box[0])*(box[3]-box[1])))
        hits = [(i,p) for i,p in enumerate(P) if iou(p[2], box) >= 0.5]
        if not hits:
            stat[name]['MISSED'] += 1; continue
        right = [(i,p) for i,p in hits if p[0]==name]
        if not right:
            stat[name]['MISNAMED'] += 1
            conf[name][hits[0][1][0]] += 1
        else:
            stat[name]['FOUND'] += 1
            rank[name].append(right[0][0])

print(f'{"class":16s} {"n":>5s} {"MISSED":>8s} {"MISNAM":>8s} {"FOUND":>8s}  {"px":>5s}  rank@right')
for name in sorted(stat, key=lambda n: -stat[n]['MISSED']/max(1,sum(stat[n].values()))):
    t = sum(stat[name].values())
    r = rank[name]
    print(f'{name:16s} {t:5d} {stat[name]["MISSED"]/t:8.2f} {stat[name]["MISNAMED"]/t:8.2f} '
          f'{stat[name]["FOUND"]/t:8.2f}  {np.mean(size[name]):5.0f}  '
          f'{("med %d p90 %d" % (np.median(r), np.percentile(r,90))) if r else "-"}')
print()
for name in stat:
    if conf[name]:
        print(f'{name:16s} called: ' + ', '.join(f'{k}x{v}' for k,v in conf[name].most_common(4)))
