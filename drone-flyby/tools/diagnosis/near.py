"""Is the box NEAR the object but the wrong shape, or is nothing there at all?

IoU 0.5 conflates two different failures. Relax to centre-distance and to
IoU 0.1 and the answer separates: a box we already emit that only needs
resizing is free score; nothing within a whole object width is a blind detector.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT = Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import score_offline as S
objects = json.loads(S.OBJECTS.read_text())['objects']
motion = S.fit_truth_motion(objects)[0]
f = S.box_factors()
if f: objects = S.loosen(objects, f); motion = S.fit_truth_motion(objects)[0]
preds = S.from_recording(sys.argv[1])
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0
R=collections.defaultdict(collections.Counter); AR=collections.defaultdict(list)
for fr in range(1,250):
    P=preds.get(fr,[])
    for name,box in S.known_boxes(objects,fr,motion):
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        d=np.sqrt((box[2]-box[0])*(box[3]-box[1]))
        same=[p for p in P if p[0]==name]
        best=max((iou(p[2],box) for p in same), default=0)
        # nearest same-class box centre, in object-widths
        nd=min(((abs((p[2][0]+p[2][2])/2-cx)**2+abs((p[2][1]+p[2][3])/2-cy)**2)**.5/max(d,1) for p in same), default=9e9)
        anyd=min(((abs((p[2][0]+p[2][2])/2-cx)**2+abs((p[2][1]+p[2][3])/2-cy)**2)**.5/max(d,1) for p in P), default=9e9)
        R[name]['n']+=1
        if best>=0.5: R[name]['iou50']+=1
        elif best>=0.1: R[name]['iou10']+=1
        elif nd<=1.0: R[name]['near_same']+=1
        elif anyd<=1.0: R[name]['near_any']+=1
        else: R[name]['nothing']+=1
        # aspect ratio of truth vs our same-class boxes when close
        if nd<=1.0 and same:
            b=min(same,key=lambda p:((p[2][0]+p[2][2])/2-cx)**2+((p[2][1]+p[2][3])/2-cy)**2)[2]
            AR[name].append((np.sqrt(max(1,(b[2]-b[0])*(b[3]-b[1])))/max(d,1)))
print(f'{"class":16s} {"n":>4s} {"iou>=.5":>8s} {"iou.1-.5":>9s} {"near":>6s} {"nearOTH":>8s} {"NOTHING":>8s}  sizeratio')
for n in sorted(R,key=lambda n:-R[n]['nothing']/R[n]['n']):
    r=R[n]; t=r['n']; a=AR.get(n)
    print(f'{n:16s} {t:4d} {r["iou50"]/t:8.2f} {r["iou10"]/t:9.2f} {r["near_same"]/t:6.2f} '
          f'{r["near_any"]/t:8.2f} {r["nothing"]/t:8.2f}  {np.median(a):.2f}x' if a is not None and len(a) else
          f'{n:16s} {t:4d} {r["iou50"]/t:8.2f} {r["iou10"]/t:9.2f} {r["near_same"]/t:6.2f} {r["near_any"]/t:8.2f} {r["nothing"]/t:8.2f}   -')
