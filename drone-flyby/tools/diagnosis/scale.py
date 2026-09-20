"""For each class, what rescale of OUR boxes maximises the IoU>=0.5 hit rate?

We currently apply BOX_GROW=1.3 to every class. The healthy classes sit near
1.0x of truth and the dead ones at 1.24-1.42x, so a single global factor is
trading the small classes away for the large ones.
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
def rescale(b,s):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*s/2,(b[3]-b[1])*s/2
    return [cx-w,cy-h,cx+w,cy+h]
pairs=collections.defaultdict(list)
for fr in range(1,250):
    P=preds.get(fr,[])
    for name,box in S.known_boxes(objects,fr,motion):
        same=[p for p in P if p[0]==name]
        if not same: pairs[name].append(None); continue
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        b=min(same,key=lambda p:((p[2][0]+p[2][2])/2-cx)**2+((p[2][1]+p[2][3])/2-cy)**2)[2]
        pairs[name].append((list(b),list(box)))
S_=[0.60,0.70,0.75,0.80,0.85,0.90,1.00,1.10,1.30]
print(f'{"class":16s} ' + ' '.join(f'{s:5.2f}' for s in S_) + '   best  now(1.00)')
tot={s:0.0 for s in S_}
for n in sorted(pairs):
    v=pairs[n]; t=len(v)
    row=[]
    for s in S_:
        hit=sum(1 for x in v if x and iou(rescale(x[0],s),x[1])>=0.5)/t
        row.append(hit); tot[s]+=hit
    best=S_[int(np.argmax(row))]
    print(f'{n:16s} ' + ' '.join(f'{r:5.2f}' for r in row) + f'   {best:.2f}  {row[S_.index(1.00)]:.2f}')
print(f'\n{"MACRO":16s} ' + ' '.join(f'{tot[s]/len(pairs):5.2f}' for s in S_))
