"""How many of our false positives are DUPLICATES of an object we already got?

COCO matches one detection per truth box; every further box on the same object
is a false positive, and because it sits near the matched one in confidence it
lands high in the ranking -- the most damaging kind of FP there is. Unlike a
size band, merging duplicates needs no fitted threshold.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
F={'hangar':(1.00,1.00),'helicopter':(0.90,1.60),'jammer':(0.60,1.40),
   'jet_plane':(1.05,1.00),'large_launcher':(0.90,1.20),'large_tower':(0.95,1.00),
   'mine_roller':(0.85,1.50),'small_launcher':(1.00,1.00),'small_plane':(0.80,1.20),
   'small_tower':(0.80,1.10),'spacecraft':(1.00,1.00),'tank':(1.00,1.10)}
objects=json.loads(S.OBJECTS.read_text())['objects']
motion=S.fit_truth_motion(objects)[0]
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf); motion=S.fit_truth_motion(objects)[0]
regions=S.ignore_regions(('confirmed',))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0
R=collections.defaultdict(collections.Counter)
for run in sys.argv[1:]:
    P=S.from_recording(run)
    for fr in range(1,250):
        v=[(n,c,rs(np.asarray(b,float),*(lambda t:(t[0],t[0]*t[1]))(F.get(n,(1.0,1.0)))))
           for n,c,b in P.get(fr,[])]
        v.sort(key=lambda p:-p[1])
        truth=list(S.known_boxes(objects,fr,motion)); ign=S.ignore_boxes(regions,fr,motion)
        used=set()
        for n,c,b in v:
            m=None
            for i,(t,tb) in enumerate(truth):
                if t==n and i not in used and iou(b,tb)>=0.5: m=i; break
            if m is not None: used.add(m); R[n]['TP']+=1; continue
            if any(iou(b,ib)>=0.5 for ib in ign): R[n]['ignored']+=1; continue
            # is it a duplicate: overlaps a truth box of its own class already taken?
            if any(t==n and iou(b,tb)>=0.5 for t,tb in truth): R[n]['DUP']+=1
            elif any(t==n and iou(b,tb)>=0.1 for t,tb in truth): R[n]['nearmiss']+=1
            else: R[n]['bg']+=1
print(f'{"class":16s} {"TP":>6s} {"DUP":>6s} {"nearmiss":>9s} {"bg":>7s}  dup/TP')
for n in sorted(R,key=lambda n:-R[n]['DUP']/max(1,R[n]['TP'])):
    r=R[n]
    if not r['TP']: continue
    print(f'{n:16s} {r["TP"]:6d} {r["DUP"]:6d} {r["nearmiss"]:9d} {r["bg"]:7d}  {r["DUP"]/r["TP"]:.2f}')
