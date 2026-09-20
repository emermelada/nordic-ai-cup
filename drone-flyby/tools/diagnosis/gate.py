"""Demote (never drop) detections whose size falls outside a class's band.

A detection ranked strictly below every real one cannot lower AP -- verified
against faster_coco_eval earlier in this project -- so demoting is safe where
dropping would risk losing a real object. Bands are fitted on FIT runs and
scored on TEST runs, which must be disjoint.
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
regions=S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
def fixed(run):
    P=S.from_recording(run); Q=collections.defaultdict(list)
    for fr,v in P.items():
        for n,c,b in v:
            i,h=F.get(n,(1.0,1.0)); Q[fr].append((n,c,rs(np.asarray(b,float),i,i*h)))
    return Q
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0
FIT=sys.argv[1].split(','); TEST=sys.argv[2].split(',')
# --- fit the size band on FIT runs only, from TRUE positives ---
T=collections.defaultdict(list)
for run in FIT:
    Q=fixed(run)
    for fr in frames:
        truth=list(S.known_boxes(objects,fr,motion))
        for n,c,b in Q.get(fr,[]):
            if any(t==n and iou(b,tb)>=0.5 for t,tb in truth):
                T[n].append(np.sqrt(max(1,(b[2]-b[0])*(b[3]-b[1]))))
BAND={n:(np.percentile(v,3),np.percentile(v,97)) for n,v in T.items() if len(v)>=20}
print('fitted size bands (px, from TPs on fit runs):')
for n,(lo,hi) in sorted(BAND.items()): print(f'   {n:16s} {lo:5.0f} - {hi:<5.0f}  (n={len(T[n])})')
# --- score on TEST runs ---
for DEMOTE in (1.0, 0.25, 0.05, 0.0):
    tot0=[];tot1=[];per=collections.defaultdict(list)
    for run in TEST:
        Q=fixed(run)
        o0,c0,_=S.score(Q,frames,objects,motion,regions)
        R=collections.defaultdict(list)
        for fr,v in Q.items():
            for n,c,b in v:
                s=np.sqrt(max(1,(b[2]-b[0])*(b[3]-b[1]))); lo,hi=BAND.get(n,(0,1e9))
                R[fr].append((n, c if lo<=s<=hi else c*DEMOTE, b))
        o1,c1,_=S.score(R,frames,objects,motion,regions)
        tot0.append(o0); tot1.append(o1)
        for k in c1: per[k].append(c1[k]-c0.get(k,0))
    print(f'\ndemote x{DEMOTE}:  {np.mean(tot0):.3f} -> {np.mean(tot1):.3f}  ({np.mean(tot1)-np.mean(tot0):+.3f})')
    for k in sorted(per,key=lambda k:-np.mean(per[k])):
        if abs(np.mean(per[k]))>0.005: print(f'      {k:16s} {np.mean(per[k]):+.3f}')
