"""Are the false positives separable from the true ones by size or position?

AP only cares about within-class ranking, so demoting a band of detections that
is nearly all false lifts AP even when the detector is unchanged. A detection
ranked below every real one cannot lower AP, so this is a safe direction.
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
T=collections.defaultdict(list); Fp=collections.defaultdict(list)
for run in sys.argv[1:]:
    P=S.from_recording(run)
    for fr in range(1,250):
        v=[(n,c,rs(np.asarray(b,float),*(lambda t:(t[0],t[0]*t[1]))(F.get(n,(1.0,1.0)))))
           for n,c,b in P.get(fr,[])]
        truth=list(S.known_boxes(objects,fr,motion)); ign=S.ignore_boxes(regions,fr,motion)
        for n,c,b in v:
            s=float(np.sqrt(max(1,(b[2]-b[0])*(b[3]-b[1])))); y=float((b[1]+b[3])/2)
            ar=float((b[2]-b[0])/max(1,(b[3]-b[1])))
            rec=(s,y,ar,c)
            if any(t==n and iou(b,tb)>=0.5 for t,tb in truth): T[n].append(rec)
            elif any(iou(b,ib)>=0.5 for ib in ign): pass
            else: Fp[n].append(rec)
print(f'{"class":16s} {"size TP":>16s} {"size FP":>16s} {"y TP":>12s} {"y FP":>12s} {"ar TP":>7s} {"ar FP":>7s}')
for n in ['small_launcher','spacecraft','tank','jammer','mine_roller']:
    t,f=np.array(T.get(n,[])),np.array(Fp.get(n,[]))
    if not len(t) or not len(f): continue
    q=lambda a,i:(np.percentile(a[:,i],10),np.percentile(a[:,i],90))
    ts,fs,ty,fy=q(t,0),q(f,0),q(t,1),q(f,1)
    print(f'{n:16s} {ts[0]:7.0f}-{ts[1]:<8.0f} {fs[0]:7.0f}-{fs[1]:<8.0f} '
          f'{ty[0]:5.0f}-{ty[1]:<6.0f} {fy[0]:5.0f}-{fy[1]:<6.0f} '
          f'{np.median(t[:,2]):7.2f} {np.median(f[:,2]):7.2f}')
    # what fraction of FPs would a size gate at the TP 5-95 band remove?
    lo,hi=np.percentile(t[:,0],5),np.percentile(t[:,0],95)
    keptF=np.mean((f[:,0]>=lo)&(f[:,0]<=hi)); keptT=np.mean((t[:,0]>=lo)&(t[:,0]<=hi))
    print(f'{"":16s}   size gate [{lo:.0f},{hi:.0f}] keeps {keptT:.0%} of TP, {keptF:.0%} of FP')
