"""Why does a class with decent recall still score a low AP?

AP punishes false positives ranked ABOVE true ones. small_launcher is found on
0.73 of its in-view frames yet scores 0.218, which recall alone cannot explain.
Compare the confidence our true detections carry against the confidence of our
false ones, per class, with the box fix already applied.
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
TP=collections.defaultdict(list); FP=collections.defaultdict(list); NT=collections.Counter()
for run in sys.argv[1:]:
    P=S.from_recording(run)
    for fr in range(1,250):
        v=[(n,c,rs(np.asarray(b,float),*(lambda t:(t[0],t[0]*t[1]))(F.get(n,(1.0,1.0)))))
           for n,c,b in P.get(fr,[])]
        truth=list(S.known_boxes(objects,fr,motion))
        ign=S.ignore_boxes(regions,fr,motion)
        for n,c,b in v:
            hit=any(t==n and iou(b,tb)>=0.5 for t,tb in truth)
            if hit: TP[n].append(c)
            elif any(iou(b,ib)>=0.5 for ib in ign): pass     # unlabelled real object
            else: FP[n].append(c)
        for t,_ in truth: NT[t]+=1
print(f'{"class":16s} {"nTP":>6s} {"nFP":>7s} {"TPconf":>7s} {"FPconf":>7s} {"FP>medTP":>9s}  verdict')
for n in sorted(set(TP)|set(FP)):
    tp,fp=TP.get(n,[]),FP.get(n,[])
    if not tp: print(f'{n:16s} {0:6d} {len(fp):7d} {"-":>7s} {np.median(fp):7.3f} {"-":>9s}  no true positives'); continue
    mt=np.median(tp)
    above=sum(1 for c in fp if c>mt)/max(1,len(tp))
    v=('ranking: FPs outrank TPs' if above>3 else
       'recall-bound' if len(tp)<0.5*NT[n] else 'ok')
    print(f'{n:16s} {len(tp):6d} {len(fp):7d} {mt:7.3f} {np.median(fp):7.3f} {above:9.1f}  {v}')
