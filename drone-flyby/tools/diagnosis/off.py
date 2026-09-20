"""Is our box systematically OFFSET from truth, not just mis-sized?

A detector that locks onto the visible superstructure of an object while the
truth box covers its full footprint produces a constant centre offset. That is
invisible to any size correction and costs exactly the near-miss band.
Offsets are measured in units of the object's own size, after the fitted
size correction, so they are directly shippable as a box shift.
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
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
D=collections.defaultdict(list)
for run in sys.argv[1:]:
    P=S.from_recording(run)
    for fr,v in P.items():
        v=[(n,c,rs(np.asarray(b,float),*(lambda t:(t[0],t[0]*t[1]))(F.get(n,(1.0,1.0))))) for n,c,b in v]
        for name,box in S.known_boxes(objects,fr,motion):
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            w,h=box[2]-box[0],box[3]-box[1]
            same=[p for p in v if p[0]==name]
            if not same: continue
            b=min(same,key=lambda p:((p[2][0]+p[2][2])/2-cx)**2+((p[2][1]+p[2][3])/2-cy)**2)[2]
            bx,by=(b[0]+b[2])/2,(b[1]+b[3])/2
            d=np.hypot(bx-cx,by-cy)
            if d < 1.5*np.sqrt(w*h):          # a plausible match, not another object
                D[name].append(((bx-cx)/w, (by-cy)/h))
print(f'{"class":16s} {"n":>5s} {"dx/w":>7s} {"dy/h":>7s}   (median, our centre minus truth)')
for n in sorted(D,key=lambda n:-abs(np.median([y for _,y in D[n]]))):
    a=np.array(D[n])
    print(f'{n:16s} {len(a):5d} {np.median(a[:,0]):7.3f} {np.median(a[:,1]):7.3f}')
