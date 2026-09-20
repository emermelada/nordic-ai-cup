"""For every truth object-frame with NOTHING near it: was it even in view?

Separates two failures that need opposite fixes:
  OUT OF VIEW  -> camera coverage; the detector never got a chance
  IN VIEW      -> detector blind on that class at that scale
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
objects=json.loads(S.OBJECTS.read_text())['objects']
motion=S.fit_truth_motion(objects)[0]
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf); motion=S.fit_truth_motion(objects)[0]
run=sys.argv[1]
views={}
for p in sorted((Path('data/recordings')/run).glob('*.json')):
    d=json.loads(p.read_text())
    views[d['frame']]=(d['view']['source_region_xyxy'], d['view']['resolution_level'])
_raw=S.from_recording(run)
F={'hangar':(1.00,1.00),'helicopter':(0.90,1.60),'jammer':(0.60,1.40),
   'jet_plane':(1.05,1.00),'large_launcher':(0.90,1.20),'large_tower':(0.95,1.00),
   'mine_roller':(0.85,1.50),'small_launcher':(1.00,1.00),'small_plane':(0.80,1.20),
   'small_tower':(0.80,1.10),'spacecraft':(1.00,1.00),'tank':(1.00,1.10)}
def _rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
preds=collections.defaultdict(list)
for _f,_v in _raw.items():
    for _n,_c,_b in _v:
        _i,_h=F.get(_n,(1.0,1.0)); preds[_f].append((_n,_c,_rs(np.asarray(_b,float),_i,_i*_h)))
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0
R=collections.defaultdict(collections.Counter)
seen=collections.defaultdict(collections.Counter)
for fr in range(1,250):
    if fr not in views: continue
    reg,lvl=views[fr]
    P=preds.get(fr,[])
    for name,box in S.known_boxes(objects,fr,motion):
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        inview = reg[0]<=cx<=reg[2] and reg[1]<=cy<=reg[3]
        best=max((iou(p[2],box) for p in P if p[0]==name), default=0)
        near=any(((p[2][0]+p[2][2])/2-cx)**2+((p[2][1]+p[2][3])/2-cy)**2 <
                 (np.sqrt((box[2]-box[0])*(box[3]-box[1])))**2 for p in P)
        R[name]['n']+=1
        if inview:
            R[name]['inview']+=1
            seen[name][f'L{lvl}']+=1
            if best>=0.5: R[name]['inview_hit']+=1
            elif near: R[name]['inview_near']+=1
            else: R[name]['inview_BLIND']+=1
        else:
            R[name]['out']+=1
            if best>=0.5: R[name]['out_hit']+=1   # memory carried it
print(f'{"class":16s} {"n":>4s} {"inview":>7s} {"hit":>6s} {"near":>6s} {"BLIND":>6s} | {"out":>5s} {"carried":>7s}')
for n in sorted(R,key=lambda n:-R[n]['inview_BLIND']/max(1,R[n]['inview'])):
    r=R[n]; iv=max(1,r['inview'])
    print(f'{n:16s} {r["n"]:4d} {r["inview"]/r["n"]:7.2f} {r["inview_hit"]/iv:6.2f} '
          f'{r["inview_near"]/iv:6.2f} {r["inview_BLIND"]/iv:6.2f} | {r["out"]/r["n"]:5.2f} '
          f'{r["out_hit"]/max(1,r["out"]):7.2f}')
print('\nviews each class was seen in:')
for n in sorted(seen): print(f'  {n:16s} ' + ' '.join(f'{k}={v}' for k,v in sorted(seen[n].items())))
