"""Find the geometry config that is positive under BOTH truth files.

validation_objects.json and scene_objects.json disagree sharply about jammer
(our AP is 0.013 under one and 0.848 under the other), so a factor fitted on
either alone is unsafe. Only keep a per-class factor when it does not hurt
under either truth -- that is the config that does not depend on which mining
run is right.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
FULL={'hangar':(1.00,1.00),'helicopter':(0.90,1.60),'jammer':(0.60,1.40),
 'jet_plane':(1.05,1.00),'large_launcher':(0.90,1.20),'large_tower':(0.95,1.00),
 'mine_roller':(0.85,1.50),'small_launcher':(1.00,1.00),'small_plane':(0.80,1.20),
 'small_tower':(0.80,1.10),'spacecraft':(1.00,1.00),'tank':(1.00,1.10)}
runs=sys.argv[1:]
frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
P={r:S.from_recording(r) for r in runs}
def setup(which):
    o=json.loads((ROOT/'training'/which).read_text())['objects']
    bf=S.box_factors()
    if bf: o=S.loosen(o,bf)
    return o, S.fit_truth_motion(o)[0], S.ignore_regions(('confirmed',))
def run(Fmap, which):
    objects,motion,regions=setup(which)
    tot=[];per=collections.defaultdict(list)
    for r in runs:
        Q=collections.defaultdict(list)
        for fr,v in P[r].items():
            for n,c,b in v:
                i,h=Fmap.get(n,(1.0,1.0)); Q[fr].append((n,c,rs(np.asarray(b,float),i,i*h)))
        o,c2,_=S.score(Q,frames,objects,motion,regions)
        tot.append(o)
        for k,val in c2.items(): per[k].append(val)
    return np.mean(tot), {k:np.mean(v) for k,v in per.items()}
NONE={}
print('per-class effect of each factor, under both truths (5 runs)')
print(f'{"class":16s} {"validation":>22s} {"scene":>22s}   keep?')
keep={}
for cls,fac in sorted(FULL.items()):
    if fac==(1.0,1.0): continue
    one={cls:fac}
    line=[]
    ok=True
    for which in ('validation_objects.json','scene_objects.json'):
        b0,_=run(NONE,which); b1,_=run(one,which)
        d=b1-b0; line.append(f'{b0:.3f}->{b1:.3f} {d:+.3f}')
        if d < -0.002: ok=False
    keep[cls]=fac if ok else None
    print(f'{cls:16s} {line[0]:>22s} {line[1]:>22s}   {"KEEP" if ok else "drop"}')
SAFE={k:v for k,v in keep.items() if v}
print('\nsafe config:', {k:(round(1.3*v[0],2), round(1.3*v[0]*v[1],2)) for k,v in SAFE.items()})
for which in ('validation_objects.json','scene_objects.json'):
    b0,_=run(NONE,which); bf,_=run(FULL,which); bs,_=run(SAFE,which)
    print(f'{which:26s} current {b0:.3f}   full {bf:.3f} ({bf-b0:+.3f})   safe {bs:.3f} ({bs-b0:+.3f})')
