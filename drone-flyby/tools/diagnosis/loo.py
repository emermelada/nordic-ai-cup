"""Leave-one-run-out: does fitting the geometry on more runs help?

The shipped factors came from three runs, and the two classes that did not move
are the two with the fewest true positives (small_launcher 64, spacecraft 81).
If thin fitting data is the reason, fitting on four runs and testing on the
fifth should beat the three-run fit on the same held-out run.
"""
import sys, json, collections, os, itertools
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
objects0=json.loads(S.OBJECTS.read_text())['objects']
bf=S.box_factors()
objects=S.loosen(list(objects0),bf) if bf else list(objects0)
motion=S.fit_truth_motion(objects)[0]
regions=S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
RUNS=sys.argv[1:]
P={r:S.from_recording(r) for r in RUNS}
ISO=[0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10]
HGT=[1.00,1.10,1.20,1.30,1.40,1.50,1.60]
def ap_grid(runs):
    """mean per-class AP over `runs` for every (iso) and then (iso,height)."""
    g=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        for s in ISO:
            Q=collections.defaultdict(list)
            for fr,v in P[r].items():
                for n,c,b in v: Q[fr].append((n,c,rs(np.asarray(b,float),s,s)))
            _,c2,_=S.score(Q,frames,objects,motion,regions)
            for k,val in c2.items(): g[k][('i',s)].append(val)
    best_i={k:max(ISO,key=lambda s:np.mean(g[k][('i',s)])) for k in g}
    h=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        for m in HGT:
            Q=collections.defaultdict(list)
            for fr,v in P[r].items():
                for n,c,b in v:
                    i=best_i.get(n,1.0); Q[fr].append((n,c,rs(np.asarray(b,float),i,i*m)))
            _,c2,_=S.score(Q,frames,objects,motion,regions)
            for k,val in c2.items(): h[k][m].append(val)
    best_h={k:max(HGT,key=lambda m:np.mean(h[k][m])) for k in h}
    return {k:(best_i.get(k,1.0),best_h.get(k,1.0)) for k in set(best_i)|set(best_h)}
def evaluate(Fmap, run):
    Q=collections.defaultdict(list)
    for fr,v in P[run].items():
        for n,c,b in v:
            i,m=Fmap.get(n,(1.0,1.0)); Q[fr].append((n,c,rs(np.asarray(b,float),i,i*m)))
    o,c2,_=S.score(Q,frames,objects,motion,regions)
    return o,c2
SHIP={'hangar':(1.00,1.00),'helicopter':(0.90,1.60),'jammer':(0.60,1.40),
 'jet_plane':(1.05,1.00),'large_launcher':(0.90,1.20),'large_tower':(0.95,1.00),
 'mine_roller':(0.85,1.50),'small_launcher':(1.00,1.00),'small_plane':(0.80,1.20),
 'small_tower':(0.80,1.10),'spacecraft':(1.00,1.00),'tank':(1.00,1.10)}
a=[];b=[];pa=collections.defaultdict(list);pb=collections.defaultdict(list)
for held in RUNS:
    F4=ap_grid([r for r in RUNS if r!=held])
    o4,c4=evaluate(F4,held); o3,c3=evaluate(SHIP,held)
    a.append(o3); b.append(o4)
    for k in c4: pa[k].append(c3.get(k,0)); pb[k].append(c4[k])
    print(f'held {held[:8]}   shipped {o3:.3f}   refit-on-4 {o4:.3f}   {o4-o3:+.3f}')
print(f'\nmean        shipped {np.mean(a):.3f}   refit-on-4 {np.mean(b):.3f}   {np.mean(b)-np.mean(a):+.3f}')
print('\nper class (leave-one-out mean):')
for k in sorted(pb,key=lambda k:-(np.mean(pb[k])-np.mean(pa[k]))):
    d=np.mean(pb[k])-np.mean(pa[k])
    if abs(d)>0.004: print(f'   {k:16s} {np.mean(pa[k]):.3f} -> {np.mean(pb[k]):.3f}  {d:+.3f}')
