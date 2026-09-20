"""MISSED / MISNAMED / FOUND per class, against scene_objects.json.

medium_launcher scores 0.000 at every box scale and ta-ta 0.078, yet we emit
thousands of boxes of both. Either we see those objects and call them something
else -- which a class remap fixes with no retraining -- or we do not see them.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
objects=json.loads((ROOT/'training'/'scene_objects.json').read_text())['objects']
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf)
motion=S.fit_truth_motion(objects)[0]
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0
stat=collections.defaultdict(collections.Counter)
conf=collections.defaultdict(collections.Counter)
for run in sys.argv[1:]:
    P=S.from_recording(run)
    for fr in range(1,250):
        v=[(n,c,np.asarray(b,float)) for n,c,b in P.get(fr,[])]
        v.sort(key=lambda p:-p[1])
        for name,box in S.known_boxes(objects,fr,motion):
            hits=[p for p in v if iou(p[2],box)>=0.5]
            stat[name]['n']+=1
            if not hits:
                loose=[p for p in v if iou(p[2],box)>=0.1]
                if loose: stat[name]['LOOSE']+=1; conf[name][loose[0][0]]+=1
                else: stat[name]['MISSED']+=1
            elif any(p[0]==name for p in hits): stat[name]['FOUND']+=1
            else:
                stat[name]['MISNAMED']+=1; conf[name][hits[0][0]]+=1
print(f'{"class":16s} {"n":>5s} {"FOUND":>7s} {"MISNAM":>7s} {"LOOSE":>7s} {"MISSED":>7s}')
for n in sorted(stat,key=lambda n:stat[n]['FOUND']/max(1,stat[n]['n'])):
    r=stat[n]; t=r['n']
    print(f'{n:16s} {t:5d} {r["FOUND"]/t:7.2f} {r["MISNAMED"]/t:7.2f} {r["LOOSE"]/t:7.2f} {r["MISSED"]/t:7.2f}')
print('\nwhat we call them instead:')
for n in sorted(conf):
    if conf[n]: print(f'  {n:16s} ' + ', '.join(f'{k} x{v}' for k,v in conf[n].most_common(5)))
