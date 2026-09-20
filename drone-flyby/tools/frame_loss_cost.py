"""Does dropping frames from our answers RAISE the offline score?

If yes, the mechanism is that a skipped frame removes far more false positives
than true positives from the global per-class ranking, which lifts precision
exactly where the real detections sit.
"""
import sys, json, pathlib, collections, random
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
import numpy as np
from score_offline import fit_truth_motion, known_boxes, loosen
from faster_coco_eval import COCO, COCOeval_faster
from dtos import OBJECT_CLASSES

RUN = open('/tmp/run_id.txt').read().strip()
objects = json.loads(open('training/scene_objects.json').read())['objects']
objects = loosen(objects, {c:{'w':1.3,'h':1.3} for c in {o['class'] for o in objects}})
motion,_ = fit_truth_motion(objects)
d = pathlib.Path('data/recordings')/RUN
raw = collections.defaultdict(list)
for p in sorted(d.glob('*.json')):
    m = json.loads(p.read_text())
    for a in (m.get('response') or {}).get('annotations') or []:
        b = np.array(a['bbox'],float)*[3840,2160,3840,2160]
        raw[m['frame']].append((a['object_id'], a['confidence'], b))

frames=list(range(1,250)); cats={n:i+1 for i,n in enumerate(OBJECT_CLASSES)}
imgs={f:i+1 for i,f in enumerate(frames)}
anns=[];aid=1;present=collections.Counter()
for f in frames:
    for cls,b in known_boxes(objects,f,motion):
        anns.append({'id':aid,'image_id':imgs[f],'category_id':cats[cls],
            'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'area':(b[2]-b[0])*(b[3]-b[1]),'iscrowd':0})
        aid+=1; present[cls]+=1
gtd={'info':{},'licenses':[],'images':[{'id':v,'file_name':f'{k}.png','width':3840,'height':2160} for k,v in imgs.items()],
     'categories':[{'id':v,'name':k,'supercategory':'o'} for k,v in cats.items()],'annotations':anns}
GT=COCO(gtd); KL=[c for c in OBJECT_CLASSES if c in present]

def score(drop):
    dts=[{'image_id':imgs[f],'category_id':cats[n],'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'score':c}
         for f in frames if f not in drop for n,c,b in raw.get(f,[]) if b[2]>b[0] and b[3]>b[1]]
    if not dts: return 0.0
    dt=GT.loadRes(dts); ev=COCOeval_faster(GT,dt,'bbox')
    ev.params.imgIds=list(imgs.values()); ev.params.catIds=[cats[c] for c in KL]
    ev.params.iouThrs=np.array([0.5]); ev.evaluate(); ev.accumulate()
    P=ev.eval['precision']; tot=0
    for i,c in enumerate(KL):
        cp=P[0,:,i,0,-1]; v=cp[cp>-1]; tot+=float(np.mean(v)) if v.size else 0.0
    return tot/len(KL)

base=score(set())
print(f'  all 249 frames answered      {base:.5f}')
rng=random.Random(0)
for k in (1,2,3,5,10,20,40):
    vals=[score(set(rng.sample(frames,k))) for _ in range(12)]
    print(f'  drop {k:3d} random frames      {np.mean(vals):.5f}   ({np.mean(vals)-base:+.5f})  min {min(vals):.4f} max {max(vals):.4f}')
