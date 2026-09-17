"""No-LLM baseline: BM25 over clause units + number/entity mismatch rule. Reports accuracy by type and tIoU."""
from pathlib import Path
import csv, json, sys, re, statistics as st
from rank_bm25 import BM25Okapi
OUT=sys.argv[1]
CSV=str(Path(__file__).resolve().parents[2] / 'data' / 'question_train.csv')
rows=list(csv.DictReader(open(CSV)))
SYN={'hba1c':'hemoglobin a1c long-term sugar','ecg':'electrocardiogram','bmi':'body mass index','tsh':'thyroid stimulating hormone','bp':'blood pressure','mmol/mol':'millimoles per mole','mmol/l':'millimoles per liter','mg':'mg milligrams','iu':'international units','nsaid':'anti-inflammatory','gp':'doctor'}
STOP=set('the a an is was were be been are did does do has have had of to in on for at with by and or that this it its as from about any there their patient doctor be will should would can could right correct didn t wasn isn'.split())
def norm(t):
    t=t.lower().replace('/',' / ')
    for k,v in SYN.items(): t=re.sub(r'\b'+re.escape(k)+r'\b', v, t)
    return re.findall(r"[a-z0-9.]+", t)
def nums(tokens): return set(x for x in tokens if re.fullmatch(r'\d+(\.\d+)?', x))
def iou(a,b):
    i=max(0,min(a[1],b[1])-max(a[0],b[0])); u=max(a[1],b[1])-min(a[0],b[0]); return i/u if u>0 else 0
def split_units(words, pause=0.5, comma_min=6, max_words=18):
    units=[]; cur=[]
    for i,w in enumerate(words):
        if cur:
            prev=words[i-1]; t=prev['word'].strip(); gap=w['start']-prev['end']
            if t.endswith(('.','?','!')) or (t.endswith((',',';',':')) and len(cur)>=comma_min) or gap>=pause or len(cur)>=max_words: units.append(cur); cur=[]
        cur.append(w)
    if cur: units.append(cur)
    return units
convs={}
for r in rows: convs.setdefault(r['transcript_id'],[]).append(r)
for tau in [2,3,4,5]:
    stats={'positive':[0,0],'hard_negative':[0,0],'off_topic':[0,0]}; tious=[]
    for tid,qs in convs.items():
        d=json.load(open(f'{OUT}/conversation_{tid}.json')); words=[w for s in d['segments'] for w in s['words']]
        units=split_units(words); texts=[''.join(w['word'] for w in u) for u in units]
        toks=[[t for t in norm(x) if t not in STOP] for x in texts]
        bm=BM25Okapi(toks)
        for r in qs:
            q=[t for t in norm(r['question']) if t not in STOP]
            sc=bm.get_scores(q); best=int(max(range(len(sc)), key=lambda i: sc[i])); s=sc[best]
            # number rule: question number must appear in the top-3 units' window, else no
            qn=nums(norm(r['question'])); ans = s>tau
            if qn:
                top=sorted(range(len(sc)), key=lambda i:-sc[i])[:3]
                win=set().union(*[nums(toks[i]) for i in top])
                if not qn<=win: ans=False
            stats[r['question_type']][1]+=1; stats[r['question_type']][0]+=int(ans==(r['label']=='1'))
            if r['label']=='1':
                g=(float(r['evidence_start']),float(r['evidence_end']))
                tious.append(iou(g,(units[best][0]['start'],units[best][-1]['end'])) if ans else 0.0)
    acc=sum(v[0] for v in stats.values())/390
    print(f'tau={tau}: acc {acc:.3f} ' + ' '.join(f'{k}={v[0]}/{v[1]}' for k,v in stats.items()) + f'  mean tIoU {st.mean(tious):.3f}  score {0.4*acc+0.6*st.mean(tious):.3f}')
