"""Probe: local LLM answers the 10 questions per conversation from cached word-level transcripts.
Usage: llm_probe.py <transcript_dir> <model> [n_conversations]"""
from pathlib import Path
import csv, json, sys, time, re, os, statistics as st
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
CSV=str(Path(__file__).resolve().parents[2] / 'data' / 'question_train.csv')
rows=list(csv.DictReader(open(CSV)))
convs={}
for r in rows: convs.setdefault(r['transcript_id'],[]).append(r)
def iou(a,b):
    i=max(0,min(a[1],b[1])-max(a[0],b[0])); u=max(a[1],b[1])-min(a[0],b[0]); return i/u if u>0 else 0
def split_units(words, pause=0.5, comma_min=6, max_words=18):
    units=[]; cur=[]
    for i,w in enumerate(words):
        if cur:
            prev=words[i-1]; t=prev['word'].strip(); gap=w['start']-prev['end']
            if t.endswith(('.','?','!')) or (t.endswith((',',';',':')) and len(cur)>=comma_min) or gap>=pause or len(cur)>=max_words:
                units.append(cur); cur=[]
        cur.append(w)
    if cur: units.append(cur)
    merged=[]
    for u in units:
        if merged and len(merged[-1])<3 and (u[0]['start']-merged[-1][-1]['end'])<1.0: merged[-1]=merged[-1]+u
        else: merged.append(u)
    return merged
SYSTEM = """You are checking claims against the transcript of a doctor-patient consultation. The transcript is split into numbered units. Answer each yes/no question ONLY from what the transcript says.
Rules:
- Answer "yes" only if the transcript explicitly states or clearly implies it. A near-miss is "no": same drug but a different dose, same test but a different value, same symptom but a different body part, a plan that was only mentioned as a possibility, or the opposite polarity (stable vs unstable, renewed vs stopped).
- Things never discussed are "no".
- Questions phrased as an absence ("free of fever", "no signs of X", "unchanged") are "yes" when the transcript states that absence or unchanged status.
- Tag questions ("..., right?") are ordinary yes/no questions.
- Abbreviations in questions may be spoken in full in the transcript (HbA1c = haemoglobin A1c / long-term sugar; mmol/mol = millimoles per mole; ECG = electrocardiogram; BMI = body mass index; BP = blood pressure).
- For every "yes", give the unit id(s) (at most 2, adjacent) and copy the EXACT words from those units that establish the answer: a contiguous substring, as short as possible while still containing the fact (usually 4-15 words).
Respond with JSON only: {"results":[{"q":1,"answer":"yes","confidence":90,"units":[12],"quote":"..."},{"q":2,"answer":"no","confidence":80,"units":[],"quote":""}, ...]}"""
if __name__ == '__main__':
    OUT, MODEL = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model, tokenizer = load(MODEL)
    sampler = make_sampler(temp=0.0)
    stats={'positive':[0,0],'hard_negative':[0,0],'off_topic':[0,0]}; tious=[]; lat=[]; fails=0; ntok=[]
    for k,(tid, qs) in enumerate(list(convs.items())[:N]):
        d=json.load(open(f"{OUT}/conversation_{tid}.json"))
        words=[w for s in d['segments'] for w in s['words']]
        units=split_units(words)
        utext=[(u[0]['start'],u[-1]['end'],''.join(w['word'] for w in u).strip(),u) for u in units]
        transcript='\n'.join(f'[{i}] {t}' for i,(a,b,t,_) in enumerate(utext))
        qtext='\n'.join(f'{i+1}. {r["question"]}' for i,r in enumerate(qs))
        msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':f'TRANSCRIPT\n{transcript}\n\nQUESTIONS\n{qtext}\n\nJSON:'}]
        try:
            prompt=tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            prompt=tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0=time.time(); out=generate(model, tokenizer, prompt=prompt, max_tokens=900, sampler=sampler, verbose=False); dt=time.time()-t0
        lat.append(dt); ntok.append(len(tokenizer.encode(prompt)))
        m=re.search(r'\{.*\}', out, flags=re.S)
        try: res=json.loads(m.group(0))['results']
        except Exception: fails+=1; res=[]
        by_q={int(x.get('q',0)):x for x in res if isinstance(x,dict)}
        for i,r in enumerate(qs):
            x=by_q.get(i+1,{}); ans = str(x.get('answer','yes')).lower().startswith('y')
            stats[r['question_type']][1]+=1; stats[r['question_type']][0]+= int(ans==(r['label']=='1'))
            if r['label']=='1':
                g=(float(r['evidence_start']),float(r['evidence_end'])); span=None
                if ans and x.get('units'):
                    ids=[u for u in x['units'] if isinstance(u,int) and 0<=u<len(utext)]
                    if ids:
                        span=(utext[min(ids)][0], utext[max(ids)][1])
                        q=(x.get('quote') or '').lower().split()
                        if q:
                            uw=[w for u in ids for w in utext[u][3]]
                            toks=[w['word'].strip().lower().strip('.,;:!?"') for w in uw]
                            qt=[t.strip('.,;:!?"') for t in q]
                            # find first/last quote token in unit words
                            try:
                                i0=next(i for i,t in enumerate(toks) if t==qt[0]); i1=max(i for i,t in enumerate(toks) if t==qt[-1] and i>=i0)
                                span=(uw[i0]['start'], uw[i1]['end'])
                            except StopIteration: pass
                            except ValueError: pass
                tious.append(iou(g,span) if span else 0.0)
        print(f'{k+1}/{N} {tid}: {dt:.1f}s prompt_tokens={ntok[-1]}', flush=True)
    acc=sum(v[0] for v in stats.values())/sum(v[1] for v in stats.values())
    print('model', MODEL); print('accuracy', round(acc,3), {k:f'{v[0]}/{v[1]}' for k,v in stats.items()})
    print('mean tIoU over gold-yes', round(st.mean(tious),3), 'n', len(tious))
    print('score', round(0.4*acc+0.6*st.mean(tious),3))
    print('latency per conversation: mean', round(st.mean(lat),1), 'max', round(max(lat),1), 'json fails', fails, 'prompt tokens mean', int(st.mean(ntok)))
