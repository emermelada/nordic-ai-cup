"""Run the LLM once per conversation and dump raw outputs + unit tables for offline alignment experiments."""
from pathlib import Path
import csv, json, sys, time, os
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_probe import SYSTEM, split_units   # reuse prompt + splitter (module-level code guarded below)
OUT, MODEL, DUMP = sys.argv[1], sys.argv[2], sys.argv[3]
CSV=str(Path(__file__).resolve().parents[2] / 'data' / 'question_train.csv')
rows=list(csv.DictReader(open(CSV))); convs={}
for r in rows: convs.setdefault(r['transcript_id'],[]).append(r)
model, tokenizer = load(MODEL); sampler=make_sampler(temp=0.0)
dump={}
for k,(tid,qs) in enumerate(convs.items()):
    d=json.load(open(f"{OUT}/conversation_{tid}.json")); words=[w for s in d['segments'] for w in s['words']]
    units=split_units(words)
    utext=[{'start':u[0]['start'],'end':u[-1]['end'],'text':''.join(w['word'] for w in u).strip(),'words':u} for u in units]
    transcript='\n'.join(f"[{i}] {u['text']}" for i,u in enumerate(utext))
    qtext='\n'.join(f'{i+1}. {r["question"]}' for i,r in enumerate(qs))
    msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':f'TRANSCRIPT\n{transcript}\n\nQUESTIONS\n{qtext}\n\nJSON:'}]
    prompt=tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    t0=time.time(); out=generate(model, tokenizer, prompt=prompt, max_tokens=900, sampler=sampler, verbose=False); dt=time.time()-t0
    dump[tid]={'raw':out,'seconds':dt,'units':utext,'questions':[{'id':r['question_id'],'q':r['question'],'type':r['question_type'],'label':r['label'],'gs':r['evidence_start'],'ge':r['evidence_end']} for r in qs]}
    print(f'{k+1}/39 {tid} {dt:.1f}s', flush=True)
json.dump(dump, open(DUMP,'w'))
print('DONE')
