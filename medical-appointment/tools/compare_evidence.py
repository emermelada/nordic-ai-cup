"""Compare evidence prompts on a fixed conversation split using cached ASR."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import _answer, _question_id, _results, build_messages
from pipeline.evidence import build_focused_messages
from tools.probes.evidence_variants import build_grounded_messages, build_refinement_messages
from pipeline.mlx_backend import LLM_MODEL, MLXBackend
from tools.eval_offline import _finite_seconds, evaluate, prepare_inputs, split_conversations
from tools.transcribe_all import atomic_write_json
from utils import group_questions_by_conversation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', type=Path, default=Path('transcripts/probe-turbo'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/reference/llm_dump_qwen8b.json'))
    parser.add_argument('--subset', choices=['dev', 'holdout', 'all'], default='dev')
    parser.add_argument('--prompt', choices=['legacy', 'focused', 'refine', 'grounded'], default='focused')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    conversations = split_conversations(group_questions_by_conversation(), args.subset)
    baseline = json.loads(args.baseline.read_text())
    prepared = prepare_inputs(conversations, args.transcripts, baseline)
    builder = {'legacy': build_messages, 'focused': build_focused_messages,
               'grounded': build_grounded_messages, 'refine': build_messages}[args.prompt]
    prompts = [
        build_refinement_messages(item['words'], [row['question'] for row in item['rows']], item['entry']['raw'])
        if args.prompt == 'refine' else builder(item['words'], [row['question'] for row in item['rows']])
        for item in prepared
    ]
    manifest = {
        'subset': args.subset, 'prompt': args.prompt,
        'conversations': [item['id'] for item in prepared],
        'prompt_sha256': hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
        'system': prompts[0][0]['content'],
    }
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing experiment has different inputs or prompt')
    else:
        atomic_write_json(manifest_path, manifest)
    backend = MLXBackend()
    for index, (item, messages) in enumerate(zip(prepared, prompts), 1):
        destination = args.output / 'generations' / f'{item["id"]}.json'
        if destination.exists():
            generation = json.loads(destination.read_text())
        else:
            raw = backend.generate_messages(messages, max_tokens=1200 if args.prompt == 'grounded' else 900)
            generation = {'raw': raw, 'seconds': backend.last_generation_seconds}
            atomic_write_json(destination, generation)
        if args.prompt == 'refine':
            count = len(item['rows'])
            revisions = {
                _question_id(entry.get('q'), count): entry for entry in _results(generation['raw'])
                if isinstance(entry, dict)
            }
            entries = _results(item['entry']['raw'])
            for entry in entries:
                if not isinstance(entry, dict) or _answer(entry.get('answer')) is not True:
                    continue
                revision = revisions.get(_question_id(entry.get('q'), count), {})
                if isinstance(revision.get('quote'), str) and revision['quote'].strip():
                    entry['quote'] = revision['quote']
                    entry['units'] = revision.get('units', [])
            first_seconds = _finite_seconds(item['entry'].get('seconds'))
            second_seconds = _finite_seconds(generation.get('seconds'))
            total_seconds = (first_seconds + second_seconds
                             if first_seconds is not None and second_seconds is not None else None)
            generation = {**generation, 'raw': json.dumps({'results': entries}), 'seconds': total_seconds}
        item['entry'] = {**item['entry'], **generation, 'prompt': args.prompt,
                         'alignment': 'legacy', 'model': LLM_MODEL}
        seconds = _finite_seconds(generation.get('seconds'))
        timing = f'{seconds:.2f}s' if seconds is not None else 'unknown time'
        print(f'{index}/{len(prepared)} {item["id"]}: {timing} generation', flush=True)
    evaluate(prepared, replay=True, retrieval_only=False, start_offset=0,
             output=args.output / 'scores', subset=args.subset)


if __name__ == '__main__':
    main()
