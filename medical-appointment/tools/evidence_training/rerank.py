"""Score both systems' spans under the trained extractor, per held-out fold.

Emits one row per question with the extractor's own best span and the margin it
assigns to an externally supplied span, so a chooser can be evaluated offline
without another GPU pass. Selection rules are not applied here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from .data import best_word_span, validate_bundle, write_json
from .features import collate, decode_window, make_features


def span_margin(features, logits, index, words):
    """Best CLS-normalized score for one exact word span, maximized over its windows."""
    a, b = words
    best = None
    for feature, (start_logits, end_logits) in zip(features, logits):
        if feature['record_index'] != index:
            continue
        starts = [i for i, (m, w) in enumerate(zip(feature['start_mask'], feature['first_word']))
                  if m and w == a]
        ends = [i for i, (m, w) in enumerate(zip(feature['end_mask'], feature['last_word'])) if m and w == b]
        if not starts or not ends or starts[0] > ends[-1]:
            continue
        cls = feature['cls']
        margin = float(start_logits[starts[0]] + end_logits[ends[-1]]
                       - start_logits[cls] - end_logits[cls])
        best = margin if best is None else max(best, margin)
    return best


def run_fold(model, tokenizer, records, args, device):
    features = make_features(records, tokenizer, args.max_length, args.stride)
    logits = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(features), args.batch_size):
            current = features[offset:offset + args.batch_size]
            batch = collate(current, tokenizer, device)
            with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda'
                  else torch.autocast(device_type='cpu', enabled=False)):
                output = model(**batch['inputs'])
            logits.extend(zip(output.start_logits.float().cpu().numpy(),
                              output.end_logits.float().cpu().numpy()))
    choices = {}
    for feature, (start_logits, end_logits) in zip(features, logits):
        candidate = decode_window(feature, start_logits, end_logits, args.max_span_words)
        index = feature['record_index']
        if candidate and (index not in choices or candidate['margin'] > choices[index]['margin']):
            choices[index] = candidate
    rows = []
    for index, record in enumerate(records):
        choice = choices.get(index)
        # The other system's span, snapped to the word grid it has to live on anyway.
        # A baseline NO carries no span, so there is nothing to score against.
        control_words, control_quality = (best_word_span(record['words'], record['baseline_span'])
                                          if record['baseline_span'] else (None, None))
        row = {'id': record['id'], 'conversation': record['conversation'], 'label': record['label'],
               'answer': record['baseline_answer'], 'gold': record['gold'],
               'control_span': record['baseline_span'], 'control_words': control_words,
               'control_snap_tiou': control_quality,
               'control_margin': (span_margin(features, logits, index, control_words)
                                  if control_words else None),
               'extractor_words': choice['words'] if choice else None,
               'extractor_margin': choice['margin'] if choice else None}
        if choice:
            start, end = choice['words']
            row['extractor_span'] = [record['words'][start]['start'], record['words'][end]['end']]
        else:
            row['extractor_span'] = None
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True,
                        help='Cross-validation run directory holding fold_N/best')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=192)
    parser.add_argument('--max-span-words', type=int, default=128)
    args = parser.parse_args(argv)
    bundle = json.loads(args.data.read_text())
    validate_bundle(bundle)
    records = {r['id']: r for r in bundle['records']}
    rows = []
    for plan in bundle['folds']:
        path = args.checkpoints / f'fold_{plan["fold"]}' / 'best'
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=False)
        tokenizer.padding_side = 'right'
        model = AutoModelForQuestionAnswering.from_pretrained(path, trust_remote_code=False).to(args.device)
        held = [r for r in bundle['records'] if r['conversation'] in set(plan['test'])]
        fold_rows = run_fold(model, tokenizer, held, args, args.device)
        for row in fold_rows:
            row['fold'] = plan['fold']
        rows.extend(fold_rows)
        print(json.dumps({'fold': plan['fold'], 'questions': len(fold_rows),
                          'scored_control_spans': sum(r['control_margin'] is not None for r in fold_rows)}),
              flush=True)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if {r['id'] for r in rows} != set(records):
        raise ValueError('Fold test sets do not cover every question exactly once')
    write_json(args.output, {'scope': 'Candidate scores only; no selection rule applied',
                             'checkpoints': str(args.checkpoints), 'rows': rows})
    print(json.dumps({'questions': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
