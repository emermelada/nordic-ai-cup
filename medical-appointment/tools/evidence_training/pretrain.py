"""Intermediate rationale training on general and medical text, before medical adaptation."""

import gc
import math
import random
import time

import numpy as np
import torch

from .data import write_json
from .external import hydrate
from .features import collate, decode_window

# SIMORD's training split stays out: it is the only medical dialogue data, and it is spent
# adapting each competition fold instead.
PRETRAIN_SOURCES = ('coqa', 'mashqa')


def word_iou(gold, prediction):
    if not gold or not prediction:
        return 0.0
    intersection = max(0, min(gold[1], prediction[1]) - max(gold[0], prediction[0]) + 1)
    union = max(gold[1], prediction[1]) - min(gold[0], prediction[0]) + 1
    return intersection / union


def evaluate(runner, model, records, features):
    """Evaluate all gold-positive evidence spans; no fake seconds or competition score."""
    model.eval()
    choices = {}
    with torch.inference_mode():
        for offset in range(0, len(features), runner.args.eval_batch_size):
            runner.check_budget()
            current = features[offset:offset + runner.args.eval_batch_size]
            batch = collate(current, runner.tokenizer, runner.device)
            with runner.autocast():
                result = model(**batch['inputs'])
            starts, ends = result.start_logits.float().cpu().numpy(), result.end_logits.float().cpu().numpy()
            for feature, a, b in zip(current, starts, ends):
                candidate = decode_window(feature, a, b, runner.args.max_span_words)
                index = feature['record_index']
                if candidate and (index not in choices or candidate['margin'] > choices[index]['margin']):
                    choices[index] = candidate
    rows = [{'id': r['id'], 'source': r['source'], 'label': r['label'], 'gold_words': r['gold_words'],
             'predicted_words': choices.get(i, {}).get('words'),
             'word_iou': word_iou(r['gold_words'], choices.get(i, {}).get('words')) if r['label'] else None}
            for i, r in enumerate(records)]
    positives = [r for r in rows if r['label']]
    if not positives:
        raise ValueError('No positive external development examples')
    # Macro over sources so the larger general corpus cannot hide a collapse on the medical one.
    by_source = {source: {'positives': sum(r['source'] == source for r in positives),
                          'positive_word_iou': float(np.mean([r['word_iou'] for r in positives
                                                              if r['source'] == source]))}
                 for source in sorted({r['source'] for r in positives})}
    return {'metric': 'word_span_iou_not_temporal',
            'selection_metric': 'unweighted mean of per-source positive word IoU',
            'questions': len(rows), 'positives': len(positives),
            'positive_word_iou': float(np.mean([r['word_iou'] for r in positives])),
            'macro_word_iou': float(np.mean([v['positive_word_iou'] for v in by_source.values()])),
            'by_source': by_source, 'rows': rows}


def pretrain(runner):
    args = runner.args
    groups = {split: [r for r in hydrate(runner.external, split) if r['source'] in PRETRAIN_SOURCES]
              for split in ('train', 'dev')}
    if not all(groups.values()):
        raise ValueError(f'Training and development examples are required from {PRETRAIN_SOURCES}')
    sources = {split: sorted({r['source'] for r in rows}) for split, rows in groups.items()}
    runner.save_status('preprocessing', scope='External text rationale pretraining; no competition score',
                       answers_frozen=False, external_sources=sources,
                       selection='External development macro positive word IoU only')
    features = {split: runner.features(records) for split, records in groups.items()}
    runner.check_budget()
    write_json(runner.out / 'split.json', {'questions': {s: len(r) for s, r in groups.items()},
               'windows': {s: len(f) for s, f in features.items()}, 'sources': sources,
               'questions_by_split_source': {f'{s}:{source}': sum(r['source'] == source for r in rows)
                                             for s, rows in groups.items() for source in sources[s]},
               'training_ids': [r['id'] for r in groups['train']],
               'development_ids': [r['id'] for r in groups['dev']],
               'medical_dialogue': 'SIMORD train is reserved for per-fold adaptation; '
                                   'published SIMORD test1 never trains or selects'})
    model = runner.new_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    steps_per_epoch = math.ceil(math.ceil(len(features['train']) / args.batch_size) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * .06))

    def learning_rate_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    initial = evaluate(runner, model, groups['dev'], features['dev'])
    best_score, best_epoch = initial['macro_word_iou'], 0
    history = [{'epoch': 0, 'updates': 0, 'dev_macro_word_iou': best_score,
                'dev_word_iou_by_source': {k: v['positive_word_iou'] for k, v in initial['by_source'].items()}}]
    best_path = runner.out / 'model'
    runner.check_budget()
    model.save_pretrained(best_path, safe_serialization=True)
    runner.tokenizer.save_pretrained(best_path)
    write_json(runner.out / 'history.json', history)
    rng = random.Random(args.seed)
    updates, stale = 0, 0
    runner.save_status('running', optimizer_steps_planned=total_steps)
    runner.emit('pretrain_start', windows=len(features['train']), initial_dev_macro_word_iou=best_score)
    start_time = time.monotonic()
    stop = False
    for epoch in range(1, args.epochs + 1):
        shuffled = list(features['train'])
        rng.shuffle(shuffled)
        batches = [shuffled[i:i + args.batch_size] for i in range(0, len(shuffled), args.batch_size)]
        steps = [batches[i:i + args.accumulation] for i in range(0, len(batches), args.accumulation)]
        # Epochs over this corpus run for tens of minutes; select inside them, not only at the end.
        interval = math.ceil(len(steps) / args.evals_per_epoch)
        losses = []
        for index, group in enumerate(steps, start=1):
            loss, norm = runner.update(model, optimizer, group)
            scheduler.step()
            updates += 1
            losses.append(loss)
            if updates == 1 or updates % 100 == 0:
                runner.emit('pretrain_step', epoch=epoch, step=updates, loss=loss, gradient_norm=norm,
                            learning_rate=optimizer.param_groups[0]['lr'],
                            seconds_per_update=(time.monotonic() - start_time) / updates)
            if index % interval and index != len(steps):
                continue
            development = evaluate(runner, model, groups['dev'], features['dev'])
            metric = development['macro_word_iou']
            history.append({'epoch': epoch, 'updates': updates, 'progress': round(index / len(steps), 3),
                            'dev_macro_word_iou': metric,
                            'dev_word_iou_by_source': {k: v['positive_word_iou']
                                                       for k, v in development['by_source'].items()},
                            'training_loss': float(np.mean(losses))})
            losses = []
            write_json(runner.out / 'history.json', history)
            runner.emit('pretrain_eval', **history[-1])
            if metric > best_score:
                best_score, best_epoch, stale = metric, epoch, 0
                runner.check_budget()
                model.save_pretrained(best_path, safe_serialization=True)
            else:
                stale += 1
            if stale >= args.patience:
                stop = True
                break
        if stop:
            break
    del optimizer, scheduler, model
    gc.collect()
    if runner.device == 'cuda':
        torch.cuda.empty_cache()
    model = runner.new_model(best_path)
    report = evaluate(runner, model, groups['dev'], features['dev'])
    report.update(selected_epoch=best_epoch, selection='External dev macro IoU; not unbiased test performance',
                  external_sources=sources, history=history)
    write_json(runner.out / 'report.json', report)
    runner.save_status('complete', selected_epoch=best_epoch, dev_macro_word_iou=best_score,
                       next_step='Conversation-disjoint CV on original data with optional SIMORD train adaptation')
