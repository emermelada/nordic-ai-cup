"""Listwise span ranking over an enumerated candidate pool, trained on temporal IoU.

The extractor scores a start and an end independently, so no part of it can express a
property of the span as a whole. Gold evidence here is a convention -- the minimal
self-contained span -- which is exactly such a property, so this model scores every
enumerated candidate as a unit and takes the argmax. The pool is sentence runs and
comma-delimited clause spans; on the public set its oracle is 0.9243 temporal IoU.

Training and inference enumerate identically, so nothing about gold leaks into the pool.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .data import digest, validate_bundle, write_json
from .external import hydrate, load_external

MAX_SENTENCES = 4
MAX_CANDIDATE_WORDS = 40
POSITIVE_IOU = 0.6
TARGET_TEMPERATURE = 0.15
MASKED_SCORE = -1e4


# --------------------------------------------------------------------------- pool

def word_texts(record):
    if 'words' in record:
        return [w['word'] for w in record['words']]
    context = record['context']
    return [context[a:b] for a, b in record['word_chars']]


def sentence_ranges(texts):
    """Inclusive word-index ranges split on end punctuation, the annotators' unit."""
    ranges, first = [], 0
    for index, text in enumerate(texts):
        if text.strip().endswith(('.', '?', '!')) or index == len(texts) - 1:
            ranges.append((first, index))
            first = index + 1
    return ranges


def candidates(texts, max_sentences=MAX_SENTENCES, max_words=MAX_CANDIDATE_WORDS):
    """Sentence runs of one to `max_sentences`, plus clause spans inside a sentence."""
    sentences = sentence_ranges(texts)
    pool = set()
    for a in range(len(sentences)):
        for n in range(1, max_sentences + 1):
            b = a + n - 1
            if b >= len(sentences):
                break
            pool.add((sentences[a][0], sentences[b][1]))
    for first, last in sentences:
        cuts = [first - 1]
        cuts += [k for k in range(first, last + 1) if texts[k].strip().endswith((',', ';', ':'))]
        cuts.append(last)
        for a in range(len(cuts) - 1):
            for b in range(a + 1, len(cuts)):
                start, end = cuts[a] + 1, cuts[b]
                if end >= start:
                    pool.add((start, end))
    return sorted(c for c in pool if c[1] - c[0] + 1 <= max_words)


def word_iou(a, b):
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi < lo:
        return 0.0
    intersection = hi - lo + 1
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return intersection / union


def temporal_iou(gold, prediction):
    if not gold or not prediction or prediction[1] <= prediction[0]:
        return 0.0
    intersection = max(0.0, min(gold[1], prediction[1]) - max(gold[0], prediction[0]))
    union = max(gold[1], prediction[1]) - min(gold[0], prediction[0])
    return intersection / union if union > 0 else 0.0


def candidate_targets(record, pool):
    """Overlap of every candidate with gold: temporal when the record has timestamps.

    Serving records carry no gold, so every candidate scores zero and every window is a
    null window, which is exactly what inference needs.
    """
    gold_words = record.get('gold_words')
    if not record.get('label') or gold_words is None:
        return [0.0] * len(pool)
    if 'words' in record:
        words = record['words']
        gold = record['gold']
        return [temporal_iou(gold, [words[a]['start'], words[b]['end']]) for a, b in pool]
    return [word_iou((a, b), tuple(gold_words)) for a, b in pool]


def span_time(words, a, b):
    return [words[a]['start'], words[b]['end']]


# ----------------------------------------------------------------------- features

def windows(record, tokenizer, max_length=512, stride=192):
    """Sliding windows with, for every word, the token that may open and close it.

    Overflow is built from the complete context before the question is paired in;
    pairing first can silently truncate a long transcript's tail.
    """
    backend = tokenizer.backend_tokenizer
    backend.no_truncation()
    backend.no_padding()
    question = backend.encode(record['question'], add_special_tokens=False)
    context = backend.encode(record['context'], add_special_tokens=False)
    available = max_length - len(question.ids) - tokenizer.num_special_tokens_to_add(pair=True)
    if available <= stride:
        raise ValueError(f'Question leaves insufficient context capacity: {record["id"]}')
    context.truncate(available, stride=stride)
    starts, ends = zip(*record['word_chars'])
    result = []
    for window in [context, *context.overflowing]:
        encoded = backend.post_process(question, window, add_special_tokens=True)
        ids = encoded.ids
        if len(ids) > max_length:
            raise ValueError('Window construction exceeded model context length')
        cls = ids.index(tokenizer.cls_token_id)
        # Offsets are read from the window before pairing, not from the paired encoding.
        # RoBERTa's post-processor trims each token's leading space a second time, which
        # moves a word's first character out of its own token; post-processing only wraps
        # the context tokens, so their order and count are unchanged.
        positions = [ti for ti, seq in enumerate(encoded.sequence_ids) if seq == 1]
        if len(positions) != len(window.ids):
            raise ValueError('Pairing changed the context tokens; cannot align words')
        offsets = window.offsets
        start_token, end_token = {}, {}
        overlap_first, overlap_last = {}, {}
        for local, (left, right) in enumerate(offsets):
            ti = positions[local]
            if right <= left:
                continue
            lo, hi = bisect_right(ends, left), bisect_left(starts, right) - 1
            if lo > hi or lo >= len(starts) or hi < 0:
                continue
            if left <= starts[lo] < right and lo not in start_token:
                start_token[lo] = ti
            if left < ends[hi] <= right:
                end_token[hi] = ti
            # Some tokenizers report offsets that miss a word's first character: RoBERTa's
            # post-processor trims the leading space a second time, so "Doctor" at 9-15
            # arrives as 10-15 and matches no boundary exactly. Overlap is the fallback.
            overlap_first.setdefault(lo, ti)
            overlap_last[hi] = ti
        for word, ti in overlap_first.items():
            start_token.setdefault(word, ti)
        for word, ti in overlap_last.items():
            end_token.setdefault(word, ti)
        result.append({
            'cls': cls,
            'inputs': {k: v for k, v in {'input_ids': ids, 'attention_mask': encoded.attention_mask,
                                         'token_type_ids': encoded.type_ids}.items()
                       if k in tokenizer.model_input_names},
            'start_token': start_token, 'end_token': end_token,
        })
    return result


def make_features(records, tokenizer, max_length=512, stride=192, max_candidates=None,
                  rng=None, keep_null=0.0, report_unreachable=None):
    """One feature per window, carrying the candidates that window can express."""
    features, unreachable = [], []
    counts = Counter()
    for index, record in enumerate(records):
        pool = candidates(word_texts(record))
        if not pool:
            continue
        targets = candidate_targets(record, pool)
        record_features = []
        for window in windows(record, tokenizer, max_length, stride):
            local = [(k, window['start_token'][a], window['end_token'][b])
                     for k, (a, b) in enumerate(pool)
                     if a in window['start_token'] and b in window['end_token']
                     and window['start_token'][a] <= window['end_token'][b]]
            if not local:
                continue
            best = max(targets[k] for k, _, _ in local)
            positive = best >= POSITIVE_IOU
            if rng is not None and not positive:
                if keep_null <= 0 or rng.random() >= keep_null:
                    continue
            if max_candidates and len(local) > max_candidates and rng is not None:
                ordered = sorted(local, key=lambda item: -targets[item[0]])
                head = ordered[:8]
                tail = rng.sample(ordered[8:], max_candidates - len(head))
                local = head + tail
            record_features.append({
                'record_index': index, 'cls': window['cls'], 'inputs': window['inputs'],
                'candidate_index': [k for k, _, _ in local],
                'start': [s for _, s, _ in local], 'end': [e for _, _, e in local],
                'target': [targets[k] for k, _, _ in local], 'positive': positive,
            })
        if record.get('label') and not any(f['positive'] for f in record_features):
            unreachable.append(record['id'])
        for feature in record_features:
            counts[index] += 1
        features.extend(record_features)
    for feature in features:
        feature['weight'] = 1 / counts[feature['record_index']]
    if report_unreachable is not None:
        report_unreachable.extend(unreachable)
    return features


def soft_targets(values, temperature=TARGET_TEMPERATURE):
    """Mass concentrated on the candidates closest to gold, none on the null slot."""
    best = max(values)
    weights = [math.exp((value - best) / temperature) for value in values]
    total = sum(weights)
    return [w / total for w in weights]


def collate(batch, tokenizer, device):
    inputs = tokenizer.pad([f['inputs'] for f in batch], padding=True, return_tensors='pt')
    width = max(len(f['candidate_index']) for f in batch)
    size = len(batch)
    start = torch.zeros(size, width, dtype=torch.long)
    end = torch.zeros(size, width, dtype=torch.long)
    mask = torch.zeros(size, width, dtype=torch.bool)
    target = torch.zeros(size, 1 + width)
    for i, feature in enumerate(batch):
        n = len(feature['candidate_index'])
        start[i, :n] = torch.tensor(feature['start'])
        end[i, :n] = torch.tensor(feature['end'])
        mask[i, :n] = True
        if feature['positive']:
            target[i, 1:1 + n] = torch.tensor(soft_targets(feature['target']))
        else:
            target[i, 0] = 1.0
    return {
        'inputs': {k: v.to(device) for k, v in inputs.items()},
        'cls': torch.tensor([f['cls'] for f in batch], device=device),
        'start': start.to(device), 'end': end.to(device), 'mask': mask.to(device),
        'target': target.to(device),
        'weight': torch.tensor([f['weight'] for f in batch], device=device),
    }


# -------------------------------------------------------------------------- model

class SpanRanker(nn.Module):
    def __init__(self, encoder, hidden, projection=256, width_buckets=64, width_size=32,
                 dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.start_projection = nn.Linear(hidden, projection)
        self.end_projection = nn.Linear(hidden, projection)
        self.mean_projection = nn.Linear(hidden, projection)
        self.width = nn.Embedding(width_buckets, width_size)
        self.width_buckets = width_buckets
        self.score = nn.Sequential(
            nn.Linear(3 * projection + width_size, projection), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(projection, 1))
        self.null = nn.Sequential(
            nn.Linear(hidden, projection), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(projection, 1))

    def forward(self, inputs, cls, start, end, mask):
        hidden = self.encoder(**inputs).last_hidden_state
        rows = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
        cumulative = torch.cat([torch.zeros_like(hidden[:, :1], dtype=torch.float32),
                                hidden.float().cumsum(1)], dim=1)
        span_mean = ((cumulative[rows, end + 1] - cumulative[rows, start])
                     / (end - start + 1).unsqueeze(-1).float())
        features = torch.cat([
            self.start_projection(hidden[rows, start]),
            self.end_projection(hidden[rows, end]),
            self.mean_projection(span_mean.to(hidden.dtype)),
            self.width(torch.clamp(end - start, 0, self.width_buckets - 1)),
        ], dim=-1)
        scores = self.score(features).squeeze(-1).float()
        # A finite sentinel, not -inf: a padded slot's log-probability multiplies a zero
        # target, and 0 * -inf is NaN.
        scores = scores.masked_fill(~mask, MASKED_SCORE)
        null = self.null(hidden[torch.arange(hidden.shape[0], device=hidden.device), cls]).float()
        return torch.cat([null, scores], dim=1)


def build_model(path, revision=None, device='cpu', dropout=0.1):
    config = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(path, revision=revision, trust_remote_code=False)
    model = SpanRanker(encoder, config.hidden_size, dropout=dropout)
    return model.to(device)


# ----------------------------------------------------------------------- training

def batches(features, size, rng=None, by_length=True):
    order = list(range(len(features)))
    if rng is not None:
        rng.shuffle(order)
    if by_length:
        order.sort(key=lambda i: len(features[i]['inputs']['input_ids']) // 64)
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    if rng is not None:
        rng.shuffle(chunks)
    return [[features[i] for i in chunk] for chunk in chunks]


def candidate_scores(model, features, tokenizer, device, batch_size=16, autocast=None,
                     deadline=None):
    """{record index: {candidate key: score}}, each window compared against its own null.

    Past `deadline` the remaining windows are dropped: a record scored in an earlier window
    keeps what that window gave it, and one scored in none gets no entry.
    """
    model.eval()
    best = {}
    with torch.no_grad():
        for batch in batches(features, batch_size):
            if deadline is not None and time.monotonic() > deadline:
                break
            tensors = collate(batch, tokenizer, device)
            with autocast() if autocast else torch.autocast('cpu', enabled=False):
                logits = model(tensors['inputs'], tensors['cls'], tensors['start'],
                               tensors['end'], tensors['mask'])
            logits = logits.float().cpu().numpy()
            for row, feature in enumerate(batch):
                null = logits[row, 0]
                for column, key in enumerate(feature['candidate_index']):
                    value = logits[row, 1 + column] - null
                    slot = best.setdefault(feature['record_index'], {})
                    if key not in slot or value > slot[key]:
                        slot[key] = float(value)
    return best


def predict(model, records, features, tokenizer, device, batch_size=16, autocast=None,
            deadline=None):
    """The highest-scoring candidate per record."""
    best = candidate_scores(model, features, tokenizer, device, batch_size, autocast, deadline)
    spans = {}
    for index, scores in best.items():
        record = records[index]
        pool = candidates(word_texts(record))
        key = max(scores, key=scores.get)
        spans[record['id']] = {'words': list(pool[key]), 'score': scores[key]}
    return spans


def score_records(records, spans):
    """Mean overlap of the chosen candidate with gold, in the record's own metric."""
    values = []
    for record in records:
        if not record.get('label'):
            continue
        chosen = spans.get(record['id'])
        if chosen is None:
            values.append(0.0)
            continue  # a pool that reaches nothing scores nothing, never silently skipped
        a, b = chosen['words']
        if 'words' in record:
            values.append(temporal_iou(record['gold'], span_time(record['words'], a, b)))
        else:
            values.append(word_iou((a, b), tuple(record['gold_words'])))
    return sum(values) / len(values) if values else 0.0


class Runner:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.max_seconds
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=False)
        self.events = self.out / 'events.jsonl'
        self.device = args.device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_bf16 = self.device == 'cuda' and torch.cuda.is_bf16_supported()
        self.bundle = json.loads(args.data.read_text())
        validate_bundle(self.bundle)
        self.records = self.bundle['records']
        self.external = load_external(args.external_data, args.data) if args.external_data else None
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                                       trust_remote_code=False)
        self.tokenizer.padding_side = 'right'
        self.metadata = {
            'status': 'initializing',
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'data_sha256': digest(args.data), 'scope': self.bundle['scope'],
            'device': self.device, 'folds': self.bundle['folds'],
            'pool': {'max_sentences': MAX_SENTENCES, 'max_words': MAX_CANDIDATE_WORDS},
            'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers')},
            'code_sha256': {Path(__file__).name: digest(Path(__file__))},
            'answers_frozen': True,
        }
        if self.device == 'cuda':
            free, total = torch.cuda.mem_get_info()
            self.metadata['gpu'] = {'name': torch.cuda.get_device_name(), 'free_bytes': free}
            if free < args.min_free_gb * 1024 ** 3:
                raise RuntimeError(f'Only {free / 1024 ** 3:.1f} GiB free; need {args.min_free_gb}. '
                                   'This runner never stops another process.')
        self.save('ready')

    def autocast(self):
        if self.use_bf16:
            return lambda: torch.autocast('cuda', dtype=torch.bfloat16)
        return None

    def emit(self, event, **values):
        row = {'event': event, 'elapsed_seconds': round(time.monotonic() - self.started, 3), **values}
        with self.events.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def save(self, status, **extra):
        self.metadata.update(status=status, elapsed_seconds=time.monotonic() - self.started, **extra)
        write_json(self.out / 'manifest.json', self.metadata)

    def budget(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Wall-time budget reached')

    def train(self, model, features, epochs, learning_rate, head_learning_rate, evals, evaluate,
              tag, batch_size=None, accumulation=None):
        """Train, evaluating `evals` times per epoch, keeping the best state seen."""
        if not features:
            raise ValueError(f'No training windows for {tag}; the candidate pool reached no gold')
        batch_size = batch_size or self.args.batch_size
        accumulation = accumulation or self.args.accumulation
        head = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
        encoder = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
        optimizer = torch.optim.AdamW(
            [{'params': encoder, 'lr': learning_rate},
             {'params': head, 'lr': head_learning_rate}], weight_decay=0.01)
        chunks = math.ceil(len(features) / batch_size)
        total = max(1, epochs * math.ceil(chunks / accumulation))
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[learning_rate, head_learning_rate], total_steps=total,
            pct_start=0.06, anneal_strategy='linear', cycle_momentum=False, div_factor=25,
            final_div_factor=100)
        rng = random.Random(self.args.seed)
        best = {'value': float('-inf'), 'state': None, 'epoch': 0}
        autocast = self.autocast()
        step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            marks = {max(1, round(chunks * (i + 1) / evals)) for i in range(evals)}
            running, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for position, batch in enumerate(batches(features, batch_size, rng), start=1):
                self.budget()
                tensors = collate(batch, self.tokenizer, self.device)
                with autocast() if autocast else torch.autocast('cpu', enabled=False):
                    logits = model(tensors['inputs'], tensors['cls'], tensors['start'],
                                   tensors['end'], tensors['mask'])
                loss = -(tensors['target'] * torch.log_softmax(logits.float(), dim=1)).sum(1)
                loss = (loss * tensors['weight']).sum() / tensors['weight'].sum()
                (loss / accumulation).backward()
                running += float(loss)
                seen += 1
                if position % accumulation == 0 or position == chunks:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if step + 1 < total:
                        schedule.step()
                    step += 1
                    optimizer.zero_grad(set_to_none=True)
                if position in marks:
                    value = evaluate(model) if evaluate else float('nan')
                    self.emit('evaluation', tag=tag, epoch=epoch, position=position,
                              loss=running / max(1, seen),
                              value=None if value != value else value)
                    running, seen = 0.0, 0
                    if self.args.select == 'dev' and value > best['value']:
                        best = {'value': value, 'epoch': epoch,
                                'state': {k: v.detach().to('cpu', copy=True)
                                          for k, v in model.state_dict().items()}}
                    model.train()
        if self.args.select == 'dev' and best['state'] is not None:
            model.load_state_dict(best['state'])
            return best
        # Nothing was selected, so there is no development score to report.
        return {'value': None, 'epoch': epochs, 'state': None}

    # ------------------------------------------------------------------ pretrain
    def pretrain(self):
        rng = random.Random(self.args.seed)
        train_records = [r for r in hydrate(self.external, 'train')
                         if r['label'] and r['gold_words'] and r['source'] in self.args.sources]
        dev_records = [r for r in hydrate(self.external, 'dev')
                       if r['label'] and r['gold_words'] and r['source'] in self.args.sources]
        if self.args.limit:
            rng.shuffle(train_records)
            train_records = train_records[:self.args.limit]
        dev_records = dev_records[:self.args.dev_limit]
        for index, record in enumerate(train_records + dev_records):
            record['id'] = record.get('id') or f'external:{index}'
        self.emit('pretrain_data', train=len(train_records), dev=len(dev_records))
        features = make_features(train_records, self.tokenizer, self.args.max_length,
                                 self.args.stride, self.args.max_candidates, rng, keep_null=0.3)
        dev_features = make_features(dev_records, self.tokenizer, self.args.max_length,
                                     self.args.stride)
        self.emit('pretrain_features', train=len(features), dev=len(dev_features))
        model = build_model(self.args.model, device=self.device, dropout=self.args.dropout)
        if self.args.ranker_state:
            model.load_state_dict(torch.load(self.args.ranker_state, map_location='cpu'))
            model.to(self.device)

        def evaluate(current):
            spans = predict(current, dev_records, dev_features, self.tokenizer, self.device,
                            self.args.eval_batch_size, self.autocast())
            return score_records(dev_records, spans)

        best = self.train(model, features, self.args.epochs, self.args.learning_rate,
                          self.args.head_learning_rate, self.args.evals_per_epoch, evaluate,
                          'pretrain')
        destination = self.out / 'model'
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination / 'ranker.pt')
        model.encoder.config.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        self.save('complete', best_dev=best['value'], best_epoch=best['epoch'])
        self.emit('pretrain_complete', best_dev=best['value'], best_epoch=best['epoch'])
        return best

    # ------------------------------------------------------------------------ cv
    def cv(self):
        rng = random.Random(self.args.seed)
        by_conversation = {}
        for record in self.records:
            by_conversation.setdefault(record['conversation'], []).append(record)
        report = {'folds': [], 'pool': self.metadata['pool']}
        chosen = {}
        for fold in self.bundle['folds']:
            if self.args.fold is not None and fold['fold'] != self.args.fold:
                continue
            self.budget()
            split = {name: [r for c in fold[name] for r in by_conversation[c]]
                     for name in ('train', 'dev', 'test')}
            if self.args.select == 'final':
                # Nothing is chosen on them, so they are training data.
                split['train'] = split['train'] + split['dev']
            train_positive = [r for r in split['train'] if r['label']]
            dev_positive = [r for r in split['dev'] if r['label']]
            test_positive = [r for r in split['test'] if r['label']]
            features = make_features(train_positive, self.tokenizer, self.args.max_length,
                                     self.args.stride, self.args.max_candidates, rng,
                                     keep_null=0.3)
            dev_features = make_features(dev_positive, self.tokenizer, self.args.max_length,
                                         self.args.stride)
            test_features = make_features(test_positive, self.tokenizer, self.args.max_length,
                                          self.args.stride)
            model = build_model(self.args.model, device=self.device, dropout=self.args.dropout)
            if self.args.ranker_state:
                state = torch.load(self.args.ranker_state, map_location='cpu')
                model.load_state_dict(state)
                model.to(self.device)

            def evaluate(current):
                spans = predict(current, dev_positive, dev_features, self.tokenizer, self.device,
                                self.args.eval_batch_size, self.autocast())
                return score_records(dev_positive, spans)

            if self.args.select == 'final':
                evaluate = None
            self.emit('fold_start', fold=fold['fold'], train_windows=len(features),
                      train_questions=len(train_positive), test_questions=len(test_positive))
            best = self.train(model, features, self.args.epochs, self.args.learning_rate,
                              self.args.head_learning_rate, self.args.evals_per_epoch, evaluate,
                              f'fold{fold["fold"]}')
            spans = predict(model, test_positive, test_features, self.tokenizer, self.device,
                            self.args.eval_batch_size, self.autocast())
            for record in test_positive:
                selection = spans.get(record['id'])
                if selection is None:
                    span, words, value = record['baseline_span'], None, None
                else:
                    words = list(selection['words'])
                    span = span_time(record['words'], *words)
                    value = selection['score']
                chosen[record['id']] = {
                    'conversation': record['conversation'], 'fold': fold['fold'],
                    'words': words, 'span': span, 'gold': record['gold'], 'score': value,
                    'tiou': temporal_iou(record['gold'], span),
                    'baseline_tiou': temporal_iou(record['gold'], record['baseline_span']),
                }
            value = score_records(test_positive, spans)
            baseline = sum(temporal_iou(r['gold'], r['baseline_span']) for r in test_positive) / len(test_positive)
            report['folds'].append({'fold': fold['fold'], 'best_dev': best['value'],
                                    'best_epoch': best['epoch'], 'test_tiou': value,
                                    'baseline_tiou': baseline, 'questions': len(test_positive)})
            self.emit('fold_complete', fold=fold['fold'], test_tiou=value, baseline_tiou=baseline,
                      best_dev=best['value'], best_epoch=best['epoch'])
            if self.args.save_folds:
                destination = self.out / f'fold{fold["fold"]}'
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), destination / 'ranker.pt')
            del model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        report['rows'] = chosen
        if len(chosen) == sum(1 for r in self.records if r['label']):
            values = [row['tiou'] for row in chosen.values()]
            base = [row['baseline_tiou'] for row in chosen.values()]
            report['pooled'] = {
                'questions': len(values), 'mean_tiou': sum(values) / len(values),
                'raw': 0.4 + 0.6 * sum(values) / len(values),
                'baseline_mean_tiou': sum(base) / len(base),
                'baseline_raw': 0.4 + 0.6 * sum(base) / len(base),
                'zero_overlap': sum(1 for v in values if v == 0),
                'high_overlap': sum(1 for v in values if v >= 0.9),
            }
            self.emit('pooled', **report['pooled'])
        write_json(self.out / 'report.json', report)
        self.save('complete')
        return report

    # -------------------------------------------------------------------- dump
    def dump(self):
        """Score every candidate of each test question with its own fold's checkpoint.

        Selection rules can then be compared offline without more GPU time; every score
        here is out of fold.
        """
        by_conversation = {}
        for record in self.records:
            by_conversation.setdefault(record['conversation'], []).append(record)
        rows = {}
        for fold in self.bundle['folds']:
            state_path = self.args.fold_states / f'fold{fold["fold"]}' / 'ranker.pt'
            if not state_path.exists():
                raise FileNotFoundError(f'No checkpoint for fold {fold["fold"]}: {state_path}')
            model = build_model(self.args.model, device=self.device, dropout=self.args.dropout)
            model.load_state_dict(torch.load(state_path, map_location='cpu'))
            model.to(self.device)
            test_positive = [r for c in fold['test'] for r in by_conversation[c] if r['label']]
            features = make_features(test_positive, self.tokenizer, self.args.max_length,
                                     self.args.stride)
            scores = candidate_scores(model, features, self.tokenizer, self.device,
                                      self.args.eval_batch_size, self.autocast())
            for index, record in enumerate(test_positive):
                pool = candidates(word_texts(record))
                ranked = sorted(scores.get(index, {}).items(), key=lambda kv: -kv[1])
                rows[record['id']] = {
                    'conversation': record['conversation'], 'fold': fold['fold'],
                    'gold': record['gold'], 'baseline_span': record['baseline_span'],
                    'candidates': [[pool[k][0], pool[k][1], v,
                                    record['words'][pool[k][0]]['start'],
                                    record['words'][pool[k][1]]['end']]
                                   for k, v in ranked[:self.args.dump_top]],
                }
            self.emit('dump_fold', fold=fold['fold'], questions=len(test_positive))
            del model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        write_json(self.out / 'candidates.json', {'rows': rows, 'pool': self.metadata['pool']})
        self.save('complete', questions=len(rows))
        return rows

    # --------------------------------------------------------------------- fit
    def fit(self):
        """Train for deployment on every public conversation but the two kept for selection."""
        rng = random.Random(self.args.seed)
        reserved = set(self.bundle['folds'][0]['dev'])
        positive = [r for r in self.records if r['label'] and r['conversation'] not in reserved]
        holdout = [r for r in self.records if r['label'] and r['conversation'] in reserved]
        features = make_features(positive, self.tokenizer, self.args.max_length, self.args.stride,
                                 self.args.max_candidates, rng, keep_null=0.3)
        dev_features = make_features(holdout, self.tokenizer, self.args.max_length, self.args.stride)
        model = build_model(self.args.model, device=self.device, dropout=self.args.dropout)
        if self.args.ranker_state:
            model.load_state_dict(torch.load(self.args.ranker_state, map_location='cpu'))
            model.to(self.device)

        def evaluate(current):
            spans = predict(current, holdout, dev_features, self.tokenizer, self.device,
                            self.args.eval_batch_size, self.autocast())
            return score_records(holdout, spans)

        self.emit('fit_start', questions=len(positive), windows=len(features))
        best = self.train(model, features, self.args.epochs, self.args.learning_rate,
                          self.args.head_learning_rate, self.args.evals_per_epoch, evaluate, 'fit')
        destination = self.out / 'model'
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination / 'ranker.pt')
        model.encoder.config.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        self.save('complete', best_dev=best['value'], best_epoch=best['epoch'],
                  selection_conversations=sorted(reserved),
                  in_sample_warning='Trained on the public conversations; only the fold CV is honest')
        self.emit('fit_complete', best_dev=best['value'], best_epoch=best['epoch'])
        return best


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--external-data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('pretrain', 'cv', 'fit', 'dump'), required=True)
    parser.add_argument('--model', default='deepset/deberta-v3-large-squad2')
    parser.add_argument('--ranker-state', type=Path, help='ranker.pt to initialize from')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--evals-per-epoch', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--head-learning-rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=192)
    parser.add_argument('--max-candidates', type=int, default=192)
    parser.add_argument('--limit', type=int, help='cap external training questions')
    parser.add_argument('--dev-limit', type=int, default=400)
    parser.add_argument('--sources', nargs='+', default=['coqa', 'mashqa'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-seconds', type=float, default=3600)
    parser.add_argument('--min-free-gb', type=float, default=16)
    parser.add_argument('--save-folds', action='store_true')
    parser.add_argument('--fold-states', type=Path, help='cv output holding foldN/ranker.pt')
    parser.add_argument('--dump-top', type=int, default=60)
    parser.add_argument('--select', choices=('dev', 'final'), default='dev',
                        help="'final' keeps the last state and trains on the development "
                             'conversations, which two of them cannot discriminate anyway')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    runner = Runner(args)
    return getattr(runner, args.mode)()


if __name__ == '__main__':
    main()
