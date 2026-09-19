"""Sliding-window QA features and exact occurrence-aware span decoding."""

from bisect import bisect_left, bisect_right
from collections import Counter

import numpy as np


def make_features(records, tokenizer, max_length=512, stride=192):
    if tokenizer.padding_side != 'right':
        raise ValueError('Right padding is required')
    if not 0 <= stride < max_length - 64:
        raise ValueError('Stride must leave room for the question and special tokens')
    features, coverage = [], Counter()
    for ri, record in enumerate(records):
        # Build overflow from the COMPLETE context before pairing with a question.
        # Some tokenizer/library combinations truncate the context once before
        # pair overflow, silently losing its tail. Keep original character offsets.
        backend = tokenizer.backend_tokenizer
        backend.no_truncation()
        backend.no_padding()
        question_encoding = backend.encode(record['question'], add_special_tokens=False)
        context_encoding = backend.encode(record['context'], add_special_tokens=False)
        available = max_length - len(question_encoding.ids) - tokenizer.num_special_tokens_to_add(pair=True)
        if available <= stride:
            raise ValueError(f'Question leaves insufficient context capacity: {record["id"]}')
        context_encoding.truncate(available, stride=stride)
        windows = [context_encoding, *context_encoding.overflowing]
        starts, ends = zip(*record['word_chars'])
        reachable_starts, reachable_ends = set(), set()
        for window in windows:
            encoded = backend.post_process(question_encoding, window, add_special_tokens=True)
            ids, sequence_ids, offsets = encoded.ids, encoded.sequence_ids, encoded.offsets
            if len(ids) > max_length:
                raise ValueError('Window construction exceeded model context length')
            cls = ids.index(tokenizer.cls_token_id)
            first_word, last_word = [-1] * len(ids), [-1] * len(ids)
            start_mask, end_mask = [False] * len(ids), [False] * len(ids)
            start_mask[cls] = end_mask[cls] = True
            for ti, ((left, right), seq) in enumerate(zip(offsets, sequence_ids)):
                if seq != 1 or right <= left:
                    continue
                lo, hi = bisect_right(ends, left), bisect_left(starts, right) - 1
                if lo > hi or lo >= len(starts) or hi < 0:
                    continue
                first_word[ti], last_word[ti] = lo, hi
                start_mask[ti] = left <= starts[lo] < right
                end_mask[ti] = left < ends[hi] <= right
                if start_mask[ti]:
                    reachable_starts.add(lo)
                if end_mask[ti]:
                    reachable_ends.add(hi)
            start_position = end_position = cls
            if record['label']:
                a, b = record['gold_words']
                possible_starts = [i for i in range(len(ids)) if start_mask[i] and first_word[i] == a]
                possible_ends = [i for i in range(len(ids)) if end_mask[i] and last_word[i] == b]
                if possible_starts and possible_ends and possible_starts[0] <= possible_ends[-1]:
                    start_position, end_position = possible_starts[0], possible_ends[-1]
                    coverage[record['id']] += 1
            features.append({
                'record_index': ri, 'cls': cls,
                'inputs': {k: v for k, v in {'input_ids': ids, 'attention_mask': encoded.attention_mask,
                                           'token_type_ids': encoded.type_ids}.items()
                           if k in tokenizer.model_input_names},
                'start_mask': start_mask, 'end_mask': end_mask,
                'first_word': first_word, 'last_word': last_word,
                'start_position': start_position, 'end_position': end_position,
            })
        if reachable_starts & reachable_ends != set(range(len(starts))):
            raise ValueError(f'Tokenizer windows do not cover every transcript word: {record["id"]}')
    missing = [r['id'] for r in records if r['label'] and not coverage[r['id']]]
    if missing:
        raise ValueError(f'Gold spans absent from every window; increase length/stride: {missing}')
    counts = Counter(f['record_index'] for f in features)
    for feature in features:
        feature['weight'] = 1 / counts[feature['record_index']]
    return features


def collate(features, tokenizer, device):
    import torch

    inputs = tokenizer.pad([f['inputs'] for f in features], padding=True, return_tensors='pt')
    width = inputs['input_ids'].shape[1]
    result = {'inputs': {k: v.to(device) for k, v in inputs.items()}}
    for key in ('start_mask', 'end_mask'):
        result[key] = torch.tensor([f[key] + [False] * (width - len(f[key])) for f in features],
                                   device=device, dtype=torch.bool)
    for key in ('start_position', 'end_position'):
        result[key] = torch.tensor([f[key] for f in features], device=device)
    result['weight'] = torch.tensor([f['weight'] for f in features], device=device)
    return result


def decode_window(feature, start_logits, end_logits, max_words=128):
    """Search all valid word-boundary pairs, comparing windows against their CLS score."""
    width = len(feature['first_word'])
    start_logits, end_logits = np.asarray(start_logits[:width]), np.asarray(end_logits[:width])
    if not np.isfinite(start_logits).all() or not np.isfinite(end_logits).all():
        raise ValueError('Non-finite model logits')
    starts = np.flatnonzero(np.array(feature['start_mask']) & (np.array(feature['first_word']) >= 0))
    ends = np.flatnonzero(np.array(feature['end_mask']) & (np.array(feature['last_word']) >= 0))
    if not len(starts) or not len(ends):
        return None
    first = np.array(feature['first_word'])[starts]
    last = np.array(feature['last_word'])[ends]
    valid = ((ends[None, :] >= starts[:, None]) & (last[None, :] >= first[:, None])
             & (last[None, :] - first[:, None] + 1 <= max_words))
    scores = start_logits[starts, None] + end_logits[None, ends]
    scores = np.where(valid, scores, -np.inf)
    if not np.isfinite(scores).any():
        return None
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    margin = scores[i, j] - start_logits[feature['cls']] - end_logits[feature['cls']]
    return {'words': [int(first[i]), int(last[j])], 'margin': float(margin)}


def prediction_spans(records, choices):
    result = {}
    for i, record in enumerate(records):
        if not record['baseline_answer']:
            result[record['id']] = None
        elif i in choices:
            start, end = choices[i]['words']
            result[record['id']] = [record['words'][start]['start'], record['words'][end]['end']]
        else:
            raise ValueError(f'No predicted span for a baseline YES: {record["id"]}')
    return result
