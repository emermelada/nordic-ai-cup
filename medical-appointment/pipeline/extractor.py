"""Trained span extractor as an independent producer for the evidence vote.

Loads nothing unless MEDICAL_EXTRACTOR_MODEL names a checkpoint, and any failure
disables it for the process instead of the request: the vote then runs with the
producers that remain, which is the existing behaviour.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get('MEDICAL_EXTRACTOR_MODEL', '')
MAX_LENGTH = int(os.environ.get('MEDICAL_EXTRACTOR_MAX_LENGTH', '512'))
STRIDE = int(os.environ.get('MEDICAL_EXTRACTOR_STRIDE', '192'))
MAX_SPAN_WORDS = int(os.environ.get('MEDICAL_EXTRACTOR_MAX_SPAN_WORDS', '128'))
BATCH_SIZE = int(os.environ.get('MEDICAL_EXTRACTOR_BATCH', '8'))
# How far either side of the chosen span an endpoint may move. Small enough that
# refinement cannot walk the span onto a different passage.
MARGIN_WORDS = int(os.environ.get('MEDICAL_EXTRACTOR_MARGIN_WORDS', '6'))

_lock = threading.Lock()
_state = {'attempted': False, 'model': None, 'tokenizer': None, 'device': None}


def _load():
    """Load once per process; a failed load is remembered so every request stays fast."""
    if _state['attempted']:
        return _state['model'] is not None
    with _lock:
        if _state['attempted']:
            return _state['model'] is not None
        _state['attempted'] = True
        if not MODEL_PATH:
            return False
        try:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=False)
            tokenizer.padding_side = 'right'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH, trust_remote_code=False)
            model.to(device).eval()
            _state.update(model=model, tokenizer=tokenizer, device=device)
            logger.info('Span extractor loaded from %s on %s', MODEL_PATH, device)
        except Exception:
            logger.exception('Span extractor unavailable; the evidence vote loses one producer')
            return False
    return True


def warmup():
    """Load the checkpoint during worker startup instead of inside the first request."""
    if MODEL_PATH and _load():
        logger.info('Span extractor warm')


def _encode(words, questions, answers):
    """Windows for every answered question, with the numbers they belong to."""
    from tools.evidence_training.data import render_words
    from tools.evidence_training.features import make_features

    context, ranges = render_words(words)
    records, numbers = [], []
    for index, question in enumerate(questions):
        if not answers[index]:
            continue
        numbers.append(index + 1)
        records.append({'id': str(index + 1), 'question': question, 'context': context,
                        'word_chars': ranges, 'label': False, 'gold_words': None})
    features = make_features(records, _state['tokenizer'], MAX_LENGTH, STRIDE)
    return features, numbers


def _logits(features, deadline=None):
    import torch

    tokenizer, model, device = _state['tokenizer'], _state['model'], _state['device']
    from tools.evidence_training.features import collate

    out = []
    with torch.inference_mode():
        for offset in range(0, len(features), BATCH_SIZE):
            if deadline is not None and time.monotonic() > deadline:
                logger.warning('Span extractor stopped at the request deadline')
                break
            batch = features[offset:offset + BATCH_SIZE]
            result = model(**collate(batch, tokenizer, device)['inputs'])
            out.extend(zip(result.start_logits.float().cpu().numpy(),
                           result.end_logits.float().cpu().numpy()))
    return out


def refine_spans(words, questions, answers, anchors, deadline=None):
    """Re-pick each span's endpoints strictly inside the neighbourhood it already occupies.

    Measured out of fold, the extractor is the better boundary model and the worse
    locator, so it is given no opportunity to relocate: candidate endpoints are limited
    to MARGIN_WORDS either side of the span the evidence stage chose. A question whose
    anchor is missing keeps its span untouched.
    """
    if not any(answers) or not _load():
        return {}
    if deadline is not None and time.monotonic() > deadline:
        return {}
    try:
        from tools.evidence_training.data import best_word_span

        features, numbers = _encode(words, questions, answers)
        logits = _logits(features, deadline)
        if len(logits) < len(features):
            return {}
        allowed = {}
        for position, number in enumerate(numbers):
            anchor = anchors[number - 1] if number - 1 < len(anchors) else None
            if not anchor or anchor[0] is None or anchor[1] is None:
                continue
            first, last = best_word_span(words, list(anchor))[0]
            allowed[position] = (max(0, first - MARGIN_WORDS), min(len(words) - 1, last + MARGIN_WORDS))
        best = {}
        for feature, (start_logits, end_logits) in zip(features, logits):
            position = feature['record_index']
            if position not in allowed:
                continue
            low, high = allowed[position]
            cls = feature['cls']
            starts = [i for i, (m, w) in enumerate(zip(feature['start_mask'], feature['first_word']))
                      if m and low <= w <= high]
            ends = [i for i, (m, w) in enumerate(zip(feature['end_mask'], feature['last_word']))
                    if m and low <= w <= high]
            for i in starts:
                for j in ends:
                    if j < i or feature['last_word'][j] < feature['first_word'][i]:
                        continue
                    if feature['last_word'][j] - feature['first_word'][i] + 1 > MAX_SPAN_WORDS:
                        continue
                    score = float(start_logits[i] + end_logits[j]
                                  - start_logits[cls] - end_logits[cls])
                    if position not in best or score > best[position][0]:
                        best[position] = (score, feature['first_word'][i], feature['last_word'][j])
        return {numbers[position]: (words[a]['start'], words[b]['end'])
                for position, (_, a, b) in best.items()}
    except Exception:
        logger.exception('Span refinement failed; the selected spans are kept')
        return {}


def predict_spans(words, questions, answers, deadline=None):
    """{question number: (start, end)} for answered questions; {} when unavailable."""
    if not any(answers) or not _load():
        return {}
    if deadline is not None and time.monotonic() > deadline:
        logger.warning('Span extractor skipped: past the request deadline')
        return {}
    try:
        import torch

        from tools.evidence_training.data import render_words
        from tools.evidence_training.features import collate, decode_window, make_features

        context, ranges = render_words(words)
        records, numbers = [], []
        for index, question in enumerate(questions):
            if not answers[index]:
                continue
            numbers.append(index + 1)
            records.append({'id': str(index + 1), 'question': question, 'context': context,
                            'word_chars': ranges, 'label': False, 'gold_words': None})
        tokenizer, model, device = _state['tokenizer'], _state['model'], _state['device']
        features = make_features(records, tokenizer, MAX_LENGTH, STRIDE)
        choices = {}
        with torch.inference_mode():
            for offset in range(0, len(features), BATCH_SIZE):
                if deadline is not None and time.monotonic() > deadline:
                    logger.warning('Span extractor stopped at the request deadline')
                    break
                batch = features[offset:offset + BATCH_SIZE]
                inputs = collate(batch, tokenizer, device)
                output = model(**inputs['inputs'])
                starts = output.start_logits.float().cpu().numpy()
                ends = output.end_logits.float().cpu().numpy()
                for feature, start_logits, end_logits in zip(batch, starts, ends):
                    candidate = decode_window(feature, start_logits, end_logits, MAX_SPAN_WORDS)
                    key = feature['record_index']
                    if candidate and (key not in choices or candidate['margin'] > choices[key]['margin']):
                        choices[key] = candidate
        spans = {}
        for key, candidate in choices.items():
            first, last = candidate['words']
            spans[numbers[key]] = (words[first]['start'], words[last]['end'])
        return spans
    except Exception:
        logger.exception('Span extractor failed; the evidence vote loses one producer')
        return {}
