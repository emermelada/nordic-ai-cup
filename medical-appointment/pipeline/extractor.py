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
