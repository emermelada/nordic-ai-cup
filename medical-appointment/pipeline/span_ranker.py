"""Trained span ranker over the enumerated candidate pool.

Stage B generates a quote and the aligner turns it into a span. This scores every
sentence run and clause span in the transcript as a unit and keeps the argmax, which is
what picking between two true mentions of the same fact needs.

Loads nothing unless MEDICAL_RANKER_MODEL names a checkpoint, and any failure disables it
for the process rather than the request: the stage-B spans then stand unchanged.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get('MEDICAL_RANKER_MODEL', '')
MAX_LENGTH = int(os.environ.get('MEDICAL_RANKER_MAX_LENGTH', '512'))
STRIDE = int(os.environ.get('MEDICAL_RANKER_STRIDE', '192'))
BATCH_SIZE = int(os.environ.get('MEDICAL_RANKER_BATCH', '16'))
MIN_SCORE = float(os.environ.get('MEDICAL_RANKER_MIN_SCORE', '-inf'))

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
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            from tools.evidence_training.ranker import SpanRanker

            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True,
                                                      trust_remote_code=False)
            tokenizer.padding_side = 'right'
            config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)
            # from_config, not from_pretrained: every weight comes from ranker.pt below,
            # so nothing is fetched and no pretrained head is silently kept.
            model = SpanRanker(AutoModel.from_config(config), config.hidden_size)
            state = torch.load(os.path.join(MODEL_PATH, 'ranker.pt'), map_location='cpu',
                               weights_only=True)
            model.load_state_dict(state)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device).eval()
            _state.update(model=model, tokenizer=tokenizer, device=device)
            logger.info('Span ranker loaded from %s on %s', MODEL_PATH, device)
        except Exception:
            logger.exception('Span ranker unavailable; stage B keeps its own spans')
            return False
    return True


def warmup():
    """Load the checkpoint during startup instead of inside the first request."""
    if MODEL_PATH and _load():
        logger.info('Span ranker warm')


def predict_spans(words, questions, answers, deadline=None):
    """{question number: (start, end)} for answered questions; {} when unavailable."""
    if not any(answers) or not _load():
        return {}
    if deadline is not None and time.monotonic() > deadline:
        logger.warning('Span ranker skipped: past the request deadline')
        return {}
    try:
        import torch

        from tools.evidence_training.data import render_words
        from tools.evidence_training.ranker import make_features, predict

        context, ranges = render_words(words)
        records = []
        for index, question in enumerate(questions):
            if not answers[index]:
                continue
            # The record id is the question number, so the result maps straight back.
            records.append({'id': str(index + 1), 'question': question, 'context': context,
                            'word_chars': ranges, 'words': words, 'label': False,
                            'gold_words': None})
        tokenizer, model, device = _state['tokenizer'], _state['model'], _state['device']
        features = make_features(records, tokenizer, MAX_LENGTH, STRIDE)
        autocast = None
        if device == 'cuda' and torch.cuda.is_bf16_supported():
            autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
        chosen = predict(model, records, features, tokenizer, device, BATCH_SIZE, autocast,
                         deadline)
        spans = {}
        for key, selection in chosen.items():
            if selection['score'] < MIN_SCORE:
                continue
            first, last = selection['words']
            spans[int(key)] = (words[first]['start'], words[last]['end'])
        return spans
    except Exception:
        logger.exception('Span ranker failed; stage B keeps its own spans')
        return {}
