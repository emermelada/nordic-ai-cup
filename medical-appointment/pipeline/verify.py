"""Pick the evidence span by verifying candidates instead of generating one.

The annotation is the *shortest passage that establishes the fact on its own*. Every pass
so far asked a model to produce that passage in one shot, which leaves it to guess both
where the passage is and how far it runs. This stage instead enumerates the contiguous
passages around the pass's pick, asks the model about each one independently — "does this
passage, on its own, establish the answer?" — and keeps the shortest that clears the bar.

Scoring each candidate in its own request is what makes this different from the earlier
lettered multiple-choice attempt, which lost to position bias: there is no list to be
biased along, and the reply is one token whose probability is read from the logprobs.
"""

from pipeline.base_asr import sentence_ranges

RADIUS_SENTENCES = 2
MAX_JOINED_SENTENCES = 3
CLAUSE_MARKS = (',', ';', ':')
COORDINATORS = frozenset({'and', 'but', 'with', 'which', 'so', 'while', 'because'})
# Below this the passage is not taken to stand on its own.
YES_THRESHOLD = 0.5
# A shorter candidate has to be at least this confident to displace a longer, surer one.
MIN_CONFIDENCE = 0.5

SYSTEM = (
    'You are shown one passage from the transcript of a doctor-patient consultation, and one '
    'question about that consultation whose answer is known to be yes.\n'
    'Answer with one word, yes or no: does this passage ON ITS OWN establish that answer, for a '
    'reader who sees nothing else of the conversation?\n'
    'Say no when the passage only hints at it, when it is about a different detail, value or body '
    'part, or when what it says depends on words that are not in it, such as a pronoun or a bare '
    '"yes" whose subject is missing.'
)


def _clause_cuts(words, first, last):
    """Sub-spans of one sentence, split at commas and coordinators."""
    spans = {(first, last)}
    marks = [i for i in range(first, last) if words[i]['word'].strip().endswith(CLAUSE_MARKS)]
    marks += [i - 1 for i in range(first + 1, last + 1)
              if words[i]['word'].strip().lower().strip(',.;:') in COORDINATORS]
    for mark in sorted(set(marks)):
        if mark - first >= 1:
            spans.add((first, mark))
        if last - mark >= 2:
            spans.add((mark + 1, last))
    return spans


def candidate_spans(words, span, sentences=None, radius=RADIUS_SENTENCES,
                    max_joined=MAX_JOINED_SENTENCES):
    """Contiguous word ranges around ``span``: whole sentences, runs of them, and clause cuts."""
    sentences = sentences if sentences is not None else sentence_ranges(words)
    if not sentences or span is None or span[0] is None:
        return []
    touched = [k for k, (a, b) in enumerate(sentences)
               if words[b]['end'] > span[0] + 1e-6 and words[a]['start'] < span[1] - 1e-6]
    if not touched:
        return []
    low = max(0, touched[0] - radius)
    high = min(len(sentences) - 1, touched[-1] + radius)
    spans = set()
    for i in range(low, high + 1):
        for j in range(i, min(i + max_joined, high + 1)):
            spans.add((sentences[i][0], sentences[j][1]))
        spans |= _clause_cuts(words, *sentences[i])
    return sorted(spans)


def passage_text(words, span):
    return ' '.join(word['word'].strip() for word in words[span[0]:span[1] + 1])


def build_verify_messages(words, span, question):
    user = (f'PASSAGE:\n{passage_text(words, span)}\n\n'
            f'QUESTION: {question}\n\n'
            'Does the passage on its own establish that the answer is yes?')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]


def select_span(candidates, anchor=None, threshold=YES_THRESHOLD):
    """The shortest candidate that clears ``threshold``; ``anchor`` when none does.

    ``candidates`` is an iterable of ``(span, probability)``. Length is counted in words, so
    the convention the annotation follows — minimal but self-contained — decides directly.
    """
    passing = [(span, p) for span, p in candidates if p is not None and p >= threshold]
    if not passing:
        best = max(((span, p) for span, p in candidates if p is not None),
                   key=lambda item: item[1], default=None)
        if anchor is not None or best is None or best[1] < MIN_CONFIDENCE:
            return anchor
        return best[0]
    return min(passing, key=lambda item: (item[0][1] - item[0][0], -item[1]))[0]


def probability_of_yes(tokens):
    """P(yes) at the first yes/no token of a logprobs payload, or None if there is none."""
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = (token.get('token') or '').strip().strip('"*').lower()
        if not (text.startswith('y') or text.startswith('n')):
            continue
        yes = no = 0.0
        import math

        for alternative in token.get('top_logprobs') or []:
            word = (alternative.get('token') or '').strip().strip('"*').lower()
            probability = math.exp(alternative.get('logprob', -math.inf))
            if word.startswith('y'):
                yes += probability
            elif word.startswith('n'):
                no += probability
        if yes + no > 0:
            return yes / (yes + no)
        return 1.0 if text.startswith('y') else 0.0
    return None
