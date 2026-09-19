"""Read one transcript, answer in another's coordinates.

The gold spans are `base`/int8 word boundaries, but `base` is the smallest Whisper and its
text is badly garbled: "Condro Malaysia Pateli" for chondromalacia patellae, "Active L,
Aromere and Esomeprizol", "Your gate and standing are normal". The model has to pick a
passage by reading that, and a name it cannot recognise is a passage it can misjudge.

So the two jobs are split. `large-v3-turbo` supplies the text the model reads; `base`
supplies the timestamps that are returned. The two transcripts agree on about 98.6% of
words, so a word-level alignment carries a span from one to the other and costs about
0.007 of the localisation ceiling — far less than a prompt patch for each misspelling,
which only covers the garbles we happened to see in the training conversations.
"""

import difflib
import re

_TOKEN = re.compile(r'[^a-z0-9]')


def _tokens(words):
    return [_TOKEN.sub('', word['word'].lower()) for word in words]


def word_map(source_words, target_words):
    """Index map from one transcript's words to another's, by word-level alignment.

    Words that align exactly map one to one; inside a replaced or inserted run the position
    is interpolated, so a span's endpoints land inside the corresponding run rather than
    collapsing to its edge.
    """
    source, target = _tokens(source_words), _tokens(target_words)
    if not source or not target:
        return {}
    mapping = {}
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for offset in range(i2 - i1):
                mapping[i1 + offset] = j1 + offset
            continue
        width = max(1, i2 - i1)
        span = max(1, j2 - j1)
        for index in range(i1, i2):
            fraction = (index - i1) / width
            mapping[index] = min(len(target) - 1, j1 + int(fraction * span))
    return mapping


def span_indices(words, span):
    """Indices of the words a time span covers."""
    if span is None or span[0] is None:
        return []
    return [i for i, word in enumerate(words)
            if word['end'] > span[0] + 1e-6 and word['start'] < span[1] - 1e-6]


def carry_span(span, source_words, target_words, mapping=None):
    """The same passage, expressed in the target transcript's timestamps."""
    indices = span_indices(source_words, span)
    if not indices:
        return None
    mapping = mapping if mapping is not None else word_map(source_words, target_words)
    first, last = mapping.get(indices[0]), mapping.get(indices[-1])
    if first is None or last is None:
        return None
    first, last = min(first, last), max(first, last)
    return target_words[first]['start'], target_words[last]['end']
