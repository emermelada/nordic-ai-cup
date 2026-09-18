"""What an answering model hands back, and the answerer to use when there is none.

An answerer takes one transcript and all of its questions and returns, per
question, how likely the answer is yes plus where it read that from. It does not
make the final yes/no call or cut the span: ``pipeline.finalize`` does both, the
same way for every answerer, so the offline harness and the live endpoint score
identical logic.

    def answer(transcript, questions, deadline) -> List[Optional[Answer]]

``deadline`` is a ``time.monotonic()`` value. An answerer should return by
then, leaving ``None`` for any question it did not get to; those fall back to
``lexical``. Select one with ``ANSWERER=module:function`` (default
``answering:lexical``).

``lexical`` is the floor, not a contender: no model, just word overlap. Measured
with the annotators' own transcript it finds the right sentence about half the
time. It exists so a failing model still leaves a scoreable answer behind.
"""

import importlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from asr import Transcript
from evidence import best_sentence

# With nothing to go on, yes and no are equally likely: both splits are exactly
# balanced. What decides the call is the threshold in ``pipeline``.
UNINFORMED = 0.5


@dataclass
class Answer:
    """One question's answer, as probability and provenance."""

    p_yes: float
    lines: Optional[Tuple[int, int]] = None  # sentence numbers, inclusive
    quote: Optional[str] = None  # verbatim from the transcript


Answerer = Callable[[Transcript, List[str], float], List[Optional[Answer]]]


def lexical(transcript: Transcript, questions: List[str], deadline: float) -> List[Optional[Answer]]:
    """Word overlap only.

    A question that shares no content word with anything said is what an
    off-topic question looks like, so that one is a confident no. Everything
    else is a coin toss pointing at the best-overlapping sentence — which the
    threshold turns into a yes, because a yes on a real positive can also earn
    evidence credit while a no can only ever earn the answer.
    """
    answers = []

    for question in questions:
        sentence, overlap = best_sentence(transcript, question)

        if sentence is None or overlap == 0:
            answers.append(Answer(p_yes=0.0))
        else:
            answers.append(Answer(p_yes=UNINFORMED, lines=(sentence, sentence)))

    return answers


def load(spec: str) -> Answerer:
    """``'module:function'`` -> the function."""
    module_name, _, function_name = spec.partition(':')
    return getattr(importlib.import_module(module_name), function_name)
