"""Rejected evidence prompt ablations, kept outside the serving path."""

import re

from pipeline.core import SYSTEM, SYN, _answer, _question_id, _results, build_messages, split_units
from pipeline.evidence import EVIDENCE_RULES


GROUNDED_SYSTEM = SYSTEM[:SYSTEM.index('- For every "yes"')] + '''
The transcript is automatic speech recognition: recognize clinical synonyms, translated medical wording and phonetic spellings from context, but never change a number or a negation.
For EACH question, FIRST find and quote the evidence, THEN decide the answer:
- Find the explicit statement most closely matching the question's specific wording. Prefer the actual clinical finding, diagnosis or agreed plan over an indirect implication, a generic confirmation or a conversational recap.
- Copy the shortest contiguous clause containing the queried fact. Do not copy extra sentences or other items in a list. Keep the relevant subject, value and negation. Include a preceding question only when its short reply would otherwise lack a subject.
- For a near-miss, quote the actual fact that contradicts the claim and answer "no". If the topic never occurs, use an empty quote and answer "no". An unanswered question is not evidence of yes.
- Give the numbered unit ids containing the quote. Copy exactly, including the transcript's spelling. Do not paraphrase.
Example: transcript [0] I am sleeping well, but my appetite is poor.
Question: Is appetite reduced? -> {"q":1,"units":[0],"quote":"my appetite is poor.","answer":"yes"}
Question: Is sleep poor? -> {"q":2,"units":[0],"quote":"I am sleeping well,","answer":"no"}
Return JSON only: {"results":[{"q":1,"units":[12],"quote":"exact relevant clause","answer":"yes"}, ...]}'''


def build_grounded_messages(words, questions):
    pattern = r'\b(' + '|'.join(re.escape(key) for key in SYN) + r')\b'
    expanded = [re.sub(pattern, lambda match: SYN[match.group().lower()], question, flags=re.I)
                for question in questions]
    messages = build_messages(words, expanded)
    messages[0] = {'role': 'system', 'content': GROUNDED_SYSTEM}
    return messages


def build_refinement_messages(words, questions, raw):
    positive_ids = {
        _question_id(entry.get('q'), len(questions)) for entry in _results(raw)
        if isinstance(entry, dict) and _answer(entry.get('answer')) is True
    } - {None}
    transcript = '\n'.join(
        f'[{i}] {"".join(word["word"] for word in unit).strip()}'
        for i, unit in enumerate(split_units(words))
    )
    qtext = '\n'.join(f'{i}. {questions[i - 1]}' for i in sorted(positive_ids))
    return [
        {'role': 'system', 'content': (
            'Locate supporting evidence for the listed claims in a doctor-patient consultation. '
            'A previous pass has already judged these questions YES. Your job is ONLY to locate '
            'the most direct evidence, not to answer again.\n' + EVIDENCE_RULES
            + '\nReturn JSON only: {"results":[{"q":1,"units":[12],"quote":"exact supporting clause"}, ...]}'
        )},
        {'role': 'user', 'content': f'TRANSCRIPT\n{transcript}\n\nSUPPORTED QUESTIONS\n{qtext}\n\nJSON:'},
    ]
