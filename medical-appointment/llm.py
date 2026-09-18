"""The answerer: a local LLM server, asked one question at a time.

Talks to any OpenAI-compatible chat server on this machine — vLLM or
llama.cpp's ``llama-server`` — so nothing leaves the box, as the rules require.
Select it with ``ANSWERER=llm:answer``. PLAN.md has the serving commands.

    LLM_BASE_URL    http://127.0.0.1:8080/v1
    LLM_MODEL       the name the server knows the model by (vLLM checks it)
    LLM_THINKING    0 (default): answer directly; 1: let a hybrid model reason
                    first; omit: send no template switch at all
    LLM_PARALLEL    10: questions in flight at once

Every question gets its own request with the same prefix — instructions, worked
examples, then the transcript — so the server's prompt cache pays for that once
per conversation and the ten requests run side by side.

The model replies in four labelled lines: the sentences it read, the shortest
verbatim quote that settles the question, a one-line check, and yes or no.
The quote becomes the span (``evidence.locate``); the probability the model put
on "yes" at the answer token becomes ``p_yes``, so the pipeline's threshold
works on a real confidence rather than a coin that only ever lands 0 or 1.

The worked examples are real training excerpts, verbatim from the ``base``
transcript. The conversations they came from are listed in
``FEW_SHOT_SOURCES``; ``eval_offline.py`` leaves those out of the score.
"""

import bisect
import concurrent.futures
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

from answering import Answer
from asr import Transcript

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get('LLM_BASE_URL', 'http://127.0.0.1:8080/v1').rstrip('/')
MODEL = os.environ.get('LLM_MODEL', 'local')
THINKING = os.environ.get('LLM_THINKING', '0')
PARALLEL = int(os.environ.get('LLM_PARALLEL', '10'))
MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '2048' if THINKING == '1' else '300'))

# Conversations the worked examples below are taken from.
FEW_SHOT_SOURCES = (
    'sample_4', 'sample_5', 'sample_17', 'sample_18', 'sample_19', 'sample_20', 'sample_71',
)

SYSTEM_PROMPT = """\
You check yes/no questions against the transcript of a recorded consultation \
between a GP and a patient.

The transcript is automatic speech recognition: one numbered sentence per line, \
both speakers mixed together. Numbers and ordinary words are reliable. Drug and \
brand names are often misspelled by sound — "Ibu Medin" is Ibumetin, "pan top \
resolve" is pantoprazole, "Active L" is Activelle, "Aromere" is Airomir — so \
match names by how they sound.

For each question:
1. Find the line or lines that bear on it.
2. Compare every detail of the question with what was actually said: drug, dose, \
unit, frequency, duration, body site, timing, test value, who said it, and \
whether it was actually done or agreed rather than only mentioned or asked \
about. The same topic with any detail different is NO. A subject that never \
comes up is NO. Statement-style questions ("..., right?", "..., didn't it?") \
are checked the same way as any other.
3. Answer yes only if the transcript establishes it.

Reply in exactly this format, four lines, nothing else:
LINES: <first>-<last>, or none
QUOTE: <the shortest verbatim excerpt that establishes the answer, or none>
CHECK: <one short sentence comparing the question with the transcript>
ANSWER: <yes or no>

The QUOTE is copied character for character from the transcript, misspellings \
included, and may run across consecutive lines. Keep it minimal: start at the \
beginning of the clause that states the fact, leaving out lead-ins such as \
"So,", "Then", "Yes,", "That is right,", "It means" or "Putting it together,". \
Stop right after the last word the fact needs, leaving out trailing clauses \
that add nothing. When the fact is established by an exchange — a question and \
its answer, or a list and its confirmation — quote the whole exchange."""

# (excerpt, question, reply). Verbatim base-model transcript lines, with the
# line numbers they have in their own conversation.
_EXAMPLES = [
    (
        '[55] Yes.\n'
        '[56] I am creating prescriptions for both Pamol and Ibu Medin now, and they will be waiting for you at the pharmacy.\n'
        '[57] That is a weight off my mind.',
        'Is Pamol one of the medicines requested?',
        'LINES: 56-56\n'
        'QUOTE: I am creating prescriptions for both Pamol\n'
        'CHECK: Pamol is one of the two medicines being prescribed.\n'
        'ANSWER: yes',
    ),
    (
        '[55] Yes.\n'
        '[56] I am creating prescriptions for both Pamol and Ibu Medin now, and they will be waiting for you at the pharmacy.\n'
        '[57] That is a weight off my mind.',
        'Were both prescriptions issued?',
        'LINES: 56-56\n'
        'QUOTE: I am creating prescriptions for both\n'
        'CHECK: The doctor is creating both prescriptions.\n'
        'ANSWER: yes',
    ),
    (
        '[18] What does the treatment involve?\n'
        '[19] Sporanox, 100 milligrams daily for two weeks.\n'
        '[20] You take it after a meal.',
        'Should the daily dose be 100 mg?',
        'LINES: 19-19\n'
        'QUOTE: 100 milligrams daily\n'
        'CHECK: The dose is 100 milligrams daily.\n'
        'ANSWER: yes',
    ),
    (
        '[12] This is the chronic pain since your neck operation.\n'
        '[13] That is right, the cervical disc prolapse operated in 2017.\n'
        '[14] And honestly, the pain has just carried on regardless.',
        'Did the patient undergo surgery for a cervical disc prolapse in 2017?',
        'LINES: 13-13\n'
        'QUOTE: the cervical disc prolapse operated in 2017.\n'
        'CHECK: The cervical disc prolapse was operated on in 2017.\n'
        'ANSWER: yes',
    ),
    (
        '[10] All right.\n'
        '[11] And you are still taking Pantoprazole alongside it?\n'
        '[12] Yes, every day with it.\n'
        '[13] Good.',
        'Is the patient also taking Pantoprazole?',
        'LINES: 11-12\n'
        'QUOTE: And you are still taking Pantoprazole alongside it? Yes, every day with it.\n'
        'CHECK: Asked whether they still take Pantoprazole, the patient confirms.\n'
        'ANSWER: yes',
    ),
    (
        '[35] Which prescriptions will I get?\n'
        '[36] Active L, Aromere and Esomeprizol.\n'
        '[37] All three renewed.\n'
        '[38] The Esomeprizol is the one for the reflux?',
        'Is Airomir among the renewed medicines?',
        'LINES: 36-37\n'
        'QUOTE: Active L, Aromere and Esomeprizol. All three renewed.\n'
        'CHECK: "Aromere" is Airomir, and all three were renewed.\n'
        'ANSWER: yes',
    ),
    (
        '[23] This is what a stable annual check looks like.\n'
        '[24] So, what happens with my treatment?\n'
        '[25] No changes.\n'
        '[26] You carry on exactly as you are.\n'
        '[27] Nothing at all.',
        'Will the current treatment continue unchanged?',
        'LINES: 24-26\n'
        'QUOTE: what happens with my treatment? No changes. You carry on exactly as you are.\n'
        'CHECK: The doctor says there are no changes and the patient carries on as before.\n'
        'ANSWER: yes',
    ),
    (
        '[27] I was worried it might have drifted since last time.\n'
        '[28] Your LDL cholesterol is 2.2 millimoles per liter.\n'
        '[29] Right.',
        'Is the LDL cholesterol 4.2 mmol/L?',
        'LINES: 28-28\n'
        'QUOTE: Your LDL cholesterol is 2.2 millimoles per liter.\n'
        'CHECK: The LDL is 2.2 mmol/L, not 4.2.\n'
        'ANSWER: no',
    ),
    (
        '[31] The one you always check for me?\n'
        '[32] Your creatinine is normal.\n'
        '[33] Well, that is all three of them behaving themselves.',
        'Did the creatinine turn out to be elevated?',
        'LINES: 32-32\n'
        'QUOTE: Your creatinine is normal.\n'
        'CHECK: The creatinine is normal, not elevated.\n'
        'ANSWER: no',
    ),
    (
        '[13] I have almost run out of my painkillers and I was hoping to have the prescriptions renewed.\n'
        '[14] Which painkillers are we talking about?\n'
        '[15] I would like to hear you name them yourself.\n'
        '[16] Pamol and Ibu Metin.',
        'Did the patient ask for morphine to be renewed?',
        'LINES: 13-16\n'
        'QUOTE: Pamol and Ibu Metin.\n'
        'CHECK: The patient asked for Pamol and Ibumetin; morphine is never mentioned.\n'
        'ANSWER: no',
    ),
    (
        '[27] I was worried it might have drifted since last time.\n'
        '[28] Your LDL cholesterol is 2.2 millimoles per liter.\n'
        '[29] Right.',
        'Was the patient discussing their pet?',
        'LINES: none\n'
        'QUOTE: none\n'
        'CHECK: Pets are never discussed.\n'
        'ANSWER: no',
    ),
]

_LINES = re.compile(r'^\s*\**LINES\**\s*:\s*(.*)$', re.IGNORECASE | re.MULTILINE)
_QUOTE = re.compile(r'^\s*\**QUOTE\**\s*:\s*(.*)$', re.IGNORECASE | re.MULTILINE)
_ANSWER = re.compile(r'ANSWER\**\s*:\s*\**\s*(yes|no)\b', re.IGNORECASE)
_THINK = re.compile(r'<think>.*?</think>', re.DOTALL)

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL)


def _user_message(transcript_text: str, question: str, excerpt: bool = False) -> str:
    label = 'TRANSCRIPT (excerpt)' if excerpt else 'TRANSCRIPT'
    return f'{label}:\n{transcript_text}\n\nQUESTION: {question}'


def _messages(transcript_text: str, question: str) -> List[Dict[str, str]]:
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for excerpt, example_question, reply in _EXAMPLES:
        messages.append({'role': 'user', 'content': _user_message(excerpt, example_question, excerpt=True)})
        messages.append({'role': 'assistant', 'content': reply})
    messages.append({'role': 'user', 'content': _user_message(transcript_text, question)})
    return messages


def _payload(messages: List[Dict[str, str]]) -> dict:
    payload = {
        'model': MODEL,
        'messages': messages,
        'temperature': 0.0,
        'max_tokens': MAX_TOKENS,
        'logprobs': True,
        'top_logprobs': 5,
    }
    if THINKING in ('0', '1'):
        payload['chat_template_kwargs'] = {'enable_thinking': THINKING == '1'}
    return payload


def _parse(content: str) -> Optional[Tuple[Optional[Tuple[int, int]], Optional[str], bool]]:
    """(lines, quote, said_yes) from a reply, or None if it has no answer."""
    content = _THINK.sub('', content)
    answers = _ANSWER.findall(content)
    if not answers:
        return None

    lines = None
    match = _LINES.search(content)
    if match:
        numbers = [int(n) for n in re.findall(r'\d+', match.group(1))]
        if numbers:
            lines = (min(numbers), max(numbers))

    quote = None
    match = _QUOTE.search(content)
    if match:
        quote = match.group(1).strip().strip('"“”\'').strip()
        if not quote or quote.lower() in ('none', 'n/a', '-'):
            quote = None

    return lines, quote, answers[-1].lower() == 'yes'


def _p_yes(choice: dict) -> Optional[float]:
    """Probability of "yes" at the final answer token, from the returned logprobs.

    Rebuilds the text from the tokens themselves rather than trusting
    ``content`` offsets, so it also works when a server splits reasoning out.
    """
    tokens = ((choice.get('logprobs') or {}).get('content')) or []
    if not tokens:
        return None

    text, ends = '', []
    for token in tokens:
        text += token.get('token') or ''
        ends.append(len(text))

    matches = list(_ANSWER.finditer(text))
    if not matches:
        return None

    position = bisect.bisect_right(ends, matches[-1].start(1))
    if position >= len(tokens):
        return None

    yes = no = 0.0
    for alternative in tokens[position].get('top_logprobs') or []:
        word = (alternative.get('token') or '').strip().strip('"*').lower()
        probability = math.exp(alternative.get('logprob', -math.inf))
        if word.startswith('y'):
            yes += probability
        elif word.startswith('n'):
            no += probability

    return yes / (yes + no) if yes + no > 0 else None


def _answer_one(transcript_text: str, question: str, timeout: float) -> Optional[Answer]:
    try:
        response = requests.post(
            f'{BASE_URL}/chat/completions',
            json=_payload(_messages(transcript_text, question)),
            timeout=timeout,
        )
        response.raise_for_status()
        choice = response.json()['choices'][0]
    except Exception as exc:
        logger.warning('LLM request failed for %r: %s', question, exc)
        return None

    parsed = _parse(choice.get('message', {}).get('content') or '')
    if parsed is None:
        logger.warning('No answer in LLM reply for %r', question)
        return None

    lines, quote, said_yes = parsed
    p_yes = _p_yes(choice)
    return Answer(
        p_yes=float(said_yes) if p_yes is None else p_yes,
        lines=lines,
        quote=quote,
    )


def answer(transcript: Transcript, questions: List[str], deadline: float) -> List[Optional[Answer]]:
    """All questions in parallel; whatever is not back by the deadline is None."""
    transcript_text = transcript.render()
    timeout = max(deadline - time.monotonic() - 0.5, 1.0)

    futures = [_pool.submit(_answer_one, transcript_text, q, timeout) for q in questions]
    done, _ = concurrent.futures.wait(futures, timeout=max(deadline - time.monotonic(), 0.0))
    return [future.result() if future in done else None for future in futures]


def _warm_up() -> None:
    """One real request at import, so the first scored conversation is not the slowest."""
    try:
        requests.get(f'{BASE_URL}/models', timeout=3).raise_for_status()
    except Exception as exc:
        logger.warning('No LLM server at %s (%s): answers fall back to lexical until there is.',
                       BASE_URL, exc)
        return

    excerpt, question, _ = _EXAMPLES[0]
    started = time.monotonic()
    result = _answer_one(excerpt, question, timeout=120)
    logger.info('LLM warm (%s at %s) in %.1fs: %s', MODEL, BASE_URL,
                time.monotonic() - started, result)


_warm_up()
