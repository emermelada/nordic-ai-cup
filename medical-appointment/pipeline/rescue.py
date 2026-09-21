"""Give the questions answered "no" a second, calibrated hearing.

Scoring is asymmetric and the answer pass does not know it. On a 190-question attempt a
missed positive costs its accuracy mark *and* its tIoU share, about 0.0067 raw, while a
false positive costs the accuracy mark alone, about 0.0021 - the miss is 3.2x worse. The
break-even confidence for answering yes is therefore near 0.24, not 0.5.

That gap is not academic here. On the 39 training conversations the answer pass is perfect,
but the captured hidden-set attempts returned 91, 91 and 92 yes answers where both platform
sets are exactly half yes (95 of 190), so at least three or four positives were being
missed on unseen conversations. Four misses alone account for the difference between the
0.832 measured locally and the 0.809 the platform returned.

So each "no" is re-asked on its own, and the probability of yes is read from the reply
token's logprobs. A "no" flips only when that probability clears ``YES_THRESHOLD``; a "yes"
is never touched, so the decisions that are already right cannot be disturbed. The default
threshold is 0.5, which changes nothing: lower it only once the training distribution shows
how many true negatives sit above the new value.
"""

import os

# A re-ask that reads the same garbled text repeats the same mistake: the two positives the
# compact prompt misses score 0.031 and 0.138 here. Independence has to come from somewhere
# else - the transcript's sound-alike spellings, or a more accurate transcript entirely.
PHONETIC = (
    ' Drug, brand and condition names are written by sound: "Aromere"/"Aromir" is Airomir, '
    '"Ibu Medin" is Ibumetin, "Active L" is Activelle, "Isomeprazole" is Esomeprazole, '
    '"panadil" is Panodil, "Malaskam Contegiosum" is molluscum contagiosum, "Condro Malaysia '
    'Pateli" is chondromalacia patellae. A name spelled differently but pronounced the same is '
    'the SAME thing, so judge it as established.'
)

SYSTEM = (
    'You check one claim against the transcript of a doctor-patient consultation, given as '
    'numbered sentences.\n'
    'Answer with one word, yes or no: does the transcript establish the claim?\n'
    'Say yes when the transcript states it or clearly implies it, even if the wording differs '
    'from the question, a name is spelled by sound, or the fact is split across a question and '
    'its reply. Say no when the transcript is silent about it, or states something different: '
    'another dose, value, body part, drug, or the opposite polarity.'
)
SYSTEM_PHONETIC = SYSTEM + PHONETIC


def system_prompt():
    return SYSTEM_PHONETIC if os.environ.get('MEDICAL_RESCUE_PHONETIC') == '1' else SYSTEM

# Break-even is about 0.24; the default disables the rescue so it is opted into deliberately.
YES_THRESHOLD = float(os.environ.get('MEDICAL_RESCUE_THRESHOLD', '0.5'))


def build_rescue_messages(transcript_text, question):
    """One claim against the whole transcript; the reply is a single yes or no token."""
    user = f'TRANSCRIPT\n{transcript_text}\n\nCLAIM: {question}\n\nDoes the transcript establish it?'
    return [{'role': 'system', 'content': system_prompt()}, {'role': 'user', 'content': user}]


def pending(answers):
    """The 1-based question numbers currently answered no."""
    return [index + 1 for index, answer in enumerate(answers) if not answer]


def flip_answers(answers, scores, threshold=None):
    """Answers with each sufficiently confident "no" turned into a yes.

    ``scores`` maps a 1-based question number to P(yes). Missing, unparsable or
    insufficiently confident entries leave the answer alone, and a yes is never revisited.
    """
    threshold = YES_THRESHOLD if threshold is None else threshold
    flipped = list(answers)
    changed = []
    for number, probability in sorted((scores or {}).items()):
        try:
            index = int(number) - 1
            value = float(probability)
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(flipped) or flipped[index]:
            continue
        if value >= threshold:
            flipped[index] = True
            changed.append(index + 1)
    return flipped, changed


def apply_rescue(response, scores, threshold=None):
    """A copy of ``response`` with rescued answers; a flipped question starts with no span."""
    flipped, changed = flip_answers(response.answers, scores, threshold)
    if not changed:
        return response, []
    result = response.model_copy(deep=True)
    result.answers = flipped
    for number in changed:
        result.evidence_start[number - 1] = result.evidence_end[number - 1] = None
    return result, changed
