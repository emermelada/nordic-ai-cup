import json
import unittest

from dtos import ASRQuestionResponseDto
from pipeline.evidence import align_quote
from pipeline.stage_b import (
    align_bridged,
    apply_quotes,
    build_refine_messages,
    build_window_messages,
    clean_window_output,
    draft_quotes,
    focus_sentences,
    parse_refine_output,
    sentence_ranges,
    sentence_text,
)


def make_words(text, step=0.5, gap_after=None):
    words = []
    clock = 0.0
    for index, token in enumerate(text.split()):
        words.append({'word': ' ' + token, 'start': clock, 'end': clock + step * 0.8})
        clock += step
        if gap_after and index in gap_after:
            clock += 2.0
    return words


TRANSCRIPT = ('How have you been feeling? Honestly, fine. Good. Your blood pressure is normal, '
              'and your feet look fine. Any side effects? None. Then the kidney test. It came back normal.')


class SentenceTableTest(unittest.TestCase):
    def test_splits_at_punctuation_and_pauses(self):
        words = make_words('one two three four five six', gap_after={2})
        self.assertEqual(sentence_ranges(words), [(0, 2), (3, 5)])
        words = make_words(TRANSCRIPT)
        texts = [sentence_text(words, span) for span in sentence_ranges(words)]
        self.assertEqual(texts[:3], ['How have you been feeling?', 'Honestly, fine.', 'Good.'])
        self.assertEqual(texts[-1], 'It came back normal.')

    def test_empty_transcript(self):
        self.assertEqual(sentence_ranges([]), [])

    def test_focus_covers_anchor_context_and_lexical_hits(self):
        words = make_words(TRANSCRIPT)
        sentences = sentence_ranges(words)
        anchor_span = sentences[-1]
        anchor = (words[anchor_span[0]]['start'], words[anchor_span[1]]['end'])
        focus = focus_sentences(words, sentences, anchor, 'Was the blood pressure normal?')
        self.assertIn(len(sentences) - 1, focus)
        self.assertIn(len(sentences) - 3, focus)
        pressure = next(k for k, span in enumerate(sentences) if 'pressure' in sentence_text(words, span))
        self.assertIn(pressure, focus)
        self.assertEqual(focus, sorted(set(focus)))

    def test_focus_without_anchor_uses_lexical_hits_only(self):
        words = make_words(TRANSCRIPT)
        sentences = sentence_ranges(words)
        focus = focus_sentences(words, sentences, None, 'Was the kidney test normal?')
        self.assertTrue(focus)
        self.assertTrue(all(0 <= k < len(sentences) for k in focus))


class PromptTest(unittest.TestCase):
    def test_refine_prompt_lists_only_yes_questions_with_drafts(self):
        words = make_words(TRANSCRIPT)
        raw = json.dumps({'results': [
            {'q': 1, 'answer': 'yes', 'quote': 'Your blood pressure is normal,\nand your feet look fine.'},
            {'q': 2, 'answer': 'no'},
        ]})
        drafts = draft_quotes(raw, 2)
        self.assertEqual(drafts, {1: 'Your blood pressure is normal, and your feet look fine.'})
        messages = build_refine_messages(words, ['BP normal?', 'Fever?'], [True, False], drafts)
        self.assertEqual(messages[0]['role'], 'system')
        user = messages[1]['content']
        self.assertIn('[0] How have you been feeling?', user)
        self.assertIn('1. BP normal?\n   draft: "Your blood pressure is normal, and your feet look fine."', user)
        self.assertNotIn('2. Fever?', user)

    def test_window_prompt_marks_gaps(self):
        filler = ' '.join(f'Filler sentence number {n}.' for n in range(12))
        words = make_words('Morning doctor. ' + filler + ' Then the kidney test. It came back normal.')
        sentences = sentence_ranges(words)
        first = (words[0]['start'], words[sentences[0][1]]['end'])
        messages = build_window_messages(words, 'Was the kidney test normal?', first, sentences)
        user = messages[1]['content']
        self.assertTrue(user.startswith('QUESTION: Was the kidney test normal?\n[0] Morning doctor.'))
        self.assertIn('\n...\n', user)
        self.assertNotIn('number 6', user)
        self.assertTrue(user.endswith('It came back normal.\nEVIDENCE:'))


class OutputTest(unittest.TestCase):
    def test_parse_drops_duplicates_and_invalid_entries(self):
        raw = '{"results":[{"q":1,"quote":"a b"},{"q":2,"quote":"c d"},{"q":2,"quote":"e f"},{"q":9,"quote":"x"},{"q":3}]}'
        self.assertEqual(parse_refine_output(raw, 5), {1: 'a b'})
        self.assertEqual(parse_refine_output('', 5), {})
        self.assertEqual(parse_refine_output('not json', 5), {})

    def test_clean_window_output(self):
        self.assertEqual(clean_window_output('<think>\nx\n</think>\n\n"No fever."\n'), 'No fever.')
        self.assertEqual(clean_window_output(None), '')
        self.assertEqual(len(clean_window_output('w ' * 200).split()), 60)

    def test_bridged_alignment_spans_a_skipped_sentence(self):
        words = make_words('Do you know of any exposure? Anyone around you who has had it? No, none that I know of.')
        quote = 'Do you know of any exposure? No, none that I know of.'
        span = align_bridged(words, quote)
        self.assertEqual(span, (words[0]['start'], words[-1]['end']))

    def test_bridged_alignment_keeps_single_match_for_distant_parts(self):
        words = make_words('The dose is 100 mg. ' + 'filler ' * 40 + 'Take it after a meal.')
        quote = 'The dose is 100 mg. Take it after a meal.'
        span = align_bridged(words, quote)
        self.assertEqual(span, align_quote(words, quote))
        self.assertTrue(span is None or span[1] - span[0] < 5.0)

    def test_bridged_alignment_falls_back_for_unmatched_parts(self):
        words = make_words('The dose is 100 mg. Take it after a meal.')
        quote = 'The dose is 100 mg. Completely unrelated words here.'
        self.assertEqual(align_bridged(words, quote), align_quote(words, quote))
        self.assertIsNone(align_bridged(words, None))
        self.assertIsNone(align_bridged(words, 'zzz qqq'))

    def test_apply_quotes_replaces_only_aligned_retained_yes(self):
        words = make_words(TRANSCRIPT)
        response = ASRQuestionResponseDto(
            answers=[True, False, True],
            evidence_start=[0.0, None, 1.0], evidence_end=[0.5, None, 1.5],
        )
        quotes = {1: 'It came back normal.', 2: 'Honestly, fine.', 3: 'zzz qqq'}
        result = apply_quotes(quotes, response, words, duration=60.0)
        self.assertEqual(result.answers, response.answers)
        last = sentence_ranges(words)[-1]
        self.assertAlmostEqual(result.evidence_start[0], words[last[0]]['start'])
        self.assertAlmostEqual(result.evidence_end[0], words[last[1]]['end'])
        self.assertIsNone(result.evidence_start[1])
        self.assertEqual((result.evidence_start[2], result.evidence_end[2]), (1.0, 1.5))

    def test_apply_quotes_ignores_out_of_range_and_expired_deadline(self):
        words = make_words(TRANSCRIPT)
        response = ASRQuestionResponseDto(answers=[True], evidence_start=[0.0], evidence_end=[0.5])
        result = apply_quotes({0: 'Good.', 2: 'Good.'}, response, words, duration=60.0)
        self.assertEqual((result.evidence_start[0], result.evidence_end[0]), (0.0, 0.5))
        result = apply_quotes({1: 'Good.'}, response, words, duration=60.0, deadline=0.0)
        self.assertEqual((result.evidence_start[0], result.evidence_end[0]), (0.0, 0.5))


if __name__ == '__main__':
    unittest.main()
