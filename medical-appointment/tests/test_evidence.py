import unittest
from unittest.mock import patch

from pipeline.core import answer_response, build_messages, floor_response
from pipeline.evidence import (
    align_quote, build_compact_messages, build_focused_messages, consensus_response,
    energy_envelope, refine_evidence,
)
from tests.test_core import make_words, raw_results, response
from tools.compare_evidence import split_conversations


class EvidenceTests(unittest.TestCase):
    def test_focused_prompt_preserves_questions_and_boolean_rules(self):
        words = make_words('The examination was normal.')
        questions = ['Were abnormal findings reported?', 'Was fever absent?']
        messages = build_focused_messages(words, questions)
        self.assertEqual(messages[1], build_messages(words, questions)[1])
        self.assertIn('A near-miss is "no"', messages[0]['content'])
        self.assertIn('Abbreviations', messages[0]['content'])
        self.assertIn('Never quote an unanswered question', messages[0]['content'])

    def test_compact_prompt_keeps_legacy_rules_and_parses_minified_output(self):
        words = make_words('The dose is 100 mg daily.')
        messages = build_compact_messages(words, ['Is the dose 100 mg?', 'Is it 200 mg?'])
        self.assertEqual(messages[1], build_messages(words, ['Is the dose 100 mg?', 'Is it 200 mg?'])[1])
        self.assertIn('A near-miss is "no"', messages[0]['content'])
        self.assertIn('give the unit id(s)', messages[0]['content'])
        self.assertIn('minified JSON', messages[0]['content'])
        raw = '{"results":[{"q":1,"answer":"yes","units":[0],"quote":"The dose is 100 mg daily."},{"q":2,"answer":"no"}]}'
        actual = answer_response(raw, words, ['a', 'b'], 30, alignment='numeric')
        self.assertEqual(actual, response([True, False], [words[0]['start'], None], [words[-1]['end'], None]))

    def test_earliest_exact_repeat_is_preserved(self):
        words = make_words('Your examination was normal. Your examination was normal.')
        self.assertEqual(align_quote(words, 'Your examination was normal.'),
                         (words[0]['start'], words[3]['end']))

    def test_decimal_can_span_multiple_whisper_words(self):
        words = make_words('The value was 7 .0 millimoles per liter.')
        words[4]['word'] = '.0'
        self.assertEqual(align_quote(words, 'The value was 7.0 millimoles per liter.'),
                         (words[0]['start'], words[-1]['end']))
        self.assertEqual(align_quote(words, '7.0'), (words[3]['start'], words[4]['end']))

    def test_decimal_and_numeric_near_misses_are_not_equivalent(self):
        words = make_words('Take 25 milligrams daily. Take 2.5 milligrams daily.')
        self.assertEqual(align_quote(words, 'Take 2.5 milligrams daily.'),
                         (words[4]['start'], words[7]['end']))
        self.assertIsNone(align_quote(words, 'Take 250 milligrams daily.'))
        self.assertEqual(align_quote(words, 'Take 2.50 milligrams daily.'),
                         (words[4]['start'], words[7]['end']))

    def test_compounds_and_apostrophes(self):
        words = make_words("Your long-term sugar isn't elevated.")
        self.assertEqual(align_quote(words, 'Your long term sugar isn’t elevated.'),
                         (words[0]['start'], words[-1]['end']))

    def test_empty_invalid_and_expired_inputs(self):
        words = make_words('There is no fever.')
        for quote in ('', '!!!', None, {}, []):
            self.assertIsNone(align_quote(words, quote))
        self.assertIsNone(align_quote([], 'fever'))
        self.assertIsNone(align_quote(words, 'There is no fever.', deadline=0))

    def test_new_alignment_is_explicit_and_does_not_change_answer(self):
        words = make_words('The dose is 25 milligrams. The dose is 2.5 milligrams.')
        raw = raw_results({'q': 1, 'answer': 'yes', 'quote': 'The dose is 2.5 milligrams.', 'units': [1]},
                          {'q': 2, 'answer': 'no', 'quote': 'The dose is 2.5 milligrams.', 'units': [1]})
        legacy = answer_response(raw, words, ['a', 'b'], 30)
        numeric = answer_response(raw, words, ['a', 'b'], 30, alignment='numeric')
        self.assertEqual(legacy.answers, numeric.answers)
        self.assertEqual(legacy.evidence_start, [words[0]['start'], None])
        self.assertEqual(numeric.evidence_start, [words[5]['start'], None])
        self.assertEqual(numeric.evidence_end, [words[-1]['end'], None])
        self.assertEqual(answer_response('', [], ['a'], 30, alignment='numeric'), floor_response(1))
        with self.assertRaises(ValueError):
            answer_response(raw, words, ['a', 'b'], 30, alignment='unknown')

    def test_deadline_during_search_discards_partial_match(self):
        words = make_words('one two three four five six seven eight nine ten')
        with patch('pipeline.evidence._expired', side_effect=[False] * 15 + [True]), \
                patch('pipeline.evidence.fuzz.ratio', return_value=80) as ratio:
            self.assertIsNone(align_quote(words, 'one two three four', deadline=1))
        self.assertGreater(ratio.call_count, 0)

    def test_numeric_mode_keeps_parser_fallbacks_and_clamps_spans(self):
        words = make_words('The dose is 2.5 milligrams.')
        fallback = response([False, True], [None, 2], [None, 3])
        raw = raw_results({'q': 1, 'answer': 'yes', 'quote': 'The dose is 2.5 milligrams.'},
                          {'q': 2, 'answer': 'no'}, {'q': 2, 'answer': 'yes'})
        actual = answer_response(raw, words, ['Dose?', 'Other?'], 2.1, fallback, alignment='numeric')
        self.assertEqual(actual, response([True, True], [1, 2], [2.1, 2.1]))
        self.assertEqual(answer_response(raw, words, ['a', 'b'], 20, fallback,
                                         deadline=0, alignment='numeric'), fallback)
        repaired = answer_response('```json\n{"results":[{"q":1,"answer":"no"},]}\n```',
                                   words, ['a', 'b'], 20, fallback, alignment='numeric')
        self.assertEqual(repaired, fallback)

    def test_consensus_takes_the_span_the_referee_agrees_with(self):
        primary = response([True, True, False], [10.0, 30.0, None], [12.0, 32.0, None])
        secondary = response([True, True, True], [20.0, 30.5, 5.0], [22.0, 32.5, 6.0])
        referee = response([True, True, True], [20.2, 10.0, 5.0], [22.2, 12.0, 6.0])
        actual = consensus_response(primary, secondary, referee)
        # Question 1: the referee backs the secondary span; question 2: it backs neither
        # clearly, so the primary stands. Answers always come from the primary.
        self.assertEqual(actual.evidence_start, [20.0, 30.0, None])
        self.assertEqual(actual.evidence_end, [22.0, 32.0, None])
        self.assertEqual(actual.answers, [True, True, False])
        missing = response([True], [10.0], [12.0])
        blank = response([True], [None], [None])
        self.assertEqual(consensus_response(missing, blank, referee), missing)

    def test_quoted_question_extends_to_short_reply(self):
        words = make_words('Any swelling? No swelling either. Is the rash itchy? It is itchy only when I sit still at night.')
        spans = response([True, True, True, False],
                         [words[0]['start'], words[5]['start'], words[0]['start'], None],
                         [words[1]['end'], words[8]['end'], words[4]['end'], None])
        actual = refine_evidence(spans, words)
        self.assertEqual(actual.evidence_end, [words[4]['end'], words[8]['end'], words[4]['end'], None])
        self.assertEqual(actual.evidence_start, spans.evidence_start)
        self.assertEqual(spans.evidence_end[0], words[1]['end'])

    def test_quoted_question_does_not_attach_another_question(self):
        words = make_words('Any swelling? Did you bring paperwork? No, I forgot it.')
        spans = response([True], [words[0]['start']], [words[1]['end']])
        actual = refine_evidence(spans, words)
        self.assertEqual(actual, spans)
        self.assertEqual(refine_evidence(actual, words), spans)

    def test_quoted_question_does_not_cross_a_long_pause(self):
        for pause_before in (2, 3):
            with self.subTest(pause_before=pause_before):
                words = make_words('Any swelling? No swelling either.')
                for word in words[pause_before:]:
                    word['start'] += 5
                    word['end'] += 5
                spans = response([True], [words[0]['start']], [words[1]['end']])
                self.assertEqual(refine_evidence(spans, words), spans)

    def test_start_moves_to_audible_speech_within_the_span(self):
        words = make_words('Your heart sounds normal.', start=1.0, step=0.5)
        envelope = [-90.0] * 130 + [-20.0] * 200
        spans = response([True, True], [1.0, 1.0], [3.0, 1.2])
        actual = refine_evidence(spans, words, envelope)
        self.assertEqual(actual.evidence_start, [1.3, 1.0])
        self.assertEqual(actual.evidence_end, [3.0, 1.2])
        self.assertEqual(refine_evidence(spans, words, [-90.0] * 400).evidence_start, [1.0, 1.0])
        self.assertEqual(refine_evidence(spans, words, [-20.0] * 400).evidence_start, [1.0, 1.0])

    def test_onset_does_not_skip_a_quiet_first_word(self):
        words = make_words('No fever today.')
        envelope = [-50.0] * 130 + [-20.0] * 200
        for start in (words[0]['start'], words[0]['start'] + 0.1):
            with self.subTest(start=start):
                spans = response([True], [start], [words[-1]['end']])
                self.assertEqual(refine_evidence(spans, words, envelope), spans)
        words[0]['end'] = words[0]['start']
        spans = response([True], [words[0]['start']], [words[-1]['end']])
        self.assertEqual(refine_evidence(spans, words, envelope), spans)

    def test_onset_uses_first_selected_word_not_preceding_word(self):
        words = make_words('Listen. Your heart sounds normal.', start=0.5, step=0.5)
        envelope = [-90.0] * 130 + [-20.0] * 200
        spans = response([True], [words[1]['start']], [words[-1]['end']])
        actual = refine_evidence(spans, words, envelope)
        self.assertEqual(actual.evidence_start, [1.3])
        self.assertEqual(refine_evidence(actual, words, envelope), actual)

    def test_energy_envelope_is_ten_millisecond_dbfs(self):
        envelope = energy_envelope([0.5] * 160 + [0.0] * 160 + [0.5] * 100)
        self.assertEqual(len(envelope), 2)
        self.assertAlmostEqual(envelope[0], -6.0, places=1)
        self.assertLess(envelope[1], -150)

    def test_conversation_split_is_stable_disjoint_and_complete(self):
        conversations = [(f'{i}.mp3', [{'transcript_id': f'sample_{i}'}] * 10) for i in range(39)]
        dev = split_conversations(conversations, 'dev')
        holdout = split_conversations(conversations, 'holdout')
        self.assertEqual(len(dev), 30)
        self.assertEqual(len(holdout), 9)
        self.assertEqual(set(name for name, _ in dev) & set(name for name, _ in holdout), set())
        self.assertEqual(set(name for name, _ in dev + holdout), set(name for name, _ in conversations))
        self.assertEqual(split_conversations(list(reversed(conversations)), 'holdout'), list(reversed(holdout)))


if __name__ == '__main__':
    unittest.main()
