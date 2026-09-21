import json
import os
import time
import unittest
from unittest.mock import Mock, patch

from pipeline import mlx_backend
from pipeline.locate import locate_evidence, parse_range, question_outputs, word_spans
from pipeline.stage_b import apply_evidence, sentence_ranges
from tests.test_core import response
from tests.test_stage_b import make_words


def location(first, last):
    return json.dumps({'first_sentence': first, 'last_sentence': last})


def boundaries(first, last):
    return json.dumps({'start_word': first, 'end_word': last})


class RangeTests(unittest.TestCase):
    def test_zero_based_inclusive_ranges_and_single_word(self):
        self.assertEqual(parse_range(boundaries(0, 0), 'start_word', 'end_word', 3), (0, 0))
        self.assertEqual(parse_range(boundaries(0, 2), 'start_word', 'end_word', 3), (0, 2))

    def test_code_fences_and_stripped_reasoning(self):
        text = '<think>not the answer</think>\n```json\n' + boundaries(1, 2) + '\n```'
        self.assertEqual(parse_range(text, 'start_word', 'end_word', 3), (1, 2))

    def test_invalid_or_ambiguous_ranges_are_not_repaired(self):
        values = [None, '', 'not JSON', '[]', '{}', '{"start_word":0}',
                  boundaries(True, 2), boundaries('0', 2), boundaries(0.0, 2),
                  boundaries(-1, 2), boundaries(2, 1), boundaries(0, 3),
                  '{"start_word":0,"start_word":1,"end_word":2}',
                  boundaries(0, 1) + boundaries(1, 2)]
        for value in values:
            with self.subTest(value=value):
                self.assertIsNone(parse_range(value, 'start_word', 'end_word', 3))
        self.assertIsNone(parse_range(boundaries(0, 0), 'start_word', 'end_word', 0))

    def test_duplicate_question_ids_remain_invalid(self):
        outputs = {1: 'first', '01': 'second', '1': 'third', '2': 'keep', '3': 'outside'}
        self.assertEqual(question_outputs(outputs, 2), {1: None, 2: 'keep'})
        self.assertEqual(question_outputs(None, 2), {})
        self.assertEqual(question_outputs({True: 'not a question ID'}, 2), {})


class LocateTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'MEDICAL_ASR_MODE': 'base'})
        env.start()
        self.addCleanup(env.stop)
        self.words = make_words('Earlier context. Test was normal. Between mentions. Test was normal. No changes.')
        self.sentences = sentence_ranges(self.words)
        self.questions = ['Was the test normal?', 'Was fever present?', 'Were there no changes?']
        self.answers = [True, False, True]
        self.original = response(self.answers, [0.0, None, 0.0], [0.4, None, 0.4])

    def generate(self):
        return Mock(side_effect=[
            {1: location(3, 3), 3: location(4, 4)},
            {1: boundaries(*self.sentences[3]), 3: boundaries(*self.sentences[4])},
        ])

    def test_two_phases_only_for_retained_yes_and_absolute_word_ids(self):
        generate = self.generate()
        frame = locate_evidence(self.words, self.questions, self.answers, generate)
        self.assertEqual([call.args[0] for call in generate.call_args_list], ['locate', 'bounds'])
        for call in generate.call_args_list:
            self.assertEqual([qid for qid, _ in call.args[1]], [1, 3])
        expected_window = [self.sentences[2][0], self.sentences[4][1]]
        self.assertEqual(frame['windows'][1], expected_window)
        text = generate.call_args_list[1].args[1][0][1][1]['content']
        self.assertIn(f'[w{expected_window[0]}]', text)
        self.assertIn(f'[w{expected_window[1]}]', text)
        self.assertNotIn('[w0]', text)
        self.assertNotIn('Earlier context.', text)

    def test_repeated_text_selects_the_later_occurrence_without_output_padding(self):
        frame = locate_evidence(self.words, self.questions, self.answers, self.generate())
        selected = apply_evidence(frame, self.original, self.words, 60.0, extend_replies=False)
        first, last = self.sentences[3]
        self.assertEqual(selected.answers, self.answers)
        self.assertEqual((selected.evidence_start[0], selected.evidence_end[0]),
                         (self.words[first]['start'], self.words[last]['end']))
        self.assertGreater(selected.evidence_start[0], self.words[self.sentences[1][0]]['start'])
        self.assertIsNone(selected.evidence_start[1])
        self.assertEqual(self.original.evidence_start, [0.0, None, 0.0])

    def test_invalid_locations_never_launch_boundary_requests(self):
        generate = Mock(return_value={1: 'not JSON', 3: location(0, len(self.sentences))})
        frame = locate_evidence(self.words, self.questions, self.answers, generate)
        generate.assert_called_once()
        self.assertEqual(frame['windows'], {})
        self.assertEqual(frame['outputs'], {})
        self.assertEqual(apply_evidence(frame, self.original, self.words, 60.0), self.original)

    def test_one_bad_location_does_not_discard_another_valid_question(self):
        generate = Mock(side_effect=[{1: location(3, 3), 3: 'bad'},
                                     {1: boundaries(*self.sentences[3])}])
        frame = locate_evidence(self.words, self.questions, self.answers, generate)
        self.assertEqual([qid for qid, _ in generate.call_args_list[1].args[1]], [1])
        selected = apply_evidence(frame, self.original, self.words, 60.0, extend_replies=False)
        self.assertEqual(selected.evidence_start[0], self.words[self.sentences[3][0]]['start'])
        self.assertEqual((selected.evidence_start[2], selected.evidence_end[2]), (0.0, 0.4))

    def test_context_is_clipped_at_transcript_edges(self):
        generate = Mock(side_effect=[{1: location(0, len(self.sentences) - 1)},
                                     {1: boundaries(0, len(self.words) - 1)}])
        frame = locate_evidence(self.words, ['Whole passage?'], [True], generate)
        self.assertEqual(frame['windows'], {1: [0, len(self.words) - 1]})

    def test_no_words_or_no_positive_answers_require_no_model(self):
        generate = Mock(side_effect=AssertionError('No generation expected'))
        self.assertEqual(locate_evidence([], ['Fact?'], [True], generate)['outputs'], {})
        self.assertEqual(locate_evidence(self.words, ['Fact?'], [False], generate)['outputs'], {})
        generate.assert_not_called()

    def test_expired_deadline_prevents_both_phases(self):
        generate = self.generate()
        frame = locate_evidence(self.words, self.questions, self.answers, generate, deadline=0.0)
        self.assertEqual(frame['skipped'], 'budget')
        generate.assert_not_called()

    def test_deadline_after_location_prevents_the_second_phase(self):
        generate = self.generate()
        with patch('pipeline.locate._expired', side_effect=[False, True]):
            frame = locate_evidence(self.words, self.questions, self.answers, generate, deadline=123.0)
        generate.assert_called_once()
        self.assertEqual(frame['skipped'], 'budget')
        self.assertTrue(frame['locations'])
        self.assertEqual(frame['outputs'], {})

    def test_outputs_ready_after_budget_are_retained_and_marked(self):
        with patch('pipeline.locate._expired', side_effect=[False, False, False, True]):
            frame = locate_evidence(self.words, self.questions, self.answers, self.generate(), deadline=123.0)
        self.assertEqual(frame['skipped'], 'budget')
        self.assertEqual(set(frame['outputs']), {1, 3})

    def test_word_indices_must_stay_inside_the_displayed_window(self):
        start, end = self.sentences[3]
        frame = {'mode': 'locate', 'windows': {1: [start, end], 2: [start, end], 3: [start, end]},
                 'outputs': {1: boundaries(start - 1, end), 2: boundaries(start, end), 3: boundaries(start, end + 1)}}
        self.assertEqual(apply_evidence(frame, self.original, self.words, 60.0), self.original)
        self.assertEqual(word_spans(frame, self.words, 3), {1: None, 2: (self.words[start]['start'], self.words[end]['end']), 3: None})

    def test_malformed_windows_zero_length_words_and_expired_application_preserve_spans(self):
        for window in (None, '0,1', [0], [True, 1], [-1, 1], [2, 1], [0, len(self.words)]):
            with self.subTest(window=window):
                frame = {'mode': 'locate', 'windows': {1: window}, 'outputs': {1: boundaries(0, 1)}}
                self.assertEqual(apply_evidence(frame, self.original, self.words, 60.0), self.original)
        frame = {'mode': 'locate', 'windows': {1: [0, 0]}, 'outputs': {1: boundaries(0, 0)}}
        words = [{'word': ' Good.', 'start': 1.0, 'end': 1.0}]
        self.assertEqual(apply_evidence(frame, self.original, words, 60.0), self.original)
        frame = locate_evidence(self.words, self.questions, self.answers, self.generate())
        self.assertEqual(apply_evidence(frame, self.original, self.words, 60.0, deadline=0.0), self.original)

    def test_exact_timing_policy_is_preserved(self):
        words = make_words('Any pain? No.')
        frame = {'mode': 'locate', 'windows': {1: [0, 2]}, 'outputs': {1: boundaries(0, 1)}}
        original = response([True], [0.0], [0.4])
        exact = apply_evidence(frame, original, words, 5.0, extend_replies=False)
        legacy = apply_evidence(frame, original, words, 5.0, extend_replies=True)
        self.assertEqual(exact.evidence_end, [words[1]['end']])
        self.assertEqual(legacy.evidence_end, [words[2]['end']])

    def test_backend_shares_original_request_start_and_token_limit(self):
        backend = mlx_backend.MLXBackend()
        backend.generate_many = Mock(side_effect=[{1: location(3, 3), 3: location(4, 4)},
                                                 {1: boundaries(*self.sentences[3]), 3: boundaries(*self.sentences[4])}])
        started = time.monotonic()
        with patch.object(mlx_backend, 'EVIDENCE_MODE', 'locate'), patch.object(mlx_backend, 'EVIDENCE_MODEL', 'mock'):
            frame = backend.complete_evidence(self.words, self.questions, self.answers, {}, [None] * 3, started)
        self.assertEqual(frame['mode'], 'locate')
        self.assertEqual(backend.generate_many.call_count, 2)
        for call in backend.generate_many.call_args_list:
            self.assertEqual(call.args[1:], (64, started))

    def test_backend_retains_disabled_and_expired_guards(self):
        backend = mlx_backend.MLXBackend()
        backend.generate_many = Mock(side_effect=AssertionError('No generation expected'))
        with patch.object(mlx_backend, 'EVIDENCE_MODE', 'locate'), \
                patch.object(mlx_backend, 'EVIDENCE_BUDGET_SECONDS', 40), \
                patch.object(mlx_backend, 'EVIDENCE_MODEL', 'mock'):
            frame = backend.complete_evidence(self.words, self.questions, self.answers, {}, [None] * 3, time.monotonic() - 100)
            self.assertEqual(frame['skipped'], 'budget')
            self.assertIsNone(backend.complete_evidence(self.words, self.questions, [False] * 3, {}, [None] * 3, time.monotonic()))
        with patch.object(mlx_backend, 'EVIDENCE_MODEL', None):
            self.assertIsNone(backend.complete_evidence(self.words, self.questions, self.answers, {}, [None] * 3, time.monotonic()))
        backend.generate_many.assert_not_called()


if __name__ == '__main__':
    unittest.main()


class BarePairTests(unittest.TestCase):
    def test_a_two_element_array_is_read_as_inclusive_endpoints(self):
        self.assertEqual(parse_range('[31, 34]', 'first_sentence', 'last_sentence', 40), (31, 34))
        self.assertEqual(parse_range('```json\n[0, 2]\n```', 'a', 'b', 3), (0, 2))
        self.assertEqual(parse_range('<think>x</think>[1, 1]', 'a', 'b', 3), (1, 1))

    def test_malformed_or_out_of_range_pairs_are_rejected(self):
        for raw in ('[31]', '[1, 2, 3]', '[3, 1]', '[0, 99]', '["a", "b"]', '[1.5, 2]', '[]'):
            self.assertIsNone(parse_range(raw, 'a', 'b', 10), raw)
