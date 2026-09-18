import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from pipeline import boundary
from tests.test_core import response
from utils import temporal_iou, validate_response


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'model.json'
        self.words = [{'word': ' ' + word, 'start': i * .5, 'end': (i + 1) * .5}
                      for i, word in enumerate('You should take 100 mg every morning. Yes. Call if needed.'.split())]
        self.baseline = response([True, False, True], [1., None, None], [3.5, None, None])
        self.primary = response([True, True, True], [1., 1., None], [3.5, 3.5, None])
        self.questions = ['Should the dose be 100 mg?', 'Is fever present?', 'Was travel discussed?']
        self.schema = {
            'schema_version': 1, 'feature_names': list(boundary.FEATURE_NAMES),
            'node_fields': boundary.NODE_FIELDS, 'baseline': .25,
            'trees': [[[0, 1., 1, 2, 0.], [-1, 0., 0, 0, 2.], [-1, 0., 0, 0, -3.]],
                      [[-1, 0., 0, 0, .5]]],
        }

    def load(self, data=None):
        self.path.write_text(json.dumps(self.schema if data is None else data))
        return boundary.load_model(self.path)

    def adjust(self, **kwargs):
        args = dict(response=self.baseline, words=self.words, questions=self.questions,
                    duration=10., envelope=None, primary=self.primary, secondary=None, referee=None)
        args.update(kwargs)
        return boundary.adjust_boundaries(**args)

    def test_tree_threshold_equality_and_additive_leaf_values(self):
        matrix = np.zeros((3, len(boundary.FEATURE_NAMES)))
        matrix[:, 0] = [0., 1., 1.001]
        np.testing.assert_array_equal(boundary.predict_scores(matrix, self.load()), [2.75, 2.75, -2.25])
        self.assertEqual(boundary.predict_scores(matrix[:0], self.load()).shape, (0,))

    def test_model_schema_and_tree_validation(self):
        cases = []
        for key, value in (('schema_version', 2), ('schema_version', True),
                           ('feature_names', list(reversed(boundary.FEATURE_NAMES))),
                           ('node_fields', []), ('baseline', float('nan')), ('baseline', True),
                           ('trees', []), ('trees', self.schema['trees'] * 129)):
            case = copy.deepcopy(self.schema)
            case[key] = value
            cases.append(case)
        for node in ([0, 1., 0, 2, 0.], [0, 1., 1, 1, 0.], [0, 1., 1, 3, 0.],
                     [36, 1., 1, 2, 0.], [True, 1., 1, 2, 0.], [0, float('inf'), 1, 2, 0.],
                     [0, 1., 1., 2, 0.], [-1, 0., 0, 0, 0.], [0, 1.]):
            case = copy.deepcopy(self.schema)
            case['trees'][0][0] = node
            cases.append(case)
        cases.extend([[], None])
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.path.write_text(json.dumps(case))
                boundary.load_model(self.path)
        self.path.write_text(' ' * (1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            boundary.load_model(self.path)

    def test_invalid_features_and_prediction_deadline(self):
        model = self.load()
        for matrix in (np.zeros(36), np.zeros((2, 35)),
                       np.full((1, 36), float('nan')), np.full((1, 36), float('inf'))):
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                boundary.predict_scores(matrix, model)
        with self.assertRaises(TimeoutError):
            boundary.predict_scores(np.zeros((1, 36)), model, deadline=0)

    def test_adjustment_preserves_answers_inputs_and_primary_passage(self):
        before = copy.deepcopy((self.baseline, self.primary, self.words, self.questions))
        trace = {}
        actual = self.adjust(trace=trace)
        self.assertEqual(trace['boundary']['status'], 'completed')
        self.assertEqual(actual.answers, self.baseline.answers)
        self.assertIsNone(actual.evidence_start[1])
        self.assertIsNone(actual.evidence_end[1])
        self.assertIsNone(actual.evidence_start[2])
        self.assertIsNone(actual.evidence_end[2])
        anchor, selected = boundary.interval(self.primary, 0), boundary.interval(actual, 0)
        self.assertGreater(temporal_iou(anchor, selected), 0)
        self.assertTrue(all(abs(a - b) <= boundary.RADIUS_SECONDS for a, b in zip(anchor, selected)))
        self.assertEqual((self.baseline, self.primary, self.words, self.questions), before)
        self.assertEqual(self.adjust(), actual)
        validate_response(actual, 3)

    def test_ties_prefer_first_candidate_and_distant_passages_are_excluded(self):
        constant = copy.deepcopy(self.schema)
        constant['trees'] = [[[-1, 0., 0, 0, 0.]]]
        context = boundary.prepare_context(self.words, 30., None)
        proposals = [(1., 3.5), (20., 22.), None]
        spans, _ = boundary.candidate_spans(context, self.questions[0], proposals)
        self.assertEqual(spans[0], proposals[0])
        for span in spans:
            self.assertGreater(temporal_iou(span, proposals[0]), 0)
            self.assertTrue(all(abs(a - b) <= 2 for a, b in zip(span, proposals[0])))
        with mock.patch.object(boundary, 'MODEL', self.load(constant)):
            self.assertEqual(self.adjust(), self.baseline)

    def test_only_selected_word_onsets_are_used_without_reply_refinement(self):
        words = [{'word': word, 'start': float(i), 'end': float(i + 1)}
                 for i, word in enumerate(('Dose?', ' Yes.', ' Call.'))]
        baseline = response([True], [0.], [2.])
        short = copy.deepcopy(self.schema)
        short['trees'] = [[[0, math.log1p(1.), 1, 2, 0.],
                           [-1, 0., 0, 0, 2.], [-1, 0., 0, 0, 0.]]]
        with mock.patch.object(boundary, 'MODEL', self.load(short)):
            actual = self.adjust(response=baseline, primary=baseline, words=words, questions=['Dose?'])
        self.assertEqual(boundary.interval(actual, 0), (0., 1.))
        self.assertEqual(boundary.interval(baseline, 0), (0., 2.))

    def test_empty_and_invalid_transcripts_preserve_baseline(self):
        variants = [{'words': []}, {'duration': 0.}, {'duration': float('nan')},
                    {'words': [dict(self.words[0], start=float('nan'))]},
                    {'words': self.words * (boundary.MAX_WORDS // len(self.words) + 1)}]
        for variant in variants:
            trace = {}
            with self.subTest(variant=variant.keys()), self.assertLogs('pipeline.boundary', level='WARNING'):
                self.assertIs(self.adjust(trace=trace, **variant), self.baseline)
            self.assertEqual(trace['boundary']['status'], 'skipped')

    def test_stopword_only_transcript_has_finite_features(self):
        words = [dict(word, word=' the') for word in self.words]
        context = boundary.prepare_context(words, 10., None)
        self.assertIsNone(context['bm25'])
        proposals = [(1., 3.5), None, None]
        spans, indices = boundary.candidate_spans(context, 'the', proposals)
        matrix = boundary.span_features(context, 'the', proposals, spans, indices)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(self.adjust(words=words).answers, self.baseline.answers)

    def test_deadline_and_missing_model_preserve_baseline(self):
        with self.assertLogs('pipeline.boundary', level='WARNING'):
            self.assertIs(self.adjust(deadline=0), self.baseline)
        with mock.patch.object(boundary, 'BUDGET_SECONDS', 0), \
                self.assertLogs('pipeline.boundary', level='WARNING'):
            self.assertIs(self.adjust(), self.baseline)
        with mock.patch.object(boundary, 'MODEL', None), \
                self.assertLogs('pipeline.boundary', level='WARNING'):
            self.assertIs(self.adjust(), self.baseline)

    def test_dense_words_and_candidate_limits_preserve_baseline(self):
        dense = [{'word': ' dose', 'start': 1., 'end': 3.5}] * 1000
        with self.assertLogs('pipeline.boundary', level='WARNING'):
            self.assertIs(self.adjust(words=dense), self.baseline)
        with mock.patch.object(boundary, 'MAX_CANDIDATES', 1), \
                self.assertLogs('pipeline.boundary', level='WARNING'):
            self.assertIs(self.adjust(), self.baseline)

    def test_text_limits_apply_before_normalization(self):
        words = [dict(self.words[0], word='x' * (boundary.MAX_TRANSCRIPT_CHARACTERS + 1))]
        with mock.patch.object(boundary, '_norm') as normalize, \
                self.assertLogs('pipeline.boundary', level='WARNING'):
            self.assertIs(self.adjust(words=words), self.baseline)
        normalize.assert_not_called()
        context = boundary.prepare_context(self.words, 10., None)
        with mock.patch.object(boundary, '_norm') as normalize, self.assertRaises(ValueError):
            boundary.candidate_spans(context, 'x' * (boundary.MAX_QUESTION_CHARACTERS + 1),
                                     [(1., 3.5), None, None])
        normalize.assert_not_called()
        with self.assertLogs('pipeline.boundary', level='WARNING'):
            self.assertIs(self.adjust(questions=['x' * (boundary.MAX_QUESTION_CHARACTERS + 1)] * 3),
                          self.baseline)

    def test_deadline_is_checked_during_query_feature_preparation(self):
        context = boundary.prepare_context(self.words, 10., None)
        proposals = [(1., 3.5), None, None]
        spans, indices = boundary.candidate_spans(context, self.questions[0], proposals)
        with mock.patch.object(boundary, '_check_deadline', side_effect=TimeoutError('Feature deadline')), \
                self.assertRaises(TimeoutError):
            boundary.span_features(context, self.questions[0], proposals, spans, indices)

    def test_late_failure_discards_partial_adjustments(self):
        baseline = response([True, True], [1., 1.], [3.5, 3.5])
        calls = []

        def scores(matrix, *args):
            calls.append(len(matrix))
            if len(calls) == 2:
                raise RuntimeError('Scoring failed')
            return np.arange(len(matrix))

        with mock.patch.object(boundary, 'predict_scores', side_effect=scores), \
                self.assertLogs('pipeline.boundary', level='WARNING'):
            actual = self.adjust(response=baseline, primary=baseline, questions=self.questions[:2])
        self.assertIs(actual, baseline)
        self.assertEqual(len(calls), 2)
        self.assertEqual(baseline.evidence_start, [1., 1.])

    def test_no_eligible_questions_do_not_build_candidates(self):
        baseline = response([False, True], [None, None], [None, None])
        with mock.patch.object(boundary, 'prepare_context') as prepare:
            actual = self.adjust(response=baseline, primary=baseline, questions=self.questions[:2])
        self.assertIs(actual, baseline)
        prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
