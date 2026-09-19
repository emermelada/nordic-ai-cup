"""The evidence vote: medoid selection, and that it stays off unless asked for."""

import os
import unittest
from unittest.mock import patch

from pipeline.evidence import medoid_span, vote_response
from tests.test_core import make_words, response


class MedoidTest(unittest.TestCase):
    def test_picks_the_span_the_others_cluster_around(self):
        # Two producers near 10-12, one far away: the outlier cannot win.
        self.assertEqual(medoid_span([(50.0, 52.0), (10.0, 12.0), (10.5, 12.5)]), (10.0, 12.0))

    def test_single_and_paired_candidates_keep_the_first(self):
        self.assertEqual(medoid_span([(1.0, 2.0)]), (1.0, 2.0))
        self.assertEqual(medoid_span([(1.0, 2.0), (30.0, 31.0)]), (1.0, 2.0))


class VoteResponseTest(unittest.TestCase):
    def test_two_agreeing_producers_move_a_selected_span(self):
        selected = response([True, True], [50.0, 5.0], [52.0, 6.0])
        second = {1: (10.0, 12.0), 2: (5.0, 6.0)}
        third = {1: (10.4, 12.4), 2: (5.1, 6.1)}
        voted, changed = vote_response(selected, [second, third])
        self.assertEqual(changed, 1)
        self.assertEqual(voted.evidence_start, [10.0, 5.0])
        self.assertEqual(voted.evidence_end, [12.0, 6.0])
        self.assertEqual(voted.answers, [True, True])

    def test_fewer_than_three_candidates_changes_nothing(self):
        selected = response([True], [50.0], [52.0])
        voted, changed = vote_response(selected, [{1: (10.0, 12.0)}, {}])
        self.assertEqual(changed, 0)
        self.assertEqual(voted, selected)

    def test_a_no_answer_is_never_given_a_span(self):
        selected = response([False], [None], [None])
        voted, changed = vote_response(selected, [{1: (10.0, 12.0)}, {1: (10.1, 12.1)}])
        self.assertEqual(changed, 0)
        self.assertEqual(voted.evidence_start, [None])
        self.assertEqual(voted.answers, [False])

    def test_the_input_response_is_not_mutated(self):
        selected = response([True], [50.0], [52.0])
        before = selected.model_copy(deep=True)
        vote_response(selected, [{1: (10.0, 12.0)}, {1: (10.4, 12.4)}])
        self.assertEqual(selected, before)


class ExtractorTest(unittest.TestCase):
    def test_no_checkpoint_means_no_producer_and_no_import_cost(self):
        from pipeline import extractor

        with patch.dict(os.environ, {}, clear=False):
            self.assertEqual(extractor.predict_spans(make_words('a b c'), ['q?'], [True]), {})

    def test_a_broken_checkpoint_disables_the_producer_instead_of_raising(self):
        from pipeline import extractor

        with patch.object(extractor, 'MODEL_PATH', '/nonexistent/checkpoint'), \
                patch.dict(extractor._state, {'attempted': False, 'model': None}, clear=False):
            self.assertEqual(extractor.predict_spans(make_words('a b c'), ['q?'], [True]), {})


class BackendVoteFlagTest(unittest.TestCase):
    """The second producer exists only when MEDICAL_EVIDENCE_VOTE is set."""

    def frame(self, vote):
        from pipeline import mlx_backend

        words = make_words('the patient takes one tablet every morning before breakfast today')
        backend = object.__new__(mlx_backend.MLXBackend)
        seen = {}

        def generate_many(prompts, max_tokens, request_started):
            seen['keys'] = sorted(key for key, _ in prompts)
            return {key: 'one tablet every morning' for key, _ in prompts}

        backend.generate_many = generate_many
        with patch.object(mlx_backend, 'EVIDENCE_MODE', 'perq'), \
                patch.object(mlx_backend, 'EVIDENCE_MODEL', 'model'), \
                patch.object(mlx_backend, 'EVIDENCE_VOTE', vote):
            result = backend.complete_evidence(words, ['Tablet daily?'], [True], {1: ''},
                                               [(None, None)], __import__('time').monotonic())
        return result, seen

    def test_off_by_default_emits_one_producer(self):
        frame, seen = self.frame(False)
        self.assertEqual(seen['keys'], [1])
        self.assertNotIn('vote_outputs', frame)
        self.assertEqual(sorted(frame['outputs']), [1])

    def test_on_emits_a_second_producer_under_positive_keys(self):
        frame, seen = self.frame(True)
        self.assertEqual(seen['keys'], [-1, 1])
        self.assertEqual(sorted(frame['outputs']), [1])
        self.assertEqual(sorted(frame['vote_outputs']), [1])


if __name__ == '__main__':
    unittest.main()
