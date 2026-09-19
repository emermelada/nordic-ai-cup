import unittest
from unittest.mock import patch

from dtos import ASRQuestionResponseDto
from pipeline import rescue
from tests.test_stage_b import make_words

TRANSCRIPT = 'Any fever? No fever today. The dose is 100 mg daily.'


def response(answers, spans=None):
    spans = spans or [(0.0, 1.0)] * len(answers)
    return ASRQuestionResponseDto(
        answers=list(answers),
        evidence_start=[s[0] if a else None for a, s in zip(answers, spans)],
        evidence_end=[s[1] if a else None for a, s in zip(answers, spans)],
    )


class PromptTests(unittest.TestCase):
    def test_prompt_states_the_claim_and_asks_for_one_word(self):
        messages = rescue.build_rescue_messages('[0] Any fever?', 'Is the patient free of fever?')
        self.assertIn('one word, yes or no', messages[0]['content'])
        self.assertIn('CLAIM: Is the patient free of fever?', messages[1]['content'])
        self.assertIn('[0] Any fever?', messages[1]['content'])

    def test_pending_lists_only_the_no_answers_one_based(self):
        self.assertEqual(rescue.pending([True, False, False, True]), [2, 3])
        self.assertEqual(rescue.pending([True, True]), [])
        self.assertEqual(rescue.pending([]), [])


class FlipTests(unittest.TestCase):
    def test_a_confident_no_flips_and_a_yes_is_never_revisited(self):
        answers, changed = rescue.flip_answers([True, False, False], {2: 0.9, 3: 0.1}, threshold=0.5)
        self.assertEqual(answers, [True, True, False])
        self.assertEqual(changed, [2])
        answers, changed = rescue.flip_answers([True, False], {1: 0.99}, threshold=0.5)
        self.assertEqual((answers, changed), ([True, False], []))

    def test_threshold_is_inclusive_and_configurable(self):
        self.assertEqual(rescue.flip_answers([False], {1: 0.24}, threshold=0.24)[0], [True])
        self.assertEqual(rescue.flip_answers([False], {1: 0.23}, threshold=0.24)[0], [False])
        with patch.object(rescue, 'YES_THRESHOLD', 0.2):
            self.assertEqual(rescue.flip_answers([False], {1: 0.3})[0], [True])
        with patch.object(rescue, 'YES_THRESHOLD', 0.5):
            self.assertEqual(rescue.flip_answers([False], {1: 0.3})[0], [False])

    def test_unusable_scores_leave_answers_alone(self):
        for scores in ({}, None, {1: None}, {1: 'x'}, {'x': 0.9}, {9: 0.9}, {0: 0.9}, {-1: 0.9}):
            self.assertEqual(rescue.flip_answers([False, False], scores, threshold=0.1)[0],
                             [False, False], scores)
        self.assertEqual(rescue.flip_answers([False], {'1': 0.9}, threshold=0.5)[0], [True])

    def test_apply_rescue_clears_the_span_of_a_flipped_answer_only(self):
        original = response([True, False, False], spans=[(1.0, 2.0)] * 3)
        result, changed = rescue.apply_rescue(original, {2: 0.9, 3: 0.1}, threshold=0.5)
        self.assertEqual(changed, [2])
        self.assertEqual(result.answers, [True, True, False])
        self.assertEqual((result.evidence_start[0], result.evidence_end[0]), (1.0, 2.0))
        self.assertIsNone(result.evidence_start[1])
        self.assertEqual(original.answers, [True, False, False])

    def test_no_change_returns_the_same_object_untouched(self):
        original = response([True, False])
        result, changed = rescue.apply_rescue(original, {2: 0.1}, threshold=0.5)
        self.assertIs(result, original)
        self.assertEqual(changed, [])


class BackendTests(unittest.TestCase):
    def test_only_the_no_questions_are_scored_and_the_scorer_is_reused(self):
        from pipeline import mlx_backend
        backend = mlx_backend.MLXBackend()
        words = make_words(TRANSCRIPT)
        with patch.object(mlx_backend, 'EVIDENCE_MODEL', 'model'), \
                patch.object(backend, 'score_yes', return_value={2: 0.8}) as scorer:
            scores = backend.complete_rescue(words, ['Fever?', 'Dose?'], [True, False], 1e18)
        self.assertEqual(scores, {2: 0.8})
        prompts = scorer.call_args.args[0]
        self.assertEqual([key for key, _ in prompts], [2])
        self.assertIn('CLAIM: Dose?', prompts[0][1][1]['content'])
        self.assertIn('[2] The dose is 100 mg daily.', prompts[0][1][1]['content'])

    def test_nothing_to_rescue_or_no_model_skips_the_call(self):
        from pipeline import mlx_backend
        backend = mlx_backend.MLXBackend()
        words = make_words(TRANSCRIPT)
        with patch.object(backend, 'score_yes') as scorer:
            with patch.object(mlx_backend, 'EVIDENCE_MODEL', 'model'):
                self.assertEqual(backend.complete_rescue(words, ['q'], [True], 1e18), {})
            with patch.object(mlx_backend, 'EVIDENCE_MODEL', None):
                self.assertEqual(backend.complete_rescue(words, ['q'], [False], 1e18), {})
        scorer.assert_not_called()

    def test_the_local_backend_cannot_score_so_nothing_is_rescued(self):
        from pipeline import mlx_backend
        backend = mlx_backend.MLXBackend()
        with patch.object(mlx_backend, 'EVIDENCE_MODEL', 'model'):
            self.assertEqual(backend.complete_rescue(make_words(TRANSCRIPT), ['q'], [False], 1e18), {})

class VetoInteractionTests(unittest.TestCase):
    def test_a_rescued_answer_is_not_put_to_the_secondary_vote(self):
        from pipeline.evidence import primary_evidence_response
        primary = response([True, True], spans=[(1.0, 2.0), (3.0, 4.0)])
        secondary = response([False, False])
        vetoed = primary_evidence_response(primary, secondary)
        self.assertEqual(vetoed.answers, [False, False])
        kept = primary_evidence_response(primary, secondary, exempt=[2])
        self.assertEqual(kept.answers, [False, True])
        self.assertEqual((kept.evidence_start[1], kept.evidence_end[1]), (3.0, 4.0))
        self.assertEqual(primary_evidence_response(primary, secondary, exempt=['1', 2]).answers,
                         [True, True])
class LexicalFallbackTests(unittest.TestCase):
    def test_the_best_overlapping_sentence_is_chosen(self):
        from pipeline.stage_b import lexical_span
        words = make_words('Morning doctor. The dose is 100 mg daily. Take it after a meal.')
        span = lexical_span(words, 'Should the daily dose be 100 mg?')
        text = ' '.join(w['word'].strip() for w in words
                        if w['end'] > span[0] + 1e-6 and w['start'] < span[1] - 1e-6)
        self.assertEqual(text, 'The dose is 100 mg daily.')

    def test_no_transcript_or_no_content_words_gives_nothing(self):
        from pipeline.stage_b import lexical_span
        self.assertIsNone(lexical_span([], 'Was the dose 100 mg?'))
        self.assertIsNone(lexical_span(make_words('The dose is 100 mg.'), 'the and of'))
        self.assertIsNone(lexical_span(make_words('Zebra giraffe.'), 'Was the dose 100 mg?'))
class ReadingSourceTests(unittest.TestCase):
    def test_the_rescue_prefers_a_more_accurate_reading_when_the_backend_offers_one(self):
        from pipeline import mlx_backend
        backend = mlx_backend.MLXBackend()
        words = make_words(TRANSCRIPT)
        backend.rescue_reading = '[0] Any fever? [1] No fever today.'
        with patch.object(mlx_backend, 'EVIDENCE_MODEL', 'model'), \
                patch.object(backend, 'score_yes', return_value={1: 0.9}) as scorer:
            backend.complete_rescue(words, ['Fever?'], [False], 1e18)
        sent = scorer.call_args.args[0][0][1][1]['content']
        self.assertIn('[1] No fever today.', sent)
        self.assertNotIn('100 mg', sent)

    def test_without_one_it_reads_the_served_transcript(self):
        from pipeline import mlx_backend
        backend = mlx_backend.MLXBackend()
        with patch.object(mlx_backend, 'EVIDENCE_MODEL', 'model'), \
                patch.object(backend, 'score_yes', return_value={1: 0.9}) as scorer:
            backend.complete_rescue(make_words(TRANSCRIPT), ['Dose?'], [False], 1e18)
        self.assertIn('100 mg', scorer.call_args.args[0][0][1][1]['content'])

if __name__ == '__main__':
    unittest.main()
