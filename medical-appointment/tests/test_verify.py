import math
import unittest
from unittest.mock import Mock, patch

from pipeline import verify
from pipeline.base_asr import sentence_ranges
from tests.test_stage_b import make_words

TEXT = ('Any side effects at all? None. Your blood pressure is normal, and your foot status is normal. '
        'Then the kidney test. It came back normal. We continue as before.')


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.words = make_words(TEXT)
        self.sentences = sentence_ranges(self.words)

    def test_pool_covers_neighbours_runs_and_clause_cuts(self):
        target = next(s for s in self.sentences if 'pressure' in verify.passage_text(self.words, s))
        span = (self.words[target[0]]['start'], self.words[target[1]]['end'])
        spans = verify.candidate_spans(self.words, span, self.sentences)
        texts = [verify.passage_text(self.words, s) for s in spans]
        self.assertIn('Your blood pressure is normal, and your foot status is normal.', texts)
        self.assertIn('Your blood pressure is normal,', texts)
        self.assertIn('and your foot status is normal.', texts)
        self.assertIn('Any side effects at all? None.', texts)
        self.assertTrue(all(a <= b for a, b in spans))
        self.assertEqual(len(spans), len(set(spans)))

    def test_pool_is_bounded_by_the_transcript_and_empty_without_an_anchor(self):
        spans = verify.candidate_spans(self.words, (0.0, self.words[-1]['end']), self.sentences)
        self.assertTrue(all(0 <= a and b < len(self.words) for a, b in spans))
        self.assertEqual(verify.candidate_spans(self.words, None, self.sentences), [])
        self.assertEqual(verify.candidate_spans(self.words, (1e9, 1e9 + 1), self.sentences), [])
        self.assertEqual(verify.candidate_spans([], (0.0, 1.0)), [])

    def test_prompt_shows_only_the_passage_and_the_question(self):
        messages = verify.build_verify_messages(self.words, (0, 4), 'Is the patient free of side effects?')
        self.assertIn('ON ITS OWN', messages[0]['content'])
        self.assertIn('PASSAGE:\nAny side effects at all?', messages[1]['content'])
        self.assertIn('QUESTION: Is the patient free of side effects?', messages[1]['content'])
        self.assertNotIn('kidney', messages[1]['content'])


class SelectionTests(unittest.TestCase):
    def test_shortest_passing_candidate_wins(self):
        candidates = [((0, 9), 0.99), ((0, 3), 0.8), ((5, 9), 0.95), ((7, 8), 0.2)]
        self.assertEqual(verify.select_span(candidates, anchor=(0, 9)), (0, 3))

    def test_ties_on_length_go_to_the_surer_candidate(self):
        self.assertEqual(verify.select_span([((0, 3), 0.7), ((5, 8), 0.9)], anchor=None), (5, 8))

    def test_nothing_passing_keeps_the_anchor(self):
        self.assertEqual(verify.select_span([((0, 3), 0.1), ((5, 9), 0.2)], anchor=(2, 4)), (2, 4))
        self.assertIsNone(verify.select_span([], anchor=None))
        self.assertEqual(verify.select_span([((0, 3), 0.1)], anchor=None), None)
        self.assertEqual(verify.select_span([((0, 3), 0.6)], anchor=None, threshold=0.9), (0, 3))

    def test_missing_scores_are_ignored(self):
        self.assertEqual(verify.select_span([((0, 9), None), ((0, 3), 0.8)], anchor=(0, 9)), (0, 3))
        self.assertEqual(verify.select_span([((0, 9), None)], anchor=(1, 2)), (1, 2))


class ProbabilityTests(unittest.TestCase):
    def token(self, text, alternatives):
        return {'token': text, 'top_logprobs': [{'token': t, 'logprob': math.log(p)} for t, p in alternatives]}

    def test_reads_the_first_yes_no_token(self):
        tokens = [self.token('\n', [('\n', 0.9)]), self.token('yes', [('yes', 0.8), ('no', 0.2)])]
        self.assertAlmostEqual(verify.probability_of_yes(tokens), 0.8)
        tokens = [self.token('No', [('No', 0.75), ('Yes', 0.25)])]
        self.assertAlmostEqual(verify.probability_of_yes(tokens), 0.25)

    def test_falls_back_to_the_emitted_token_and_handles_junk(self):
        self.assertEqual(verify.probability_of_yes([self.token('yes', [('maybe', 0.5)])]), 1.0)
        self.assertEqual(verify.probability_of_yes([self.token('no', [])]), 0.0)
        self.assertIsNone(verify.probability_of_yes([self.token('...', [('...', 1.0)])]))
        self.assertIsNone(verify.probability_of_yes(None))
        self.assertIsNone(verify.probability_of_yes([]))


class BackendTests(unittest.TestCase):
    def test_server_backend_reads_logprobs_and_survives_failures(self):
        from pipeline import vllm_backend
        backend = vllm_backend.VLLMBackend()
        models = Mock(); models.json.return_value = {'data': [{'id': 'qwen'}]}
        good = Mock(); good.json.return_value = {'choices': [{'logprobs': {'content': [
            {'token': 'yes', 'top_logprobs': [{'token': 'yes', 'logprob': math.log(0.9)},
                                              {'token': 'no', 'logprob': math.log(0.1)}]}]}}]}
        with patch.object(vllm_backend.requests, 'get', return_value=models), \
                patch.object(vllm_backend.requests, 'post', return_value=good) as post:
            scores = backend.score_yes([(1, ['a']), (2, ['b'])], __import__('time').monotonic())
        self.assertEqual(set(scores), {1, 2})
        self.assertAlmostEqual(scores[1], 0.9)
        body = post.call_args.kwargs['json']
        self.assertTrue(body['logprobs'])
        self.assertEqual(body['max_tokens'], 4)
        with patch.object(vllm_backend.requests, 'get', return_value=models), \
                patch.object(vllm_backend.requests, 'post', side_effect=RuntimeError('boom')):
            self.assertEqual(backend.score_yes([(1, ['a'])], __import__('time').monotonic()), {})
        self.assertEqual(backend.score_yes([(1, ['a'])], 0.0), {})
        self.assertEqual(backend.score_yes([], __import__('time').monotonic()), {})

    def test_local_backend_reports_no_support(self):
        from pipeline import mlx_backend
        self.assertEqual(mlx_backend.MLXBackend().score_yes([(1, ['a'])], 0.0), {})


if __name__ == '__main__':
    unittest.main()


class ThinkingTests(unittest.TestCase):
    def test_reasoning_is_off_for_answers_and_on_only_for_evidence_when_enabled(self):
        from pipeline import vllm_backend
        backend = vllm_backend.VLLMBackend()
        models = Mock(); models.json.return_value = {'data': [{'id': 'qwen'}]}
        reply = Mock(); reply.json.return_value = {'choices': [{'message': {'content': '<think>x</think>quote'}}]}
        with patch.object(vllm_backend, 'EVIDENCE_THINKING', True), \
                patch.object(vllm_backend.requests, 'get', return_value=models), \
                patch.object(vllm_backend.requests, 'post', return_value=reply) as post:
            self.assertEqual(backend.generate_messages([{'role': 'user', 'content': 'q'}], 60), 'quote')
            self.assertTrue(post.call_args.kwargs['json']['chat_template_kwargs']['enable_thinking'])
            backend.generate_messages([{'role': 'user', 'content': 'q'}], 60, vllm_backend.ANSWER_PASS)
            self.assertFalse(post.call_args.kwargs['json']['chat_template_kwargs']['enable_thinking'])
        with patch.object(vllm_backend, 'EVIDENCE_THINKING', False), \
                patch.object(vllm_backend.requests, 'get', return_value=models), \
                patch.object(vllm_backend.requests, 'post', return_value=reply) as post:
            backend.generate_messages([{'role': 'user', 'content': 'q'}], 60)
            self.assertFalse(post.call_args.kwargs['json']['chat_template_kwargs']['enable_thinking'])

    def test_thinking_raises_the_token_cap_for_evidence_prompts(self):
        from pipeline import vllm_backend
        backend = vllm_backend.VLLMBackend()
        with patch.object(vllm_backend, 'EVIDENCE_THINKING', True), \
                patch.object(vllm_backend, 'THINKING_MAX_TOKENS', 777), \
                patch.object(backend, 'generate_messages', return_value='q') as generate:
            backend.generate_many([(1, ['m'])], 60, __import__('time').monotonic())
        self.assertEqual(generate.call_args.args[1], 777)
