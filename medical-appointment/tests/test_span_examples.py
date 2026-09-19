import copy
import json
import os
import time
import unittest
from unittest.mock import Mock, patch

from pipeline import mlx_backend, span_examples
from pipeline.core import answer_response
from pipeline.runtime import _stage_b
from pipeline.stage_b import (
    apply_evidence,
    build_perq_messages,
    refine_system,
    render_sentences,
)
from tests.test_evidence_variants import make_item
from tools.gpu_eval.evidence_variants import EvidenceVariants


def clear_caches():
    span_examples._bank.cache_clear()
    span_examples._pool.cache_clear()


def freeze(items):
    variant = EvidenceVariants(items, 'retrieved')
    union = {}
    for fold in sorted(set(variant.folds.values())):
        for demo in variant._pool(fold)[0]:
            union[demo['question_id']] = demo
    return {'folds': variant.folds, 'question_sets': [
        {'conversation': item['id'], 'questions': [row['question'] for row in item['rows']]}
        for item in items
    ], 'examples': list(union.values())}


class SpanExamplesTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'MEDICAL_ASR_MODE': 'base', 'MEDICAL_EVIDENCE_PROMPT': 'v3'})
        env.start()
        self.addCleanup(env.stop)
        self.items = [make_item(i) for i in range(9)]
        for item in self.items:
            item['rows'][1]['question'] += ' ' + item['id']
            for index in range(8):
                item['rows'].append(dict(item['rows'][1], question_id=f'{item["id"]}_q{index + 3:02d}',
                                         question=f'Unrelated detail {index} in {item["id"]}?'))
        self.bank = freeze(self.items)
        self.questions = [row['question'] for row in self.items[0]['rows']]
        self.question = self.questions[0]
        clear_caches()
        self.addCleanup(clear_caches)
        loader = patch('pipeline.span_examples.Path.read_text', return_value=json.dumps(self.bank))
        self.load = loader.start()
        self.addCleanup(loader.stop)

    def test_exact_offline_prompts_and_donors_for_all_known_sets(self):
        offline = EvidenceVariants(self.items, 'retrieved')
        draft = 'raw DRAFT\n  deliberately  untrimmed'
        for item in self.items:
            questions = [row['question'] for row in item['rows']]
            for row in item['rows']:
                with self.subTest(conversation=item['id'], question=row['question_id']):
                    expected, ids = offline.messages(item, row['question'], draft)
                    demos = span_examples.examples_for(questions[::-1], row['question'])
                    with patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': 'retrieved'}):
                        actual = build_perq_messages(render_sentences(item['words']), row['question'],
                                                     draft, examples=demos)
                    self.assertEqual(actual, expected)
                    self.assertEqual([demo['question_id'] for demo in demos], ids)
                    self.assertEqual(len({demo['conversation'] for demo in demos}), 2)
        self.load.assert_called_once()

    def test_full_question_identity_excludes_fold_despite_fresh_asr(self):
        for item in self.items:
            questions = [row['question'] for row in item['rows']]
            demos = span_examples.examples_for(questions, self.question)
            fold = self.bank['folds'][item['id']]
            pool, _ = span_examples._pool(fold)
            self.assertTrue(all(self.bank['folds'][demo['conversation']] != fold for demo in pool))
            fresh = copy.deepcopy(item['words'])
            fresh[0].update(word=' Changed-ASR', start=0.123, end=0.456)
            original = build_perq_messages(render_sentences(item['words']), self.question, examples=demos)
            changed = build_perq_messages(render_sentences(fresh), self.question,
                                          examples=span_examples.examples_for(questions[::-1], self.question))
            self.assertEqual(original[0], changed[0])
            self.assertNotEqual(original[1], changed[1])

    def test_unseen_question_set_uses_all_public_donors(self):
        unknown = self.questions[:-1] + ['A new platform question?']
        demos = span_examples.examples_for(unknown, 'Unmatchedtoken?')
        pool, _ = span_examples._pool(None)
        self.assertEqual({demo['question_id'] for demo in pool},
                         {demo['question_id'] for demo in self.bank['examples']})
        self.assertEqual([demo['conversation'] for demo in demos], ['c00', 'c01'])
        self.assertNotEqual(demos, span_examples.examples_for(self.questions, 'Unmatchedtoken?'))
        _, identities = span_examples._bank()
        self.assertNotIn(tuple(sorted(unknown)), identities)
        self.assertNotIn(tuple(sorted(self.questions[:1])), identities)

    def test_target_fold_example_content_is_never_indexed_or_used(self):
        expected = span_examples.examples_for(self.questions, self.question)
        data, _ = span_examples._bank()
        for demo in data['examples']:
            if data['folds'][demo['conversation']] == data['folds']['c00']:
                demo.update(question=None, excerpt='SECRET_TARGET_GOLD', quote='SECRET_TARGET_GOLD')
        span_examples._pool.cache_clear()
        actual = span_examples.examples_for(self.questions, self.question)
        self.assertEqual(actual, expected)
        messages = build_perq_messages('fresh transcript', self.question, examples=actual)
        self.assertNotIn('SECRET_TARGET_GOLD', json.dumps(messages))

    def test_bm25_is_cached_per_eligible_pool_and_only_indexes_questions(self):
        with patch('pipeline.span_examples.BM25Okapi', wraps=span_examples.BM25Okapi) as ranker:
            for item in self.items * 2:
                span_examples.examples_for([row['question'] for row in item['rows']], self.question)
            span_examples.examples_for(['Unseen set'], self.question)
            span_examples.examples_for(['Another unseen set'], self.question)
        self.assertEqual(ranker.call_count, 4)
        self.assertEqual(span_examples._pool.cache_info().currsize, 4)
        from pipeline.stage_b import _content
        for call in ranker.call_args_list:
            self.assertTrue(all(tokens == _content(self.question) for tokens in call.args[0]))
            self.assertEqual(call.kwargs, {'k1': 1.5, 'b': 0.75, 'epsilon': 0.25})

    def test_ties_and_multiple_examples_still_choose_two_distinct_conversations(self):
        duplicate = next(demo for demo in self.bank['examples'] if demo['conversation'] == 'c01')
        self.bank['examples'].append(dict(duplicate, question_id='c01_q00'))
        self.bank['examples'].reverse()
        self.load.return_value = json.dumps(self.bank)
        demos = span_examples.examples_for(self.questions, 'Unmatchedtoken?')
        self.assertEqual([demo['question_id'] for demo in demos], ['c01_q00', 'c02_q01'])

    def test_original_perq_prompts_unchanged_without_examples(self):
        for prompt in ('v3', 'annot', 'match', 'exchange', None):
            with self.subTest(prompt=prompt), patch.dict(os.environ):
                if prompt is None:
                    os.environ.pop('MEDICAL_EVIDENCE_PROMPT', None)
                else:
                    os.environ['MEDICAL_EVIDENCE_PROMPT'] = prompt
                expected = (refine_system().rsplit('Respond with minified JSON only', 1)[0].rstrip()
                            + '\nReply with that one passage, copied exactly from the transcript, and nothing else: '
                            'no quotation marks, no line number, no explanation.')
                for draft in ('', 'untrimmed\n  draft'):
                    draft_line = f'\ndraft: "{draft}"' if draft else ''
                    self.assertEqual(build_perq_messages('[0] Transcript.', 'Question?', draft), [
                        {'role': 'system', 'content': expected},
                        {'role': 'user', 'content': 'TRANSCRIPT\n[0] Transcript.\n\nQUESTION (answered yes): '
                         f'Question?{draft_line}\n\nEVIDENCE:'},
                    ])
        self.load.assert_not_called()


class FrozenBankTest(unittest.TestCase):
    def test_only_curated_examples_and_full_unlabeled_question_sets_are_stored(self):
        clear_caches()
        self.addCleanup(clear_caches)
        data, identities = span_examples._bank()
        self.assertEqual(set(data), {'folds', 'question_sets', 'examples'})
        self.assertEqual(len(data['folds']), 39)
        self.assertEqual(len(identities), 39)
        self.assertEqual(len(data['examples']), 193)
        self.assertEqual(set(data['folds'].values()), {0, 1, 2})
        for item in data['question_sets']:
            self.assertEqual(set(item), {'conversation', 'questions'})
            self.assertEqual(len(item['questions']), 10)
            demos = span_examples.examples_for(item['questions'], item['questions'][0])
            self.assertEqual(len({demo['conversation'] for demo in demos}), 2)
            self.assertTrue(all(data['folds'][demo['conversation']] != data['folds'][item['conversation']]
                                for demo in demos))
        for demo in data['examples']:
            self.assertEqual(set(demo), {'conversation', 'question_id', 'question', 'excerpt', 'quote'})
        self.assertEqual(len(span_examples._pool(None)[0]), len(data['examples']))


class RetrievedBackendTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': 'retrieved', 'MEDICAL_ASR_MODE': 'base'})
        env.start()
        self.addCleanup(env.stop)
        for name, value in (('EVIDENCE_MODE', 'perq'), ('EVIDENCE_MODEL', 'existing-model'),
                            ('EVIDENCE_BUDGET_SECONDS', 40.0)):
            setting = patch.object(mlx_backend, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        self.backend = mlx_backend.MLXBackend()
        self.backend.generate_many = Mock(return_value={1: 'Your blood pressure is normal,'})
        self.item = make_item(0)
        self.questions = ['Was the blood pressure normal?', 'Was there a fever?', 'Is the pulse regular?']
        self.answers = [True, False, True]
        self.drafts = {1: 'Original\n  draft', 2: 'Negative draft', 3: 'Other draft'}
        self.anchors = [(0.0, 1.0), (None, None), (1.0, 2.0)]
        self.demos = [{'conversation': 'donor', 'question_id': 'donor_q1', 'question': 'Donor question?',
                       'excerpt': '[0] Donor fact.', 'quote': 'Donor fact.'}]

    def test_retrieved_builder_gets_full_set_only_for_yes_and_preserves_inputs_and_budget(self):
        original = copy.deepcopy((self.item, self.questions, self.answers, self.drafts, self.anchors))
        with patch('pipeline.span_examples.examples_for', return_value=self.demos) as examples, \
                patch('pipeline.mlx_backend.time.monotonic', side_effect=[10.0, 11.0]):
            frame = self.backend.complete_evidence(self.item['words'], self.questions, self.answers,
                                                   self.drafts, self.anchors, 2.0)
        self.assertEqual(examples.call_args_list, [
            unittest.mock.call(self.questions, self.questions[0]),
            unittest.mock.call(self.questions, self.questions[2]),
        ])
        expected = [(i + 1, build_perq_messages(render_sentences(self.item['words']), question,
                                               self.drafts[i + 1], examples=self.demos))
                    for i, question in enumerate(self.questions) if self.answers[i]]
        self.backend.generate_many.assert_called_once_with(expected, 120, 2.0)
        self.assertEqual(frame, {'mode': 'perq', 'outputs': {1: 'Your blood pressure is normal,'}, 'seconds': 1.0})
        self.assertEqual((self.item, self.questions, self.answers, self.drafts, self.anchors), original)

    def test_default_path_never_loads_examples(self):
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': 'v3'}), \
                patch('pipeline.span_examples.examples_for', side_effect=AssertionError('unexpected bank access')):
            self.backend.complete_evidence(self.item['words'], self.questions, self.answers,
                                            self.drafts, self.anchors, time.monotonic())
            expected = [(i + 1, build_perq_messages(render_sentences(self.item['words']), question,
                                                   self.drafts[i + 1]))
                        for i, question in enumerate(self.questions) if self.answers[i]]
        self.assertEqual(self.backend.generate_many.call_args.args[0], expected)

    def test_budget_and_no_answers_skip_retrieval_and_generation(self):
        with patch('pipeline.span_examples.examples_for', side_effect=AssertionError('late retrieval')), \
                patch('pipeline.mlx_backend.time.monotonic', return_value=41.0):
            frame = self.backend.complete_evidence(self.item['words'], self.questions, self.answers,
                                                   self.drafts, self.anchors, 0.0)
            self.assertEqual(frame, {'mode': 'perq', 'skipped': 'budget'})
            self.assertIsNone(self.backend.complete_evidence(self.item['words'], self.questions, [False] * 3,
                                                             {}, self.anchors, 0.0))
        self.backend.generate_many.assert_not_called()

    def test_retrieval_errors_keep_existing_runtime_fallback(self):
        raw = json.dumps({'results': [{'q': 1, 'answer': 'yes', 'quote': 'Your blood pressure is normal,'},
                                     {'q': 2, 'answer': 'no'}, {'q': 3, 'answer': 'no'}]})
        words = self.item['words']
        response = answer_response(raw, words, self.questions, 100.0, alignment='numeric')
        for error in (FileNotFoundError('missing bank'), ValueError('invalid bank')):
            with self.subTest(error=error), \
                    patch('pipeline.span_examples.examples_for', side_effect=error), self.assertLogs(level='ERROR'):
                frame = _stage_b(self.backend, words, self.questions, raw,
                                 {'duration': 100.0, 'exact_timestamps': True}, time.monotonic())
            self.assertEqual(frame, {'error': f'{type(error).__name__}: {error}'})
            self.assertEqual(apply_evidence(frame, response, words, 100.0), response)
        self.backend.generate_many.assert_not_called()


if __name__ == '__main__':
    unittest.main()
