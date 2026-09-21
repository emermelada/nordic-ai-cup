import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dtos import ASRQuestionResponseDto
from pipeline import mlx_backend, vllm_backend
from pipeline.stage_b import build_perq_messages, draft_quotes, render_sentences
from tools.gpu_eval import eval as gpu_eval


MODEL_ID = 'test/model@0123456789abcdef'


def make_item(cid='conversation_0'):
    text = 'Any side effects? None. The blood pressure is normal. No fever.'
    words = [{'word': ' ' + token, 'start': i * 0.5, 'end': i * 0.5 + 0.4}
             for i, token in enumerate(text.split())]
    rows = [
        {'question_id': cid + '_side', 'question': 'Are there no side effects?', 'label': '1',
         'evidence_start': '0', 'evidence_end': '1.4'},
        {'question_id': cid + '_pressure', 'question': 'Is blood pressure normal?', 'label': '1',
         'evidence_start': '2', 'evidence_end': '4.4'},
        {'question_id': cid + '_fever', 'question': 'Does the patient have fever?', 'label': '0',
         'evidence_start': '', 'evidence_end': ''},
    ]
    raw = json.dumps({'results': [
        {'q': 1, 'answer': 'yes', 'quote': 'Any  side\neffects?'},
        {'q': 2, 'answer': 'yes', 'quote': 'pressure is normal.'},
        {'q': 3, 'answer': 'no'},
    ]})
    return {'id': cid, 'words': words, 'rows': rows, 'duration': 10.0,
            'envelope': None, 'primary': {'raw': raw}}


def source_rows(item):
    return [{'conversation': item['id'], 'question_id': row['question_id'],
             'answer': i < 2, 'candidate': [2.0, 4.4] if i < 2 else None,
             'raw': item['primary']['raw']}
            for i, row in enumerate(item['rows'])]


class GPUEvalTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.item = make_item()
        self.items = [self.item]
        self.backend = SimpleNamespace(
            prompt='compact', last_generation_seconds=999,
            complete=Mock(return_value=self.item['primary']['raw']),
            generate_many=Mock(return_value={1: 'Any side effects?', 2: 'pressure is normal.'}),
            generate_messages=Mock(return_value='{"results":[{"q":1,"quote":"Any side effects?"},{"q":2,"quote":"pressure is normal."}]}'),
        )
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'MEDICAL_BACKEND': 'vllm', 'MEDICAL_EVIDENCE_PROMPT': 'v3'}, clear=True).start()
        patch.object(gpu_eval, 'ITEMS', self.items).start()
        patch.object(gpu_eval, 'BASELINE', {self.item['id']: {'response': {
            'answers': [True, True, False], 'evidence_start': [0.0, 2.0, None],
            'evidence_end': [1.4, 4.4, None],
        }}}).start()
        patch.object(gpu_eval, 'RESULTS', self.root / 'results').start()
        patch.object(vllm_backend, 'VLLM_URL', 'http://mock.invalid/v1').start()
        patch.object(vllm_backend, 'VLLM_ANSWER_URL', 'http://mock.invalid/v1').start()
        patch.object(vllm_backend, 'VLLM_MODEL', 'qwen').start()
        patch.object(vllm_backend, 'EVIDENCE_THINKING', False).start()
        patch.object(mlx_backend, 'EVIDENCE_BUDGET_SECONDS', 40).start()
        patch.object(vllm_backend.requests, 'get', side_effect=AssertionError('network discovery forbidden')).start()
        patch.object(vllm_backend.requests, 'post', side_effect=AssertionError('network generation forbidden')).start()
        patch.object(mlx_backend, 'resolve_snapshot', side_effect=AssertionError('model resolution forbidden')).start()
        patch.object(gpu_eval, 'make_backend', side_effect=AssertionError('backend construction forbidden')).start()
        self.source = self.root / 'legacy-answers.json'
        self.write_source()

    def write_source(self, rows=None):
        self.source.write_text(json.dumps({'rows': rows if rows is not None else source_rows(self.item)}))

    def stageb(self, **kwargs):
        options = {'source': self.source, 'mode': 'perq', 'asr_mode': 'base', 'model_id': MODEL_ID}
        options.update(kwargs)
        return gpu_eval.run_stageb(self.backend, 'test-stageb', **options)

    def result(self, summary):
        return json.loads(Path(summary['result_path']).read_text())

    def cache_files(self):
        return list((self.root / 'results').glob('gen-v2-*/*.json'))

    def test_exact_timestamps_no_reply_extension_in_answers_and_stageb(self):
        answers = gpu_eval.run_answers(self.backend, 'answers', asr_mode='base', model_id=MODEL_ID)
        answer_rows = self.result(answers)['rows']
        self.assertEqual(answer_rows[0]['candidate'], [0.0, 1.4])
        stageb_rows = self.result(self.stageb())['rows']
        self.assertEqual(stageb_rows[0]['baseline'], [0.0, 1.4])
        self.assertEqual(stageb_rows[0]['candidate'], [0.0, 1.4])
        self.assertEqual(stageb_rows[0]['raw'], self.item['primary']['raw'])

    def test_turbo_extends_reply_even_without_envelope(self):
        summary = gpu_eval.run_answers(self.backend, 'turbo', asr_mode='turbo', model_id=MODEL_ID)
        self.assertEqual(self.result(summary)['rows'][0]['candidate'], [0.0, 1.9])
        self.assertFalse(gpu_eval.timing_policy(self.item)['exact_timestamps'])

    def test_explicit_environment_supplies_legacy_base_policy(self):
        self.item.pop('primary')
        with patch.dict(os.environ, {'MEDICAL_ASR_MODE': 'base'}):
            self.assertTrue(gpu_eval.timing_policy(self.item)['exact_timestamps'])
        self.assertFalse(gpu_eval.timing_policy(self.item)['exact_timestamps'])

    def test_metadata_and_explicit_timing_conflict_is_not_silent(self):
        self.item['exact_timestamps'] = True
        self.assertEqual(gpu_eval.timing_policy(self.item)['asr_mode'], 'base')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            gpu_eval.timing_policy(self.item, 'turbo')

    def test_energy_handling_matches_parent(self):
        self.item['envelope'] = [-80.0] * 20 + [-20.0] * 1000
        summary = gpu_eval.run_answers(self.backend, 'energy', asr_mode='turbo', model_id=MODEL_ID)
        self.assertEqual(self.result(summary)['rows'][0]['candidate'], [0.2, 1.9])

    def test_raw_drafts_not_reconstructed_from_saved_spans(self):
        self.stageb()
        prompts = self.backend.generate_many.call_args.args[0]
        drafts = draft_quotes(self.item['primary']['raw'], len(self.item['rows']))
        with patch.dict(os.environ, {'MEDICAL_ASR_MODE': 'base'}):
            expected = build_perq_messages(render_sentences(self.item['words']), self.item['rows'][0]['question'], drafts[1])
        self.assertEqual(prompts[0], [1, expected])
        self.assertIn('draft: "Any side effects?"', prompts[0][1][1]['content'])
        self.assertNotIn('draft: "The blood pressure', prompts[0][1][1]['content'])

    def test_missing_raw_rejected_before_generation(self):
        rows = source_rows(self.item)
        del rows[-1]['raw']
        self.write_source(rows)
        with self.assertRaisesRegex(ValueError, 'original ANSWER raw'):
            self.stageb()
        self.backend.generate_many.assert_not_called()
        self.assertFalse((self.root / 'results').exists())

    def test_inconsistent_source_raw_rejected(self):
        rows = source_rows(self.item)
        rows[-1]['raw'] = '{}'
        self.write_source(rows)
        with self.assertRaisesRegex(ValueError, 'Inconsistent source raw'):
            self.stageb()
        self.backend.generate_many.assert_not_called()

    def test_source_answer_disagreement_is_rejected(self):
        rows = source_rows(self.item)
        rows[1]['answer'] = False
        self.write_source(rows)
        with self.assertRaisesRegex(ValueError, 'Source answers disagree'):
            self.stageb()
        self.backend.generate_many.assert_not_called()

    def test_stageb_outputs_cannot_change_fixed_source_answers(self):
        self.backend.generate_many.return_value = {1: 'Any side effects?', 2: 'pressure is normal.', 3: 'No fever.'}
        result = self.result(self.stageb())
        self.assertEqual([r['answer'] for r in result['rows']], [True, True, False])
        self.assertEqual([p[0] for p in self.backend.generate_many.call_args.args[0]], [1, 2])
        self.assertIsNone(result['rows'][2]['candidate'])
        self.assertEqual(result['evidence_completeness']['expected_yes'], 2)

    def test_missing_and_invalid_outputs_report_expected_yes_ids(self):
        self.backend.generate_many.return_value = {1: '', 3: 'No fever.'}
        summary = self.stageb()
        completeness = summary['evidence_completeness']
        self.assertFalse(completeness['complete'])
        self.assertEqual(completeness['missing'], [self.item['rows'][1]['question_id']])
        self.assertEqual(completeness['invalid'], [self.item['rows'][0]['question_id']])
        self.assertEqual(completeness['batches'][0]['unexpected'], [3])
        self.assertEqual(completeness['valid'], 0)
        self.assertEqual(self.result(summary)['rows'][0]['candidate'], [0.0, 1.4])

    def test_empty_skipped_batch_is_not_complete(self):
        self.backend.generate_many.return_value = {}
        summary = self.stageb()
        self.assertFalse(summary['evidence_completeness']['complete'])
        self.assertEqual(len(summary['evidence_completeness']['missing']), 2)

    def test_refine_duplicates_and_unalignable_quotes_are_invalid(self):
        self.backend.generate_messages.return_value = json.dumps({'results': [
            {'q': 1, 'quote': 'Any side effects?'}, {'q': 1, 'quote': 'Any side effects?'},
            {'q': 2, 'quote': 'zzzz qqqqq xxxx'},
        ]})
        summary = self.stageb(mode='refine')
        self.assertEqual(len(summary['evidence_completeness']['invalid']), 2)
        self.assertEqual(summary['evidence_completeness']['missing'], [])

    def test_parallel_batch_timing_uses_complete_wall_time(self):
        with patch.object(gpu_eval.time, 'monotonic', side_effect=[10.0, 11.0, 19.0]):
            summary = self.stageb()
        self.assertEqual(summary['generation_mean'], 9.0)
        self.assertEqual(summary['generation_max'], 9.0)
        self.assertEqual(json.loads(self.cache_files()[0].read_text())['seconds'], 9.0)

    def test_replay_miss_never_generates_or_creates_cache(self):
        with self.assertRaisesRegex(gpu_eval.CacheError, 'Replay cache miss'):
            self.stageb(replay=True)
        self.backend.generate_many.assert_not_called()
        self.assertFalse((self.root / 'results').exists())

    def test_replay_hit_never_generates_or_mutates_generation_cache(self):
        original = self.stageb()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.cache_files()}
        self.backend.generate_many.reset_mock()
        replay = self.stageb(replay=True)
        self.backend.generate_many.assert_not_called()
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.cache_files()})
        self.assertEqual(original['candidate'], replay['candidate'])
        self.assertNotEqual(original['result_path'], replay['result_path'])
        self.assertTrue(Path(original['result_path']).exists())

    def test_replay_incompatible_entry_fails_without_generation(self):
        self.stageb()
        path = self.cache_files()[0]
        record = json.loads(path.read_text())
        record['manifest']['generation']['model_id'] = 'other/weights@different'
        path.write_text(json.dumps(record))
        before = path.read_bytes()
        self.backend.generate_many.reset_mock()
        with self.assertRaisesRegex(gpu_eval.CacheError, 'fingerprint mismatch'):
            self.stageb(replay=True)
        self.backend.generate_many.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_configuration_and_input_changes_get_new_fingerprints(self):
        settings = gpu_eval.generation_settings('stageb', 'perq', model_id=MODEL_ID)
        policy = gpu_eval.timing_policy(self.item, 'base')
        original = gpu_eval.request_manifest(self.item, [{'role': 'user', 'content': 'quote'}], settings,
                                             policy, stage='stageb', mode='perq')
        changes = [
            ('generation', 'model_id', 'other/model@rev'),
            ('generation', 'endpoint', 'http://other.invalid/v1'),
            ('generation', 'served_alias', 'other-alias'),
            ('generation', 'max_tokens', 333),
            ('generation', 'parallel', 1),
            ('generation', 'budget_seconds', 2),
            ('generation', 'chat_template_kwargs', {'enable_thinking': True}),
            ('timing_policy', 'extend_replies', True),
        ]
        for section, key, value in changes:
            changed = copy.deepcopy(original)
            changed[section][key] = value
            self.assertNotEqual(gpu_eval.fingerprint(original), gpu_eval.fingerprint(changed), key)
        changed = copy.deepcopy(original)
        changed['words'][0]['end'] += 0.01
        self.assertNotEqual(gpu_eval.fingerprint(original), gpu_eval.fingerprint(changed))
        changed = copy.deepcopy(original)
        changed['messages'][0]['content'] += '!'
        self.assertNotEqual(gpu_eval.fingerprint(original), gpu_eval.fingerprint(changed))

    def test_changed_prompt_or_model_replay_is_a_miss(self):
        self.stageb()
        self.backend.generate_many.reset_mock()
        with self.assertRaises(gpu_eval.CacheError):
            self.stageb(replay=True, model_id='other/model@rev')
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': 'match'}):
            with self.assertRaises(gpu_eval.CacheError):
                self.stageb(replay=True)
        self.backend.generate_many.assert_not_called()

    def test_model_revision_is_explicit_not_guessed_from_qwen_alias(self):
        with self.assertRaisesRegex(ValueError, 'full model revision'):
            gpu_eval.generation_settings('stageb', 'perq')
        settings = gpu_eval.generation_settings('stageb', 'perq', model_id=MODEL_ID)
        self.assertEqual(settings['model_id'], MODEL_ID)
        self.assertEqual(settings['served_alias'], 'qwen')

    def test_separate_answer_endpoint_needs_separate_provenance_run(self):
        with patch.object(vllm_backend, 'VLLM_ANSWER_URL', 'http://answer.invalid/v1'):
            with self.assertRaisesRegex(ValueError, 'separate run'):
                gpu_eval.generation_settings('answers', model_id=MODEL_ID)

    def test_thinking_token_budget_matches_parallel_backend(self):
        with patch.object(vllm_backend, 'EVIDENCE_THINKING', True), patch.object(vllm_backend, 'THINKING_MAX_TOKENS', 777):
            self.assertEqual(gpu_eval.generation_settings('stageb', 'perq', model_id=MODEL_ID)['max_tokens'], 777)
            self.assertEqual(gpu_eval.generation_settings('stageb', 'refine', model_id=MODEL_ID)['max_tokens'], 320)
            self.assertFalse(gpu_eval.generation_settings('answers', model_id=MODEL_ID)['chat_template_kwargs']['enable_thinking'])

    def test_legacy_generation_cache_never_reused_for_corrected_stageb(self):
        path = self.root / 'results' / 'gen-test-stageb' / 'conversation_0-old.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'raw': '{"1":"wrong"}', 'seconds': 1.0}))
        before = path.read_bytes()
        with self.assertRaises(gpu_eval.CacheError):
            self.stageb(replay=True)
        self.assertEqual(path.read_bytes(), before)
        self.backend.generate_many.assert_not_called()

    def test_answer_prompt_selected_and_fingerprinted(self):
        self.backend.prompt = 'minimal'
        summary = gpu_eval.run_answers(self.backend, 'answers', asr_mode='base', model_id=MODEL_ID)
        record = json.loads(self.cache_files()[0].read_text())
        self.assertEqual(record['manifest']['messages'], gpu_eval.answer_messages(self.item, 'minimal'))
        self.assertEqual(summary['manifest']['generation']['answer_prompt'], 'minimal')

    def test_cli_replay_does_not_construct_backend(self):
        args = ['eval.py', 'stageb', '--source', str(self.source), '--evidence-mode', 'perq',
                '--asr-mode', 'base', '--model-id', MODEL_ID, '--replay']
        with patch.object(gpu_eval.sys, 'argv', args):
            with self.assertRaises(gpu_eval.CacheError):
                gpu_eval.main()
        gpu_eval.make_backend.assert_not_called()

    def test_zero_yes_requires_no_generation_cache(self):
        rows = source_rows(self.item)
        raw = json.dumps({'results': [{'q': i + 1, 'answer': 'no'} for i in range(len(rows))]})
        for row in rows:
            row['answer'] = False
            row['candidate'] = None
            row['raw'] = raw
        self.write_source(rows)
        for mode in ('perq', 'locate'):
            with self.subTest(mode=mode):
                summary = self.stageb(mode=mode, replay=True)
                self.assertTrue(summary['evidence_completeness']['complete'])
                self.assertEqual(summary['evidence_completeness']['expected_yes'], 0)
        self.assertEqual(self.cache_files(), [])
        self.backend.generate_many.assert_not_called()

    def test_variants_are_perq_only(self):
        with self.assertRaisesRegex(ValueError, 'only supported'):
            self.stageb(mode='refine', variant='no-draft')
        self.backend.generate_messages.assert_not_called()

    def test_no_draft_variant_changes_only_stageb_messages(self):
        original = self.result(self.stageb())
        no_draft = self.result(self.stageb(variant='no-draft'))
        messages = self.backend.generate_many.call_args.args[0][0][1]
        self.assertNotIn('draft', messages[0]['content'].lower())
        self.assertNotIn('draft:', messages[1]['content'])
        self.assertEqual([r['answer'] for r in original['rows']], [r['answer'] for r in no_draft['rows']])
        self.assertEqual(no_draft['manifest']['variants']['variant'], 'no-draft')
        self.assertEqual(len(self.cache_files()), 2)

    def locate_generations(self):
        return [{1: '{"first_sentence":0,"last_sentence":1}',
                 2: '{"first_sentence":2,"last_sentence":2}'},
                {1: '{"start_word":0,"end_word":3}',
                 2: '{"start_word":4,"end_word":8}'}]

    def test_locate_times_and_records_the_complete_chain(self):
        outputs = iter(self.locate_generations())
        elapsed = iter([2.0, 3.0])
        clock = [10.0]

        def generate(prompts, max_tokens, request_started):
            clock[0] += next(elapsed)
            return next(outputs)

        self.backend.generate_many.side_effect = generate
        with patch.object(gpu_eval.time, 'monotonic', side_effect=lambda: clock[0]):
            summary = self.stageb(mode='locate')
        self.assertEqual(summary['generation_mean'], 5.0)
        self.assertEqual(summary['generation_max'], 5.0)
        self.assertTrue(summary['evidence_completeness']['complete'])
        self.assertEqual([call.args[1:] for call in self.backend.generate_many.call_args_list], [(64, 10.0), (64, 10.0)])
        calls = summary['manifest']['requests'][0]['calls']
        self.assertEqual([call['phase'] for call in calls], ['locate', 'bounds'])
        self.assertEqual([call['seconds'] for call in calls], [2.0, 3.0])
        rows = self.result(summary)['rows']
        self.assertEqual([row['answer'] for row in rows], [True, True, False])
        self.assertEqual(rows[0]['candidate'], [0.0, 1.9])
        self.assertEqual(rows[1]['candidate'], [2.0, 4.4])
        self.assertEqual(len(self.cache_files()), 1)
        cache = json.loads(self.cache_files()[0].read_text())
        self.assertIn('chain_source_sha256', cache['manifest'])
        self.assertIn('bounds_full_context_templates', cache['manifest'])
        self.assertIn('0', summary['report']['folds'])

    def test_locate_replay_reads_both_phases_without_generating_or_mutating_cache(self):
        self.backend.generate_many.side_effect = self.locate_generations()
        original = self.stageb(mode='locate')
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.cache_files()}
        self.backend.generate_many.reset_mock()
        self.backend.generate_many.side_effect = AssertionError('No model during replay')
        replay = self.stageb(mode='locate', replay=True)
        self.backend.generate_many.assert_not_called()
        self.assertEqual(self.result(original)['rows'], self.result(replay)['rows'])
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.cache_files()})
        self.assertEqual(original['manifest']['requests'][0]['calls'], replay['manifest']['requests'][0]['calls'])

    def test_locate_replay_preserves_budget_skips_instead_of_launching_a_missing_phase(self):
        clock = [10.0]

        def generate(*args):
            clock[0] += 41.0
            return self.locate_generations()[0]

        self.backend.generate_many.side_effect = generate
        with patch.object(gpu_eval.time, 'monotonic', side_effect=lambda: clock[0]):
            original = self.stageb(mode='locate')
        self.backend.generate_many.assert_called_once()
        self.assertEqual(original['evidence_completeness']['batches'][0]['skipped'], 'budget')
        self.assertEqual(original['generation_max'], 41.0)
        self.backend.generate_many.reset_mock()
        self.backend.generate_many.side_effect = AssertionError('The boundary batch was never generated')
        replay = self.stageb(mode='locate', replay=True)
        self.backend.generate_many.assert_not_called()
        self.assertEqual(original['evidence_completeness'], replay['evidence_completeness'])
        self.assertEqual(self.result(original)['rows'], self.result(replay)['rows'])

    def test_locate_replay_miss_has_no_generation_or_network(self):
        args = ['eval.py', 'stageb', '--source', str(self.source), '--evidence-mode', 'locate',
                '--asr-mode', 'base', '--model-id', MODEL_ID, '--replay']
        with patch.object(gpu_eval.sys, 'argv', args), self.assertRaises(gpu_eval.CacheError):
            gpu_eval.main()
        gpu_eval.make_backend.assert_not_called()
        self.backend.generate_many.assert_not_called()
        self.assertEqual(self.cache_files(), [])

    def test_locate_context_change_invalidates_the_chain_cache(self):
        self.backend.generate_many.side_effect = self.locate_generations()
        self.stageb(mode='locate')
        self.backend.generate_many.reset_mock()
        with patch('pipeline.locate.CONTEXT_SENTENCES', 2), self.assertRaises(gpu_eval.CacheError):
            self.stageb(mode='locate', replay=True)
        self.backend.generate_many.assert_not_called()

    def test_locate_invalid_indices_preserve_drafts_and_are_reported(self):
        generated = self.locate_generations()
        generated[1][1] = '{"start_word":9,"end_word":10}'
        self.backend.generate_many.side_effect = generated
        summary = self.stageb(mode='locate')
        self.assertEqual(summary['evidence_completeness']['invalid'], [self.item['rows'][0]['question_id']])
        self.assertEqual(summary['evidence_completeness']['missing'], [])
        self.assertEqual(self.result(summary)['rows'][0]['candidate'], [0.0, 1.4])

    def test_locate_failed_location_does_not_trigger_a_boundary_batch(self):
        self.backend.generate_many.return_value = {1: 'bad', 2: '{"first_sentence":0,"last_sentence":999}'}
        summary = self.stageb(mode='locate')
        self.backend.generate_many.assert_called_once()
        self.assertFalse(summary['evidence_completeness']['complete'])
        self.assertEqual(len(summary['evidence_completeness']['missing']), 2)

    def test_locate_target_labels_do_not_change_the_cached_prompt_chain(self):
        self.backend.generate_many.side_effect = self.locate_generations()
        original = self.stageb(mode='locate')
        for row in self.item['rows']:
            row['label'], row['evidence_start'], row['evidence_end'] = '0', '', ''
        self.backend.generate_many.reset_mock()
        replay = self.stageb(mode='locate', replay=True)
        self.backend.generate_many.assert_not_called()
        self.assertEqual(original['manifest']['requests'], replay['manifest']['requests'])
        self.assertEqual([(r['answer'], r['candidate']) for r in self.result(original)['rows']],
                         [(r['answer'], r['candidate']) for r in self.result(replay)['rows']])

    def test_retrieved_integration_full_data_folds_and_target_label_independence(self):
        self.items.extend(make_item(f'conversation_{i}') for i in range(1, 6))
        first = self.stageb(variant='retrieved', limit=1)
        first_messages = copy.deepcopy(self.backend.generate_many.call_args.args[0])
        manifest = first['manifest']['variants']
        self.assertEqual(len(manifest['folds']), 6)
        self.assertIn('0', first['report']['folds'])
        row_to_conv = {row['question_id']: item['id'] for item in self.items for row in item['rows']}
        for donor_ids in first['manifest']['used_donors'].values():
            self.assertEqual(len(donor_ids), 2)
            conversations = {row_to_conv[qid] for qid in donor_ids}
            self.assertEqual(len(conversations), 2)
            self.assertTrue(all(manifest['folds'][cid] != manifest['folds'][self.item['id']] for cid in conversations))
        for item in self.items:
            if manifest['folds'][item['id']] == manifest['folds'][self.item['id']]:
                for row in item['rows']:
                    row['label'] = '0'
                    row['evidence_start'], row['evidence_end'] = '', ''
        self.backend.generate_many.reset_mock()
        second = self.stageb(variant='retrieved', limit=1, replay=True)
        self.backend.generate_many.assert_not_called()
        self.assertEqual(first['manifest']['used_donors'], second['manifest']['used_donors'])
        cached_messages = json.loads(self.cache_files()[0].read_text())['manifest']['messages']
        self.assertEqual(first_messages, cached_messages)
        self.assertEqual([r['answer'] for r in self.result(first)['rows']], [r['answer'] for r in self.result(second)['rows']])


if __name__ == '__main__':
    unittest.main()
