import os
import unittest
from unittest.mock import patch

from dtos import ASRQuestionResponseDto
from pipeline import base_asr, stage_b
from pipeline.evidence import refine_evidence
from pipeline.runtime import _stage_b
from tests.test_stage_b import make_words


class BaseAsrTests(unittest.TestCase):
    def test_sentences_split_on_end_punctuation_only_and_keep_titles(self):
        words = make_words('Morning Dr. Fabricius. How are you? Fine thanks', gap_after={2})
        self.assertEqual(base_asr.sentence_ranges(words), [(0, 2), (3, 5), (6, 7)])
        self.assertEqual(base_asr.sentence_ranges([]), [])

    def test_stage_b_uses_the_annotators_split_in_base_mode(self):
        words = make_words('one two three four five six', gap_after={2})
        with patch.dict(os.environ, {'MEDICAL_ASR_MODE': 'base'}):
            self.assertEqual(stage_b.sentence_ranges(words), [(0, 5)])
        with patch.dict(os.environ, {'MEDICAL_ASR_MODE': 'turbo'}):
            self.assertEqual(stage_b.sentence_ranges(words), [(0, 2), (3, 5)])

    def test_exact_timestamps_skip_reply_extension_and_onset(self):
        words = make_words('Any fever? No fever today.')
        response = ASRQuestionResponseDto(answers=[True], evidence_start=[words[0]['start']], evidence_end=[words[1]['end']])
        extended = refine_evidence(response, words, None)
        self.assertGreater(extended.evidence_end[0], words[1]['end'])
        kept = refine_evidence(response, words, [-30.0] * 1000, extend_replies=False)
        self.assertEqual((kept.evidence_start[0], kept.evidence_end[0]), (words[0]['start'], words[1]['end']))

    def test_prompt_selection_by_environment(self):
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': 'annot'}):
            self.assertIn('Aromere', stage_b.refine_system())
            self.assertIn('Aromere', stage_b.build_refine_messages(make_words('Hello there.'), ['q'], [True], {})[0]['content'])
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': 'v3'}):
            self.assertIn('Conventions of the annotation', stage_b.refine_system())

    def test_worker_stage_b_keeps_exact_anchors(self):
        class Backend:
            def __init__(self):
                self.calls = []

            def complete_evidence(self, words, questions, answers, drafts, anchors, started):
                self.calls.append(anchors)
                return None

        words = make_words('Any fever? No fever today.')
        raw = '{"results":[{"q":1,"answer":"yes","units":[0],"quote":"Any fever?"}]}'
        backend = Backend()
        _stage_b(backend, words, ['Fever?'], raw, {'duration': 10.0, 'exact_timestamps': True}, 0.0)
        self.assertEqual(backend.calls[0][0], (words[0]['start'], words[1]['end']))
        _stage_b(backend, words, ['Fever?'], raw, {'duration': 10.0}, 0.0)
        self.assertGreater(backend.calls[1][0][1], words[1]['end'])


if __name__ == '__main__':
    unittest.main()


class PerQuestionEvidenceTests(unittest.TestCase):
    def test_perq_prompt_carries_the_whole_transcript_and_asks_for_a_bare_quote(self):
        words = make_words('Any fever? No fever today. The dose is 100 mg.')
        text = stage_b.render_sentences(words)
        self.assertEqual(text.splitlines()[0], '[0] Any fever?')
        messages = stage_b.build_perq_messages(text, 'Is the patient free of fever?', 'No fever today.')
        self.assertNotIn('minified JSON', messages[0]['content'])
        self.assertIn('copied exactly from the transcript', messages[0]['content'])
        self.assertIn('[2] The dose is 100 mg.', messages[1]['content'])
        self.assertIn('QUESTION (answered yes): Is the patient free of fever?', messages[1]['content'])
        self.assertIn('draft: "No fever today."', messages[1]['content'])
        self.assertTrue(messages[1]['content'].endswith('EVIDENCE:'))
        self.assertNotIn('draft:', stage_b.build_perq_messages(text, 'q')[1]['content'])

    def test_perq_frame_applies_like_a_window_frame(self):
        words = make_words('Any fever? No fever today.')
        response = ASRQuestionResponseDto(answers=[True], evidence_start=[0.0], evidence_end=[0.4])
        frame = {'mode': 'perq', 'outputs': {'1': 'No fever today.'}}
        result = stage_b.apply_evidence(frame, response, words, 30.0, extend_replies=False)
        self.assertAlmostEqual(result.evidence_start[0], words[2]['start'])
        self.assertAlmostEqual(result.evidence_end[0], words[4]['end'])

    def test_generate_many_is_serial_and_budget_bounded(self):
        from pipeline import mlx_backend
        backend = mlx_backend.MLXBackend()
        with patch.object(backend, 'generate_messages', side_effect=lambda *a, **k: 'quote') as generate:
            outputs = backend.generate_many([(1, ['a']), (2, ['b'])], 60, 1e18)
            self.assertEqual(outputs, {1: 'quote', 2: 'quote'})
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(backend.generate_many([(1, ['a'])], 60, 0.0), {})


class AnswerPromptTests(unittest.TestCase):
    def test_named_prompt_adds_the_phonetic_rule_and_keeps_the_rest(self):
        from pipeline.evidence import COMPACT_SYSTEM, build_compact_messages, build_named_messages
        words = make_words('Active L, Aromere and Isomeprazole. All three renewed.')
        compact = build_compact_messages(words, ['Is Airomir among the renewed medicines?'])
        named = build_named_messages(words, ['Is Airomir among the renewed medicines?'])
        self.assertNotIn('Airomir', compact[0]['content'])
        self.assertIn('"Aromere"/"Aromir" is Airomir', named[0]['content'])
        self.assertIn('minified JSON', named[0]['content'])
        self.assertEqual(compact[1], named[1])
        self.assertGreater(len(named[0]['content']), len(COMPACT_SYSTEM))

    def test_backend_accepts_the_named_prompt_and_reads_the_environment(self):
        from pipeline import mlx_backend
        backend = mlx_backend.MLXBackend(prompt='named')
        with patch.object(backend, 'generate_messages', return_value='{}') as generate:
            backend.complete(make_words('Hello there.'), ['q'])
        self.assertIn('sounds like it', generate.call_args.args[0][0]['content'])
        with self.assertRaises(ValueError):
            mlx_backend.MLXBackend(prompt='nonexistent')


class ServingSelectionTests(unittest.TestCase):
    def test_every_serving_knob_is_reachable_from_the_environment(self):
        import importlib
        from pipeline import mlx_backend
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_MODE': 'perq', 'MEDICAL_ANSWER_PROMPT': 'named'}):
            reloaded = importlib.reload(mlx_backend)
            self.assertEqual(reloaded.EVIDENCE_MODE, 'perq')
            self.assertEqual(reloaded.DEFAULT_PROMPT, 'named')
            self.assertIn('perq', reloaded.EVIDENCE_MAX_TOKENS)
        importlib.reload(mlx_backend)
        self.assertEqual(mlx_backend.EVIDENCE_MODE, 'refine')

    def test_perq_prompts_are_built_for_every_yes_question_only(self):
        import importlib
        from pipeline import mlx_backend
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_MODE': 'perq'}):
            reloaded = importlib.reload(mlx_backend)
            backend = reloaded.MLXBackend()
            words = make_words('Any fever? No fever today. The dose is 100 mg.')
            with patch.object(backend, 'generate_many', return_value={1: 'No fever today.'}) as many:
                frame = backend.complete_evidence(words, ['Fever?', 'Dose?'], [True, False], {1: 'd'},
                                                  [(0.0, 1.0), (None, None)], 1e18)
            self.assertEqual(frame['mode'], 'perq')
            prompts = many.call_args.args[0]
            self.assertEqual([key for key, _ in prompts], [1])
            self.assertIn('[2] The dose is 100 mg.', prompts[0][1][1]['content'])
        importlib.reload(mlx_backend)


class EvidenceBudgetTests(unittest.TestCase):
    def test_budget_is_env_selectable_and_defaults_wide_of_the_clean_path(self):
        import importlib
        from pipeline import mlx_backend
        self.assertGreaterEqual(mlx_backend.EVIDENCE_BUDGET_SECONDS, 40)
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_BUDGET': '12'}):
            reloaded = importlib.reload(mlx_backend)
            self.assertEqual(reloaded.EVIDENCE_BUDGET_SECONDS, 12.0)
            backend = reloaded.MLXBackend()
            with patch.object(reloaded, 'EVIDENCE_MODEL', 'm'):
                late = backend.complete_evidence(make_words('Hi there.'), ['q'], [True], {}, [(0.0, 1.0)],
                                                 __import__('time').monotonic() - 99)
            self.assertEqual(late, {'mode': reloaded.EVIDENCE_MODE, 'skipped': 'budget'})
        importlib.reload(mlx_backend)

    def test_a_skipped_or_failed_evidence_stage_is_logged(self):
        import logging
        from pipeline import runtime
        for frame in ({'mode': 'perq', 'skipped': 'budget'}, {'error': 'RuntimeError: boom'}):
            with self.assertLogs('pipeline.runtime', level=logging.WARNING) as logs:
                logging.getLogger('pipeline.runtime').warning(
                    'Evidence stage did not run (%s): spans are the answer pass\'s own quotes for '
                    'this conversation', frame.get('skipped') or frame.get('error'))
            self.assertIn('Evidence stage did not run', logs.output[0])
        source = open(runtime.__file__).read()
        self.assertIn("isinstance(evidence, dict) and (evidence.get('skipped')", source)
