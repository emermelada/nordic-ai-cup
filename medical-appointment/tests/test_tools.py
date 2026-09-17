import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pipeline import mlx_backend
from tools import eval_offline, transcribe_all

ROOT = Path(__file__).resolve().parents[1]
WORDS = [
    {'word': ' No', 'start': 0.0, 'end': 1.0, 'p': 0.9},
    {'word': ' fever', 'start': 1.0, 'end': 2.0, 'p': 0.9},
    {'word': ' today.', 'start': 2.0, 'end': 3.0, 'p': 0.9},
]
ROWS = [
    {'transcript_id': 'sample_x', 'question_id': 'sample_x_yes',
     'question': 'Was the patient free of fever?', 'question_type': 'positive',
     'label': '1', 'evidence_start': '0', 'evidence_end': '3'},
    {'transcript_id': 'sample_x', 'question_id': 'sample_x_no',
     'question': 'Was aspirin prescribed?', 'question_type': 'off_topic',
     'label': '0', 'evidence_start': '', 'evidence_end': ''},
]
RAW = json.dumps({'results': [
    {'q': 1, 'answer': 'yes', 'units': [0], 'quote': 'No fever today.'},
    {'q': 2, 'answer': 'no', 'units': [], 'quote': ''},
]})


def transcript():
    return {
        'model': mlx_backend.WHISPER_MODEL, 'seconds': 2.0, 'duration': 4.0,
        'text': ' No fever today.',
        'segments': [{'start': 0.0, 'end': 3.0, 'text': ' No fever today.',
                      'words': copy.deepcopy(WORDS)}],
    }


def replay_dump():
    return {'sample_x': {
        'raw': RAW, 'seconds': 4.5, 'duration': 4.0,
        'units': [{'words': copy.deepcopy(WORDS)}],
        'questions': [{'id': row['question_id'], 'q': row['question'],
                       'label': 'wrong-on-purpose'} for row in ROWS],
    }}


class BackendTests(unittest.TestCase):
    def test_import_and_constructor_do_not_import_models(self):
        subprocess.run([
            sys.executable, '-c',
            'import sys; from pipeline.mlx_backend import MLXBackend; MLXBackend(); '
            'assert not any(name in sys.modules for name in '
            '("mlx", "mlx_whisper", "mlx_lm", "huggingface_hub", "numpy", "av"))',
        ], cwd=ROOT, check=True, capture_output=True)

    def test_snapshot_resolution_is_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            download = Mock(return_value=directory)
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(snapshot_download=download)}), \
                    patch.dict(os.environ, {'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'}):
                self.assertEqual(mlx_backend.resolve_snapshot('model/id'), directory)
                download.assert_called_once_with('model/id', local_files_only=True)
                self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
                self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')

    def test_transcribe_loads_only_asr_and_returns_decoded_duration(self):
        result = transcript()
        result['segments'][0]['words'][0]['probability'] = 0.8
        transcribe = Mock(return_value=result)
        samples = [0.0] * 32000
        holder = SimpleNamespace(model='loaded', model_path='/cached/whisper')
        modules = {
            'mlx_whisper': SimpleNamespace(transcribe=transcribe),
            'mlx_whisper.transcribe': SimpleNamespace(ModelHolder=holder),
            'mlx_lm': None,
        }
        with patch.object(mlx_backend, 'decode_audio', return_value=samples), \
                patch.object(mlx_backend, 'resolve_snapshot', return_value='/cached/whisper') as resolve, \
                patch.dict(sys.modules, modules):
            backend = mlx_backend.MLXBackend()
            output = backend.transcribe(b'mp3')
        resolve.assert_called_once_with(mlx_backend.WHISPER_MODEL)
        # The answering model needs the memory; ASR weights are reloaded from page cache.
        self.assertIsNone(holder.model)
        self.assertEqual(output['duration'], 2.0)
        self.assertEqual(output['segments'][0]['words'][0]['p'], 0.8)
        self.assertIsNone(backend._llm)
        transcribe.assert_called_once_with(
            samples, path_or_hf_repo='/cached/whisper', **mlx_backend.ASR_SETTINGS
        )

    def test_complete_uses_exact_template_flags_and_generation_settings(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = 'prompt'
        load = Mock(return_value=('model', tokenizer))
        generate = Mock(return_value=RAW)
        sampler = Mock(return_value='sampler')
        modules = {
            'mlx_lm': SimpleNamespace(load=load, generate=generate),
            'mlx_lm.sample_utils': SimpleNamespace(make_sampler=sampler),
        }
        with patch.dict(sys.modules, modules), \
                patch.object(mlx_backend, 'resolve_snapshot', return_value='/cached/qwen'), \
                patch('pipeline.core.build_messages', return_value=['messages']) as messages:
            backend = mlx_backend.MLXBackend(prompt='legacy')
            self.assertEqual(backend.complete(WORDS, ['question']), RAW)
            self.assertIsNone(backend._whisper_path)
            self.assertGreaterEqual(backend.last_generation_seconds, 0)
            messages.assert_called_once_with(WORDS, ['question'])
            load.assert_called_once_with('/cached/qwen', tokenizer_config={'local_files_only': True})
            tokenizer.apply_chat_template.assert_called_once_with(
                ['messages'], add_generation_prompt=True, tokenize=False,
                enable_thinking=False, reasoning_effort='low',
            )
            sampler.assert_called_once_with(temp=0.0)
            generate.assert_called_once_with(
                'model', tokenizer, prompt='prompt', max_tokens=1200,
                sampler='sampler', verbose=False,
            )
            tokenizer.apply_chat_template.side_effect = TypeError('thinking flag unsupported')
            generate.reset_mock()
            with self.assertRaises(TypeError):
                backend.complete(WORDS, ['question'])
            generate.assert_not_called()

    def test_serving_default_uses_compact_prompt(self):
        backend = mlx_backend.MLXBackend()
        with patch('pipeline.evidence.build_compact_messages', return_value=['compact']) as messages, \
                patch.object(backend, 'generate_messages', return_value=RAW) as generate:
            self.assertEqual(backend.complete(WORDS, ['question']), RAW)
        messages.assert_called_once_with(WORDS, ['question'])
        generate.assert_called_once_with(['compact'], 1200)
        with self.assertRaises(ValueError):
            mlx_backend.MLXBackend(prompt='missing')

    def test_warmup_runs_minimal_generation(self):
        backend = mlx_backend.MLXBackend()
        zeros = Mock(return_value='silence')
        with patch.dict(sys.modules, {'numpy': SimpleNamespace(zeros=zeros, float32='float32')}), \
                patch.object(backend, '_transcribe') as asr, patch.object(backend, '_generate') as llm:
            backend.warmup()
        zeros.assert_called_once_with(16000, dtype='float32')
        asr.assert_called_once_with('silence')
        self.assertEqual(llm.call_args.kwargs['max_tokens'], 1)

    def test_decode_uses_sample_count_and_mono_float32(self):
        audio = io.BytesIO()
        with wave.open(audio, 'wb') as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(b'\x00\x00\x00\x00' * 8000)
        samples = mlx_backend.decode_audio(audio.getvalue())
        self.assertEqual(samples.shape, (16000,))
        self.assertEqual(str(samples.dtype), 'float32')
        with self.assertRaises(ValueError):
            mlx_backend.decode_audio(b'')


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.conversations = [('conversation_sample_x.mp3', ROWS)]
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(eval_offline, 'load_sample_audio', return_value=b'audio'))
        self.enterContext(patch.object(eval_offline, 'decode_audio', return_value=[0.0] * 64000))

    def test_atomic_writes_never_overwrite_unknown_files(self):
        target = self.directory / 'output.json'
        target.write_text('user content')
        with self.assertRaises(FileExistsError):
            transcribe_all.atomic_write_json(target, {'new': True})
        self.assertEqual(target.read_text(), 'user content')
        self.assertEqual(list(self.directory.iterdir()), [target])

    def test_cache_resume_checks_hash_and_complete_shape(self):
        settings = {'producer': transcribe_all.CACHE_PRODUCER, 'version': 1}
        backend = Mock()
        backend.transcribe.side_effect = lambda audio: transcript()
        with patch.object(transcribe_all, 'cache_settings', return_value=settings), \
                patch.object(transcribe_all, 'group_questions_by_conversation', return_value=self.conversations), \
                patch.object(transcribe_all, 'load_sample_audio', return_value=b'audio') as audio, \
                patch.object(transcribe_all, 'MLXBackend', return_value=backend):
            args = ['--output', str(self.directory)]
            transcribe_all.main(args)
            transcribe_all.main(args)
            self.assertEqual(backend.transcribe.call_count, 1)
            audio.return_value = b'changed audio'
            transcribe_all.main(args)
            self.assertEqual(backend.transcribe.call_count, 2)
            target = self.directory / 'conversation_sample_x.json'
            damaged = json.loads(target.read_text())
            del damaged['segments']
            target.write_text(json.dumps(damaged))
            transcribe_all.main(args)
            self.assertEqual(backend.transcribe.call_count, 3)

    def test_legacy_cache_is_not_overwritten(self):
        target = self.directory / 'conversation_sample_x.json'
        original = json.dumps(transcript())
        target.write_text(original)
        with patch.object(transcribe_all, 'cache_settings', return_value={}), \
                patch.object(transcribe_all, 'group_questions_by_conversation', return_value=self.conversations), \
                patch.object(transcribe_all, 'load_sample_audio', return_value=b'audio'), \
                patch.object(transcribe_all, 'MLXBackend') as backend:
            with self.assertRaises(FileExistsError):
                transcribe_all.main(['--output', str(self.directory)])
            backend.return_value.transcribe.assert_not_called()
        self.assertEqual(target.read_text(), original)

    def test_replay_uses_official_gold_and_preserves_timing_semantics(self):
        prepared = eval_offline.prepare_inputs(self.conversations, None, replay_dump())
        with patch.object(eval_offline, 'MLXBackend') as backend:
            summary = eval_offline.evaluate(
                prepared, replay=True, retrieval_only=False, start_offset=0,
                output=self.directory,
            )
            backend.assert_not_called()
        self.assertEqual(summary['questions'], 2)
        self.assertEqual(summary['correct'], 2)
        self.assertEqual(summary['score'], 1.0)
        self.assertIsNone(summary['http_latency_seconds'])
        timings = summary['timing_seconds']
        self.assertEqual(timings['historical_generation_seconds']['mean'], 4.5)
        self.assertEqual(timings['fresh_generation_seconds']['count'], 0)
        self.assertEqual(summary['diagnostics_not_scored']['right_passage_count'], 1)
        saved = json.loads((self.directory / 'raw_outputs.json').read_text())
        self.assertEqual(saved['sample_x']['questions'][0]['label'], '1')
        replayed = eval_offline.prepare_inputs(self.conversations, None, saved)
        self.assertEqual(replayed[0]['words'], prepared[0]['words'])
        self.assertTrue((self.directory / 'questions.csv').is_file())

    def test_replay_retains_recorded_alignment_and_prompt(self):
        dump = replay_dump()
        dump['sample_x'].update(prompt='focused', alignment='numeric', model='recorded/model')
        prepared = eval_offline.prepare_inputs(self.conversations, None, dump)
        with patch.object(eval_offline, 'answer_response', wraps=eval_offline.answer_response) as answer:
            summary = eval_offline.evaluate(
                prepared, replay=True, retrieval_only=False, start_offset=0, output=self.directory,
                subset='holdout',
            )
        self.assertEqual(answer.call_args.kwargs['alignment'], 'numeric')
        self.assertEqual(summary['prompt'], ['focused'])
        self.assertEqual(summary['alignment'], ['numeric'])
        self.assertEqual(summary['subset'], 'holdout')
        saved = json.loads((self.directory / 'raw_outputs.json').read_text())
        self.assertEqual(saved['sample_x']['model'], 'recorded/model')

    def test_missing_conversation_and_changed_questions_fail(self):
        with self.assertRaises(FileNotFoundError):
            eval_offline.prepare_inputs(self.conversations, None, {})
        dump = replay_dump()
        dump['sample_x']['questions'][0]['q'] = 'different question'
        with self.assertRaisesRegex(ValueError, 'question text/order'):
            eval_offline.prepare_inputs(self.conversations, None, dump)
        with self.assertRaisesRegex(FileNotFoundError, 'Missing transcript'):
            eval_offline.prepare_inputs(self.conversations, self.directory, replay_dump())

    def test_replay_rejects_changed_words(self):
        changed = transcript()
        changed['segments'][0]['words'][0]['word'] = ' Yes'
        (self.directory / 'conversation_sample_x.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'differs from replay'):
            eval_offline.prepare_inputs(self.conversations, self.directory, replay_dump())

    def test_fresh_generation_never_receives_gold(self):
        prepared = eval_offline.prepare_inputs(self.conversations, None, replay_dump())
        backend = Mock(last_generation_seconds=1.25)
        backend.complete.return_value = RAW
        with patch.object(eval_offline, 'MLXBackend', return_value=backend):
            summary = eval_offline.evaluate(
                prepared, replay=False, retrieval_only=False, start_offset=0,
                output=self.directory,
            )
        backend.complete.assert_called_once_with(WORDS, [row['question'] for row in ROWS])
        self.assertEqual(summary['timing_seconds']['fresh_generation_seconds']['mean'], 1.25)
        self.assertEqual(summary['timing_seconds']['historical_generation_seconds']['count'], 0)

    def test_output_collision_fails_before_generation(self):
        (self.directory / 'summary.json').write_text('user content')
        prepared = eval_offline.prepare_inputs(self.conversations, None, replay_dump())
        with patch.object(eval_offline, 'MLXBackend') as backend:
            with self.assertRaises(FileExistsError):
                eval_offline.evaluate(prepared, replay=False, retrieval_only=False,
                                      start_offset=0, output=self.directory)
            backend.assert_not_called()

    def test_cli_entrypoints(self):
        for name in ('transcribe_all', 'eval_offline'):
            for command in (['-m', f'tools.{name}'], [str(ROOT / 'tools' / f'{name}.py')]):
                result = subprocess.run([sys.executable, *command, '--help'], cwd=ROOT,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--limit', result.stdout)


if __name__ == '__main__':
    unittest.main()
