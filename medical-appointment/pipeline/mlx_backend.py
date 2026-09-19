"""Local-cache-only Whisper and Qwen inference; importing this module is cheap."""

import io
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# The 8-bit ASR is half the size of the fp16 one and scored the same; mlx-whisper wants
# the weights named weights.safetensors, so it is staged locally (see RUNNING.md).
WHISPER_MODEL = 'mlx-community/whisper-large-v3-turbo'
# A 16 GB answering model leaves no room for idle ASR weights on a 24 GB machine.
RELEASE_ASR_AFTER_TRANSCRIBE = True
# The 19B MoE refines evidence far better than the 9B (0.675 vs 0.628 tIoU on the
# training set) but its 14 GB leaves no room for a second dense model beside it.
LLM_MODEL = 'mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit-REAP-19B'
# The second model can reject a primary yes; its evidence is diagnostic only.
# None disables the second pass.
SECOND_LLM_MODEL = None
# Skip the second pass when the first already ran long, so it cannot cost the deadline.
SECOND_PASS_BUDGET_SECONDS = 25.0
# Stage B re-selects the evidence of retained yes answers over a sentence table.
# None disables it. 'refine' is one batched call with the annotation conventions;
# 'perq' is one call per yes over the whole transcript (parallel on a server backend);
# 'window' is one bare call per yes over a local window, for an adapter trained on it;
# 'locate' finds sentences globally, then selects word boundaries within that passage.
EVIDENCE_MODEL = LLM_MODEL
EVIDENCE_MODE = os.environ.get('MEDICAL_EVIDENCE_MODE', 'refine')
EVIDENCE_ADAPTER = None
# Stage B starts only while the request is young enough to finish inside the deadline. The
# clean path reaches it at about 11 s, so this is a wide margin; it was 32 s, which a
# validation run alongside an offline experiment crossed on 6 of 19 conversations, silently
# dropping the evidence stage and about 0.017 raw. Keep it well under the 52 s watchdog.
EVIDENCE_BUDGET_SECONDS = float(os.environ.get('MEDICAL_EVIDENCE_BUDGET', '40'))
# Refine outputs are 90 tokens on median and under 160 except when the model loops.
EVIDENCE_MAX_TOKENS = {'refine': 320, 'window': 60, 'perq': 120, 'locate': 64}
# A second per-question pass whose prompt deliberately differs from the first, so the
# evidence vote has two disagreeing generations to place against the extractor's span.
EVIDENCE_VOTE = os.environ.get('MEDICAL_EVIDENCE_VOTE') == '1'
DEFAULT_PROMPT = os.environ.get('MEDICAL_ANSWER_PROMPT', 'compact')
SAMPLE_RATE = 16000
BACKEND_VERSION = 1
ASR_SETTINGS = {
    'language': 'en',
    'word_timestamps': True,
    'temperature': 0.0,
    'condition_on_previous_text': False,
}


def resolve_snapshot(model_id: str) -> str:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    local = Path(__file__).resolve().parent.parent / model_id
    if local.is_dir():
        return str(local)
    from huggingface_hub import snapshot_download

    try:
        snapshot = Path(snapshot_download(model_id, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(f'Local model cache unavailable for {model_id}: {exc}') from exc
    if not snapshot.is_dir():
        raise RuntimeError(f'Local model snapshot is missing: {snapshot}')
    return str(snapshot)


def decode_audio(audio_bytes: bytes):
    """Decode in-process so terminating inference cannot orphan an ffmpeg child."""
    import av
    import numpy as np

    if not audio_bytes:
        raise ValueError('Audio is empty')
    chunks = []
    with av.open(io.BytesIO(audio_bytes)) as container:
        if not container.streams.audio:
            raise ValueError('Input contains no audio stream')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            chunks.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(frame))
        chunks.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(None))
    if not chunks:
        raise ValueError('Input decoded to no audio samples')
    samples = np.concatenate(chunks).astype(np.float32, copy=False)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('Input decoded to empty or non-finite audio')
    return samples


def release_asr():
    """Drop cached Whisper weights so the answering model keeps the memory."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = None
    ModelHolder.model_path = None
    mx.clear_cache()


class MLXBackend:
    def __init__(self, prompt=DEFAULT_PROMPT):
        if prompt not in ('legacy', 'focused', 'compact', 'minimal', 'named'):
            raise ValueError(f'Unknown answer prompt: {prompt}')
        self.prompt = prompt
        self._whisper_path = None
        self._models = {}
        self._sampler = None
        self.last_generation_seconds = None

    def _transcribe(self, samples):
        if self._whisper_path is None:
            self._whisper_path = resolve_snapshot(WHISPER_MODEL)
        import mlx_whisper

        return mlx_whisper.transcribe(
            samples, path_or_hf_repo=self._whisper_path, **ASR_SETTINGS
        )

    def transcribe(self, audio_bytes: bytes) -> dict:
        from pipeline.evidence import energy_envelope

        started = time.monotonic()
        samples = decode_audio(audio_bytes)
        try:
            result = self._transcribe(samples)
        finally:
            if RELEASE_ASR_AFTER_TRANSCRIBE:
                release_asr()
        segments = [
            {
                'start': float(segment['start']),
                'end': float(segment['end']),
                'text': segment['text'],
                'words': [
                    {
                        'word': word['word'],
                        'start': float(word['start']),
                        'end': float(word['end']),
                        'p': (float(word['probability'])
                              if word.get('probability') is not None else None),
                    }
                    for word in segment.get('words', [])
                ],
            }
            for segment in result['segments']
        ]
        return {
            'model': WHISPER_MODEL,
            'seconds': time.monotonic() - started,
            'duration': len(samples) / SAMPLE_RATE,
            'segments': segments,
            'text': result['text'],
            'energy_db': energy_envelope(samples),
        }

    def _generate(self, words: list[dict], questions: list[str], max_tokens: int,
                  model_id: str | None = None) -> str:
        from pipeline.core import build_messages
        from pipeline.evidence import (
            build_compact_messages, build_focused_messages, build_minimal_messages,
            build_named_messages,
        )

        builder = {'legacy': build_messages, 'focused': build_focused_messages,
                   'compact': build_compact_messages, 'minimal': build_minimal_messages,
                   'named': build_named_messages}[self.prompt]
        return self.generate_messages(builder(words, questions), max_tokens, model_id)

    def generate_messages(self, messages: list[dict], max_tokens: int = 900,
                          model_id: str | None = None, adapter: str | None = None) -> str:
        model_id = model_id or LLM_MODEL
        key = (model_id, adapter)
        if key not in self._models:
            snapshot = resolve_snapshot(model_id)
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler

            options = {'tokenizer_config': {'local_files_only': True}}
            if adapter:
                options['adapter_path'] = adapter
            self._models[key] = load(snapshot, **options)
            self._sampler = make_sampler(temp=0.0)
        model, tokenizer = self._models[key]
        from mlx_lm import generate

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            # Templates ignore the flag they do not define: Qwen reads enable_thinking,
            # gpt-oss reads reasoning_effort. Both keep reasoning short enough to fit.
            enable_thinking=False,
            reasoning_effort='low',
        )
        started = time.monotonic()
        result = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=self._sampler, verbose=False,
        )
        self.last_generation_seconds = time.monotonic() - started
        return result

    def complete(self, words: list[dict], questions: list[str]) -> str:
        # gpt-oss reasons before answering: 769 output tokens at most over the training set.
        return self._generate(words, questions, max_tokens=1200)

    def complete_second(self, words: list[dict], questions: list[str]) -> str:
        if not SECOND_LLM_MODEL:
            return ''
        if (self.last_generation_seconds or 0) > SECOND_PASS_BUDGET_SECONDS:
            return ''
        return self._generate(words, questions, max_tokens=1200, model_id=SECOND_LLM_MODEL)

    def complete_evidence(self, words, questions, answers, drafts, anchors, request_started):
        """Stage-B output for the parent to apply, or None when disabled or out of budget."""
        if not EVIDENCE_MODEL or not any(answers):
            return None
        started = time.monotonic()
        if started - request_started > EVIDENCE_BUDGET_SECONDS:
            return {'mode': EVIDENCE_MODE, 'skipped': 'budget'}
        if EVIDENCE_MODE == 'locate':
            from pipeline.locate import locate_evidence

            frame = locate_evidence(
                words, questions, answers,
                lambda phase, prompts: self.generate_many(prompts, EVIDENCE_MAX_TOKENS['locate'], request_started),
                deadline=request_started + EVIDENCE_BUDGET_SECONDS,
            )
            return {**frame, 'seconds': time.monotonic() - started}
        from pipeline.stage_b import (
            build_perq_messages, build_refine_messages, build_window_messages,
            render_sentences, sentence_ranges,
        )

        if EVIDENCE_MODE == 'refine':
            messages = build_refine_messages(words, questions, answers, drafts)
            raw = self.generate_messages(messages, EVIDENCE_MAX_TOKENS['refine'],
                                         EVIDENCE_MODEL, EVIDENCE_ADAPTER)
            return {'mode': 'refine', 'raw': raw, 'seconds': time.monotonic() - started}

        sentences = sentence_ranges(words)
        if EVIDENCE_MODE == 'perq':
            from pipeline.span_examples import examples_for

            retrieved = os.environ.get('MEDICAL_EVIDENCE_PROMPT') == 'retrieved'
            transcript = render_sentences(words, sentences)
            prompts = [(i + 1, build_perq_messages(
                transcript, question, drafts.get(i + 1, ''),
                examples=examples_for(questions, question) if retrieved else None,
            )) for i, question in enumerate(questions) if answers[i]]
            if EVIDENCE_VOTE:
                # The opposite prompt of whichever is serving, negated keys keeping the
                # two sets apart in one batched call.
                try:
                    prompts += [(-(i + 1), build_perq_messages(
                        transcript, question, drafts.get(i + 1, ''),
                        examples=None if retrieved else examples_for(questions, question),
                    )) for i, question in enumerate(questions) if answers[i]]
                except Exception:
                    logger.exception('Second evidence prompt unavailable; the vote loses a producer')
        else:
            from rank_bm25 import BM25Okapi
            from pipeline.stage_b import _content, sentence_text

            ranker = BM25Okapi([_content(sentence_text(words, span)) for span in sentences])
            prompts = [(i + 1, build_window_messages(words, question, anchors[i], sentences, ranker))
                       for i, question in enumerate(questions) if answers[i]]
        outputs = self.generate_many(prompts, EVIDENCE_MAX_TOKENS[EVIDENCE_MODE], request_started)
        frame = {'mode': EVIDENCE_MODE, 'outputs': {k: v for k, v in outputs.items() if k > 0},
                 'seconds': time.monotonic() - started}
        if EVIDENCE_VOTE:
            frame['vote_outputs'] = {-k: v for k, v in outputs.items() if k < 0}
        return frame

    def complete_rescue(self, words, questions, answers, request_started):
        """P(yes) for each question currently answered no, or {} without token probabilities."""
        from pipeline.rescue import build_rescue_messages, pending
        from pipeline.stage_b import render_sentences

        numbers = pending(answers)
        if not EVIDENCE_MODEL or not numbers:
            return {}
        # A re-ask over the same garbled text repeats the same mistake, so the backend may
        # offer a more accurate reading of the audio; the spans still come from `words`.
        transcript = getattr(self, 'rescue_reading', None) or render_sentences(words)
        prompts = [(number, build_rescue_messages(transcript, questions[number - 1]))
                   for number in numbers]
        return self.score_yes(prompts, request_started)

    def score_yes(self, prompts, request_started):
        """P(yes) per prompt. Needs token probabilities, which only the server backend exposes."""
        return {}

    def generate_many(self, prompts, max_tokens, request_started):
        """One generation per prompt, in order, stopping at the stage-B budget."""
        outputs = {}
        for key, messages in prompts:
            if time.monotonic() - request_started > EVIDENCE_BUDGET_SECONDS:
                break
            outputs[key] = self.generate_messages(messages, max_tokens, EVIDENCE_MODEL, EVIDENCE_ADAPTER)
        return outputs

    def warmup(self) -> None:
        import numpy as np

        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self._generate([], ['Was fever discussed?'], max_tokens=1)
        if SECOND_LLM_MODEL:
            self._generate([], ['Was fever discussed?'], max_tokens=1, model_id=SECOND_LLM_MODEL)
        if EVIDENCE_MODEL:
            self.generate_messages([{'role': 'user', 'content': 'Warm up.'}], 1,
                                   EVIDENCE_MODEL, EVIDENCE_ADAPTER)
        self.last_generation_seconds = None
