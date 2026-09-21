from faster_whisper import WhisperModel

MODEL = WhisperModel('large-v3', device='cpu', compute_type='float16')

def transcribe(audio_bytes: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
        f.write(audio_bytes)
        f.flush()
        segments, _ = MODEL.transcribe(f.name, language='en')

    return [
        {'start': segment.start, 'end': segment.end, 'text': segment.text}
        for segment in segments
    ]