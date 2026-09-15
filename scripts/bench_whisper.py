"""Time faster-whisper transcription on this machine.

    python scripts/bench_whisper.py                     # JFK sample, model "small"
    python scripts/bench_whisper.py audio.wav medium
"""
import platform
import sys
import time
import urllib.request
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

SAMPLE_URL = "https://github.com/SYSTRAN/faster-whisper/raw/master/tests/data/jfk.flac"

audio = sys.argv[1] if len(sys.argv) > 1 else "data/jfk.flac"
size = sys.argv[2] if len(sys.argv) > 2 else "small"

if not Path(audio).exists():
    Path(audio).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SAMPLE_URL, audio)

duration = sf.info(audio).duration

# faster-whisper (CTranslate2) has no MPS backend: on a Mac this is CPU, which is still fast.
try:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
compute_type = "float16" if device == "cuda" else "int8"

t0 = time.perf_counter()
model = WhisperModel(size, device=device, compute_type=compute_type)
load = time.perf_counter() - t0

t0 = time.perf_counter()
segments, info = model.transcribe(audio, beam_size=5)
text = " ".join(s.text.strip() for s in segments)  # generator: work happens here
elapsed = time.perf_counter() - t0

print(f"machine : {platform.node()} ({platform.machine()})")
print(f"model   : {size}  device {device}  compute {compute_type}  (load {load:.1f}s)")
print(f"audio   : {duration:.1f}s  language {info.language}")
print(f"time    : {elapsed:.2f}s  -> {duration / elapsed:.1f}x realtime")
print(f"text    : {text[:120]}")
