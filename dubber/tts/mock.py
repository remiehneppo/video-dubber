from __future__ import annotations

import math
import wave
from pathlib import Path


def synthesize_tone_wav(output_path: Path, duration_ms: int, *, sample_rate: int = 44100) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(sample_rate * max(duration_ms, 100) / 1000))
    amplitude = 2800
    frequency = 220.0
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(frames))
