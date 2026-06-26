from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from dubber.core.paths import WorkspacePaths
from dubber.providers.base import TTSResult
from dubber.providers.factory import ProviderBundle
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_segment


def test_produce_mock_tts_segment_returns_manifest_row(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }

    row = produce_mock_tts_segment(paths=paths, segment=segment)

    assert row["segment_id"] == "seg_000001"
    assert row["orig_duration_ms"] == 1000
    assert row["tts_duration_ms"] == 1100
    assert row["alignment_action"] == "time_stretch"
    assert row["raw_audio_path"] == "tts/seg_000001.raw.wav"
    assert row["audio_path"] == "tts/seg_000001.wav"
    assert (paths.root / row["audio_path"]).exists()
    with wave.open(str(paths.root / row["audio_path"]), "rb") as wav:
        assert wav.getnchannels() == 1


class FakeProviderTTS:
    def __init__(self) -> None:
        self.voice = "Trúc Ly"
        self.seen_voice = ""

    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        self.seen_voice = voice
        output_path.write_bytes(b"fake audio")
        return TTSResult(audio_path=output_path, duration_ms=900, provider_metadata={"voice": voice})


class FakeFFmpeg:
    def duration_ms(self, audio_path: Path) -> int:
        return 900


def test_produce_provider_tts_segment_uses_provider_voice(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = FakeProviderTTS()
    providers = ProviderBundle(asr=None, llm=None, tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Xin chào",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
        )
    )

    assert tts.seen_voice == "Trúc Ly"
    assert row["provider_metadata"] == {"voice": "Trúc Ly"}
