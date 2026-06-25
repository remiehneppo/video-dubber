from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from dubber.mixing.ducking import build_commentary_filter


class FFmpegAdapter:
    def probe(self, input_path: Path) -> dict[str, Any]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(input_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return json.loads(result.stdout)

    def has_audio_stream(self, input_path: Path) -> bool:
        metadata = self.probe(input_path)
        return any(stream.get("codec_type") == "audio" for stream in metadata.get("streams", []))

    def duration_ms(self, input_path: Path) -> int:
        metadata = self.probe(input_path)
        duration = metadata.get("format", {}).get("duration", "0")
        return max(0, int(float(duration) * 1000))

    def extract_audio(self, input_video: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(output_wav),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def mix_commentary_audio(self, original_audio: Path, tts_audio: Path, output_audio: Path) -> None:
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(original_audio),
                "-i",
                str(tts_audio),
                "-filter_complex",
                build_commentary_filter(original_ducking_db=-22, final_loudness_normalization=True),
                "-map",
                "[out]",
                "-c:a",
                "pcm_s16le",
                str(output_audio),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def mux_video_audio(self, input_video: Path, input_audio: Path, output_video: Path) -> None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-i",
                str(input_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
