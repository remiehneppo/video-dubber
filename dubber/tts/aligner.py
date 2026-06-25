from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def build_atempo_chain(ratio: float) -> str:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    factors: list[float] = []
    remaining = ratio
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def apply_time_stretch(input_wav: Path, output_wav: Path, ratio: float) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if abs(ratio - 1.0) < 0.001:
        shutil.copy2(input_wav, output_wav)
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-filter:a",
            build_atempo_chain(ratio),
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
