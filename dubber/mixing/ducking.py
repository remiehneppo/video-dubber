from __future__ import annotations


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def build_commentary_filter(
    *,
    original_ducking_db: float = -22,
    final_loudness_normalization: bool = True,
) -> str:
    duck_gain = db_to_linear(original_ducking_db)
    mixed_label = "mixed"
    graph = (
        f"[0:a]volume={duck_gain:.6f}[ducked];"
        f"[1:a]volume=1.0[tts];"
        f"[ducked][tts]amix=inputs=2:duration=longest:dropout_transition=0[{mixed_label}]"
    )
    if final_loudness_normalization:
        graph += f";[{mixed_label}]loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    else:
        graph += f";[{mixed_label}]anull[out]"
    return graph
