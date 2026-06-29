from __future__ import annotations


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def build_commentary_filter(
    *,
    original_ducking_db: float = -22,
    tts_boost_db: float = 8.0,
    final_loudness_normalization: bool = True,
) -> str:
    tts_gain = db_to_linear(tts_boost_db)
    mixed_label = "mixed"
    threshold = max(0.001, min(0.25, db_to_linear(original_ducking_db)))
    graph = (
        f"[1:a]volume={tts_gain:.6f},asplit=2[tts][sidechain];"
        f"[0:a][sidechain]sidechaincompress=threshold={threshold:.6f}:ratio=10:attack=20:release=350:makeup=1[ducked];"
        f"[ducked][tts]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[{mixed_label}]"
    )
    if final_loudness_normalization:
        graph += f";[{mixed_label}]loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    else:
        graph += f";[{mixed_label}]anull[out]"
    return graph
