from __future__ import annotations

from pathlib import Path

from dubber.core.enums import StageStatus
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore


def test_segment_checkpoint_store_tracks_done_counts_and_incomplete_ids(tmp_path: Path) -> None:
    state_path = tmp_path / "asr_segments.json"
    store = SegmentCheckpointStore.create(
        state_path,
        stage="asr",
        segment_ids=["seg_000001", "seg_000002", "seg_000003"],
    )

    store.mark("seg_000001", StageStatus.COMPLETED, artifact="raw/asr/seg_000001.json")
    store.mark("seg_000002", StageStatus.FAILED, error="provider timeout")
    store.save()

    reloaded = SegmentCheckpointStore.load(state_path)

    assert reloaded.done_count == 1
    assert reloaded.total_count == 3
    assert reloaded.incomplete_segment_ids() == ["seg_000002", "seg_000003"]
    assert reloaded.segments["seg_000001"].artifact == "raw/asr/seg_000001.json"
    assert reloaded.segments["seg_000002"].error == "provider timeout"


def test_segment_checkpoint_store_rejects_unknown_segment_id(tmp_path: Path) -> None:
    store = SegmentCheckpointStore.create(tmp_path / "tts_segments.json", stage="tts", segment_ids=["seg_000001"])

    try:
        store.mark("seg_999999", StageStatus.COMPLETED)
    except KeyError as exc:
        assert "Unknown segment_id" in str(exc)
    else:
        raise AssertionError("Expected unknown segment_id to raise")
