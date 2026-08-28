"""Shared fixtures.  Nothing here touches Qdrant, Elasticsearch or any LLM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NeighborExpansionConfig, SubmissionConfig, TrakeConfig  # noqa: E402
from src.schemas import Candidate  # noqa: E402


def make_candidate(video: str, frame: int, score: float, source: str = "fused", rank: int = 0):
    """Terse candidate constructor for tests."""
    return Candidate(video_id=video, frame_id=frame, score=score, source=source, rank=rank)


@pytest.fixture
def submission_cfg() -> SubmissionConfig:
    return SubmissionConfig(
        max_rows=100,
        top_diverse=8,
        head_max_per_video=3,
        shot_window=60,
        neighbor_expansion=NeighborExpansionConfig(
            enabled=True, start_rank=15, offsets=[-15, 15, -30, 30]
        ),
    )


@pytest.fixture
def trake_cfg() -> TrakeConfig:
    return TrakeConfig(
        anchor_index=1,
        per_step_topk=150,
        max_videos=40,
        paths_per_video=3,
        coverage_bonus=0.5,
        miss_penalty=-0.35,
        min_gap=1,
        max_gap=None,
        allow_fill=True,
    )


class FakeKeyframeIndex:
    """A KeyframeIndex stand-in backed by an in-memory frame table."""

    def __init__(self, frames: dict[str, list[int]]):
        self.frames = {v: sorted(f) for v, f in frames.items()}

    def all_frames(self, video_id: str) -> list[int]:
        return list(self.frames.get(video_id, []))

    def nearest_frame(self, video_id: str, frame_id: int) -> int:
        available = self.frames.get(video_id)
        if not available:
            return int(frame_id)
        return min(available, key=lambda f: (abs(f - frame_id), f))

    def median_gap(self, video_id: str) -> int:
        available = self.frames.get(video_id, [])
        if len(available) < 2:
            return 25
        gaps = sorted(b - a for a, b in zip(available, available[1:]))
        return gaps[len(gaps) // 2]

    def neighbors(self, video_id: str, frame_id: int, offsets) -> list[int]:
        available = self.frames.get(video_id, [])
        if not available:
            return []
        out: list[int] = []
        seen = {frame_id}
        for offset in offsets:
            candidate = min(available, key=lambda f: (abs(f - (frame_id + offset)), f))
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
        return out


@pytest.fixture
def fake_kf() -> FakeKeyframeIndex:
    # Three videos, keyframes every 10 frames.
    return FakeKeyframeIndex(
        {
            "L01_V001": list(range(0, 1000, 10)),
            "L01_V002": list(range(0, 1000, 10)),
            "L02_V003": list(range(0, 1000, 10)),
        }
    )
