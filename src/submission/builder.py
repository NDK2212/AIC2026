"""Row-ordering strategy tuned for the competition's ``Final Score``.

``R@k`` is a **max** over the first ``k`` rows, averaged over
``k ∈ {1, 5, 20, 50, 100}``.  Two consequences drive this module:

* the first few rows are worth far more than the rest, and five near-identical
  frames of one shot waste four of them - so the head is diversified by shot;
* the deep ranks are nearly free, and the ground-truth window ``[s, e]`` is
  narrow while keyframes are sparse - so the tail is padded with neighbouring
  frames of strong candidates.

Hence: diversity head, dense body, neighbour expansion.
"""

from __future__ import annotations

from typing import Sequence

from ..config import SubmissionConfig
from ..logging_utils import get_logger
from ..schemas import Candidate
from ..utils.keyframe_index import KeyframeIndex

log = get_logger(__name__)


def same_shot(a: Candidate, b: Candidate, shot_window: int) -> bool:
    """True when two candidates plausibly belong to the same shot."""
    return a.video_id == b.video_id and abs(a.frame_id - b.frame_id) < shot_window


def build_rows(
    candidates: Sequence[Candidate],
    cfg: SubmissionConfig,
    kf: KeyframeIndex | None = None,
) -> list[Candidate]:
    """Order candidates into the final submission rows.

    Guarantees: no duplicate ``(video_id, frame_id)``, at most ``max_rows``
    rows, and exactly ``min(max_rows, distinct candidates available)`` rows
    whenever expansion can supply them.
    """
    ordered = sorted(
        candidates, key=lambda c: (-c.score, c.video_id, c.frame_id)
    )
    if not ordered:
        return []

    # De-duplicate up front, keeping the best-scoring copy of each key.
    unique: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    for candidate in ordered:
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        unique.append(candidate)

    rows: list[Candidate] = []
    taken: set[tuple[str, int]] = set()

    # -- phase 1: diversity head -------------------------------------------
    # Two independent constraints, because R@1/R@5 can be lost two ways: by
    # spending them on one shot (which may miss the narrow [s, e] window) or on
    # one video (which scores zero outright if the video is wrong).
    head_target = min(cfg.top_diverse, cfg.max_rows)
    per_video: dict[str, int] = {}
    deferred: list[Candidate] = []
    for candidate in unique:
        if len(rows) >= head_target:
            break
        if any(same_shot(candidate, chosen, cfg.shot_window) for chosen in rows):
            deferred.append(candidate)
            continue
        if per_video.get(candidate.video_id, 0) >= cfg.head_max_per_video:
            deferred.append(candidate)
            continue
        rows.append(candidate)
        taken.add(candidate.key)
        per_video[candidate.video_id] = per_video.get(candidate.video_id, 0) + 1

    # Not enough distinct shots to fill the head - fall back to raw order.
    if len(rows) < head_target:
        for candidate in deferred:
            if len(rows) >= head_target:
                break
            if candidate.key in taken:
                continue
            rows.append(candidate)
            taken.add(candidate.key)

    # -- phase 2: dense body ------------------------------------------------
    for candidate in unique:
        if len(rows) >= cfg.max_rows:
            break
        if candidate.key in taken:
            continue
        rows.append(candidate)
        taken.add(candidate.key)

    # -- phase 3: neighbour expansion --------------------------------------
    if (
        cfg.neighbor_expansion.enabled
        and kf is not None
        and len(rows) < cfg.max_rows
    ):
        rows = _expand_neighbors(rows, taken, unique, cfg, kf)

    for i, candidate in enumerate(rows[: cfg.max_rows], start=1):
        rows[i - 1] = candidate.replace(rank=i)
    final = rows[: cfg.max_rows]
    log.info(
        "Built %d submission rows (head=%d over %d videos, expanded=%d)",
        len(final), head_target, len({c.video_id for c in final[:head_target]}),
        sum(1 for c in final if c.source == "expanded"),
    )
    return final


def _expand_neighbors(
    rows: list[Candidate],
    taken: set[tuple[str, int]],
    unique: Sequence[Candidate],
    cfg: SubmissionConfig,
    kf: KeyframeIndex,
) -> list[Candidate]:
    """Interleave neighbouring keyframes of strong candidates into the tail.

    Round-robin over the offsets so every strong candidate contributes its
    closest neighbour before any contributes its second-closest.
    """
    offsets = cfg.neighbor_expansion.offsets
    if not offsets:
        return rows

    sources = [c for c in unique if c.source != "expanded"][: cfg.max_rows]
    if not sources:
        return rows

    # neighbours[i] = the still-unused neighbours of sources[i], nearest first
    neighbours: list[list[Candidate]] = []
    for candidate in sources:
        frames = kf.neighbors(candidate.video_id, candidate.frame_id, offsets)
        bucket = [
            Candidate(
                video_id=candidate.video_id,
                frame_id=frame,
                # Keep expanded rows just below their parent so relative order
                # among them still reflects the parent's strength.
                score=candidate.score * 0.5 - 1e-6 * (i + 1),
                source="expanded",
                rank=0,
                extra={
                    "expanded_from": candidate.frame_id,
                    "offset": frame - candidate.frame_id,
                },
            )
            for i, frame in enumerate(frames)
        ]
        neighbours.append(bucket)

    start = max(0, min(cfg.neighbor_expansion.start_rank, len(rows)))
    head, tail = rows[:start], rows[start:]
    additions: list[Candidate] = []

    depth = 0
    max_depth = max((len(b) for b in neighbours), default=0)
    budget = cfg.max_rows - len(rows)
    while depth < max_depth and len(additions) < budget:
        for bucket in neighbours:
            if len(additions) >= budget:
                break
            if depth >= len(bucket):
                continue
            candidate = bucket[depth]
            if candidate.key in taken:
                continue
            taken.add(candidate.key)
            additions.append(candidate)
        depth += 1

    if not additions:
        return rows

    # Splice: keep the protected head, then interleave the expansion with the
    # remaining original rows so neither block is starved.
    merged = list(head)
    i = j = 0
    while i < len(tail) or j < len(additions):
        if i < len(tail):
            merged.append(tail[i])
            i += 1
        if j < len(additions):
            merged.append(additions[j])
            j += 1
    return merged


def head_video_diversity(rows: Sequence[Candidate], window: int) -> int:
    """How many distinct videos appear in the first ``window`` rows."""
    return len({c.video_id for c in rows[:window]})
