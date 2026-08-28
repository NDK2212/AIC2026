"""Rank fusion across the OCR / ASR / visual retrieval paths.

Three strategies, all keyed on ``(video_id, frame_id)``:

* :func:`rrf` - plain Reciprocal Rank Fusion.
* :func:`weighted_rrf` - RRF with a per-path weight (Adaptive Score Fusion).
* :func:`weighted_norm` - min-max normalise each path's scores, then add.

A document that only appears in one path is never penalised; it simply gets
contributions from that path alone.  Every ordering has an explicit tie-break so
identical inputs always produce identical outputs.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..logging_utils import get_logger
from ..schemas import Candidate

log = get_logger(__name__)

Paths = Mapping[str, Sequence[Candidate]]


def _accumulate(
    paths: Paths,
    weights: Mapping[str, float] | None,
    contribution,
) -> list[Candidate]:
    """Shared fusion body: sum per-path contributions, then sort.

    ``contribution(path, rank, candidate, path_candidates) -> float`` returns the
    score one path awards one candidate.
    """
    totals: dict[tuple[str, int], float] = {}
    per_path_rank: dict[tuple[str, int], dict[str, int]] = {}
    per_path_score: dict[tuple[str, int], dict[str, float]] = {}
    extras: dict[tuple[str, int], dict[str, Any]] = {}

    for path, candidates in paths.items():
        if not candidates:
            continue
        weight = 1.0 if weights is None else float(weights.get(path, 0.0))
        # Rank by list order; ``rank`` on the candidate is informational only.
        for position, candidate in enumerate(candidates, start=1):
            key = candidate.key
            totals[key] = totals.get(key, 0.0) + weight * contribution(
                path, position, candidate, candidates
            )
            ranks = per_path_rank.setdefault(key, {})
            # A path may return the same key twice (e.g. manual multi-vector
            # fallback) - keep the best rank it achieved.
            if path not in ranks or position < ranks[path]:
                ranks[path] = position
                per_path_score.setdefault(key, {})[path] = candidate.score
            merged = extras.setdefault(key, {})
            for k, v in (candidate.extra or {}).items():
                merged.setdefault(k, v)

    fused = [
        Candidate(
            video_id=key[0],
            frame_id=key[1],
            score=score,
            source="fused",
            rank=0,
            extra={
                **extras.get(key, {}),
                "per_path_rank": per_path_rank.get(key, {}),
                "per_path_score": per_path_score.get(key, {}),
            },
        )
        for key, score in totals.items()
    ]
    fused.sort(key=lambda c: (-c.score, c.video_id, c.frame_id))
    return [c.replace(rank=i) for i, c in enumerate(fused, start=1)]


def rrf(paths: Paths, k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion: ``score(d) = Σ_p 1 / (k + rank_p(d))``."""
    return _accumulate(paths, None, lambda _p, rank, _c, _l: 1.0 / (k + rank))


def weighted_rrf(
    paths: Paths, weights: Mapping[str, float], k: int = 60
) -> list[Candidate]:
    """Weighted RRF: ``score(d) = Σ_p w_p / (k + rank_p(d))``."""
    return _accumulate(paths, weights, lambda _p, rank, _c, _l: 1.0 / (k + rank))


def weighted_norm(paths: Paths, weights: Mapping[str, float]) -> list[Candidate]:
    """Min-max normalise each path to ``[0, 1]``, then add with weights.

    Keeps the absolute score gaps that RRF throws away - useful for A/B testing
    against :func:`weighted_rrf`.
    """
    ranges: dict[str, tuple[float, float]] = {}
    for path, candidates in paths.items():
        if candidates:
            scores = [c.score for c in candidates]
            ranges[path] = (min(scores), max(scores))

    def contribution(path: str, _rank: int, candidate: Candidate, _l) -> float:
        low, high = ranges.get(path, (0.0, 0.0))
        if high <= low:
            return 1.0
        return (candidate.score - low) / (high - low)

    return _accumulate(paths, weights, contribution)


def fuse(
    paths: Paths,
    method: str = "weighted_rrf",
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> list[Candidate]:
    """Dispatch to the configured fusion strategy."""
    active = {p: list(c) for p, c in paths.items() if c}
    if not active:
        return []
    if method == "rrf" or weights is None:
        return rrf(active, k=k)
    if method == "weighted_norm":
        return weighted_norm(active, weights)
    return weighted_rrf(active, weights, k=k)


def normalize_weights(
    weights: Mapping[str, float],
    active_paths: Iterable[str],
    floor: float = 0.0,
    clamp_max: float | None = None,
) -> dict[str, float]:
    """Clamp, floor and re-normalise path weights so they sum to 1.

    Only ``active_paths`` (those with a non-null sub-query) get a floor; an
    inactive path stays at zero so a disabled modality cannot leak back in.

    ``clamp_max`` caps individual weights *before* normalising.  Pass ``1.0``
    for LLM-produced weights, where a hallucinated 5.0 should not swamp the
    others; leave it ``None`` for the static config weights, which are relative
    ratios (``visual: 1.5``) and would be destroyed by an upper clamp.
    """
    active = list(active_paths)
    if not active:
        return {}

    cleaned: dict[str, float] = {}
    for path in active:
        value = max(0.0, float(weights.get(path, 0.0) or 0.0))
        cleaned[path] = min(clamp_max, value) if clamp_max is not None else value

    if floor > 0:
        cleaned = {p: max(floor, w) for p, w in cleaned.items()}

    total = sum(cleaned.values())
    if total <= 0:
        share = 1.0 / len(active)
        return {p: share for p in active}
    return {p: w / total for p, w in cleaned.items()}
