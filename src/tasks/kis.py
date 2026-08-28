"""Task 1 - Textual Known Item Search.

The fused ranking already answers this task; all that is left is ordering the
rows for the ``R@k`` metric (see :mod:`src.submission.builder`).
"""

from __future__ import annotations

from ..config import Config
from ..logging_utils import get_logger
from ..retrieval.pipeline import RetrievalPipeline
from ..schemas import Candidate
from ..submission.builder import build_rows
from ..utils.keyframe_index import KeyframeIndex

log = get_logger(__name__)


def run_kis(
    query: str,
    pipeline: RetrievalPipeline,
    kf: KeyframeIndex | None,
    cfg: Config,
    *,
    trace_name: str | None = None,
    use_cache: bool = True,
) -> list[tuple[str, int]]:
    """Return up to ``submission.max_rows`` ``(video_id, frame_id)`` rows."""
    rows = run_kis_candidates(
        query, pipeline, kf, cfg, trace_name=trace_name, use_cache=use_cache
    )
    return [(c.video_id, c.frame_id) for c in rows]


def run_kis_candidates(
    query: str,
    pipeline: RetrievalPipeline,
    kf: KeyframeIndex | None,
    cfg: Config,
    *,
    trace_name: str | None = None,
    use_cache: bool = True,
) -> list[Candidate]:
    """Same as :func:`run_kis` but keeps the full candidates (used by Q&A)."""
    candidates, _ = pipeline.run(
        query,
        topk=cfg.submission.max_rows * 5,
        trace_name=trace_name,
        use_cache=use_cache,
    )
    rows = build_rows(candidates, cfg.submission, kf)
    log.info("KIS produced %d rows for %r", len(rows), query[:60])
    return rows[: cfg.submission.max_rows]
