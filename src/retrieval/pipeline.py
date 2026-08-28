"""The online retrieval pipeline: query -> 3 parallel paths -> weighted fusion.

Design rules enforced here:

* the three paths run concurrently (they are all I/O bound);
* the visual path optionally runs BLIP-2 ITM reranking immediately after Qdrant;
* fused candidates optionally run BGE Cross-Encoder reranking after WRRF;
* a path that raises logs the error and contributes ``[]`` instead of taking the
  whole query down;
* every run dumps a JSON trace under ``outputs/runs/`` - that trace is the main
  debugging tool for fusion weights and per-path quality.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..clients.llm import LLMClient
from ..config import Config
from ..logging_utils import get_logger
from ..schemas import (
    ALL_PATHS,
    Candidate,
    DecomposeResult,
    PATH_ASR,
    PATH_DESCRIPTION,
    PATH_OCR,
    PATH_VISUAL,
)
from .decompose import decompose
from .fusion import fuse, normalize_weights
from .rerank import BGEReranker, BLIP2Reranker
from .search_text import TextSearcher
from .search_visual import VisualSearcher

log = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


class RetrievalPipeline:
    """Orchestrates query decomposition, 4-path retrieval, and weighted fusion."""

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        text_searcher: TextSearcher,
        visual_searcher: VisualSearcher,
        reranker: BLIP2Reranker | None = None,
        blip2_reranker: BLIP2Reranker | None = None,
        bge_reranker: BGEReranker | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.text = text_searcher
        self.visual = visual_searcher
        self.blip2 = blip2_reranker or reranker
        self.bge = bge_reranker

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        topk: int = 300,
        decompose_result: DecomposeResult | None = None,
        *,
        trace_name: str | None = None,
        write_trace: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[Candidate], DecomposeResult]:
        """Decompose, search all 4 paths in parallel with symmetric reranking, fuse, and return ranking."""
        started = time.time()
        result = decompose_result or decompose(
            self.llm,
            query,
            adaptive_floor=self.cfg.fusion.adaptive_floor,
            default_weights=self.cfg.fusion.weights,
            use_cache=use_cache,
        )

        paths = self._run_paths(result)
        weights = self.resolve_weights(result)
        fused = fuse(
            paths,
            method=self.cfg.fusion.method,
            weights=weights,
            k=self.cfg.fusion.rrf_k,
        )

        ranked = fused[:topk]
        elapsed = time.time() - started
        log.info(
            "Retrieved %d candidates in %.2fs (paths: %s | weights: %s)",
            len(ranked), elapsed,
            {p: len(c) for p, c in paths.items()},
            {p: round(w, 3) for p, w in weights.items()},
        )

        if write_trace:
            self._write_trace(query, result, paths, weights, ranked, elapsed, trace_name)
        return ranked, result

    # ------------------------------------------------------------------
    def _run_paths(self, result: DecomposeResult) -> dict[str, list[Candidate]]:
        """Fan the 4 retrieval paths out over a thread pool with symmetric reranking."""
        jobs: dict[str, Callable[[], list[Candidate]]] = {}

        if result.ocr_query:
            def _run_ocr() -> list[Candidate]:
                cands = self.text.search_ocr(
                    result.ocr_query,
                    result.ocr_terms,
                    size=self.cfg.elasticsearch.size,
                    exact_text=result.exact_text,
                )
                if self.bge is not None and self.bge.enabled and result.ocr_query:
                    cands = self.bge.rerank(result.ocr_query, cands)
                return cands

            jobs[PATH_OCR] = _run_ocr

        if result.asr_query:
            def _run_asr() -> list[Candidate]:
                cands = self.text.search_asr(
                    result.asr_query,
                    result.asr_terms,
                    size=self.cfg.elasticsearch.size,
                )
                if self.bge is not None and self.bge.enabled and result.asr_query:
                    cands = self.bge.rerank(result.asr_query, cands)
                return cands

            jobs[PATH_ASR] = _run_asr

        if result.description_query:
            def _run_desc() -> list[Candidate]:
                desc_q = result.description_query
                cands = self.text.search_description(
                    desc_q,
                    result.description_terms,
                    size=self.cfg.elasticsearch.size,
                )
                if self.bge is not None and self.bge.enabled and desc_q:
                    cands = self.bge.rerank(desc_q, cands)
                return cands

            jobs[PATH_DESCRIPTION] = _run_desc

        if result.image_query:
            def _run_visual() -> list[Candidate]:
                cands = self.visual.search(
                    result.image_query, limit=self.cfg.qdrant.search_limit
                )
                if self.blip2 is not None and self.blip2.enabled and result.image_query:
                    cands = self.blip2.rerank(result.image_query, cands)
                return cands

            jobs[PATH_VISUAL] = _run_visual

        if not jobs:
            log.warning("Every retrieval path is disabled for this query")
            return {}

        out: dict[str, list[Candidate]] = {}
        with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="path") as pool:
            futures = {pool.submit(job): path for path, job in jobs.items()}
            for future, path in futures.items():
                try:
                    out[path] = future.result()
                except Exception as exc:  # noqa: BLE001 - one path must not kill the run
                    log.error("%s path failed: %s", path, exc, exc_info=True)
                    out[path] = []
        return out

    def resolve_weights(self, result: DecomposeResult) -> dict[str, float]:
        """Pick the fusion weights: adaptive (LLM) or the static config ones."""
        active = [path for path in ALL_PATHS if result.query_for(path)]
        if not active:
            return {}
        if self.cfg.fusion.adaptive and result.modality_weights:
            return normalize_weights(
                result.path_weights(), active,
                floor=self.cfg.fusion.adaptive_floor, clamp_max=1.0,
            )
        # Static config weights are ratios (e.g. visual: 1.5) - never clamped.
        raw = {path: float(self.cfg.fusion.weights.get(path, 1.0)) for path in active}
        return normalize_weights(raw, active, floor=self.cfg.fusion.adaptive_floor)

    # ------------------------------------------------------------------
    def _write_trace(
        self,
        query: str,
        result: DecomposeResult,
        paths: dict[str, list[Candidate]],
        weights: dict[str, float],
        fused: list[Candidate],
        elapsed: float,
        trace_name: str | None,
    ) -> Path | None:
        """Dump a per-run JSON trace for debugging fusion behaviour."""
        try:
            self.cfg.runs.dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            slug = _SLUG_RE.sub("-", trace_name or query[:40]).strip("-").lower() or "query"
            path = self.cfg.runs.dir / f"{stamp}_{slug}.json"
            payload = {
                "timestamp": stamp,
                "query": query,
                "elapsed_sec": round(elapsed, 3),
                "decompose": result.to_dict(),
                "fusion": {
                    "method": self.cfg.fusion.method,
                    "rrf_k": self.cfg.fusion.rrf_k,
                    "adaptive": self.cfg.fusion.adaptive,
                    "weights_used": weights,
                },
                "path_sizes": {p: len(c) for p, c in paths.items()},
                "path_top20": {
                    p: [_candidate_dict(c) for c in c_list[:20]]
                    for p, c_list in paths.items()
                },
                "fused_top50": [_candidate_dict(c) for c in fused[:50]],
            }
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            log.debug("Run trace written to %s", path)
            return path
        except OSError as exc:  # pragma: no cover
            log.warning("Could not write run trace: %s", exc)
            return None


def _candidate_dict(candidate: Candidate) -> dict[str, Any]:
    """Compact JSON view of a candidate for the run trace."""
    return {
        "video_id": candidate.video_id,
        "frame_id": candidate.frame_id,
        "score": round(candidate.score, 6),
        "source": candidate.source,
        "rank": candidate.rank,
        "extra": {
            k: v
            for k, v in (candidate.extra or {}).items()
            if k in {
                "per_path_rank",
                "per_path_score",
                "matched_text",
                "blip2_itm_score",
                "bge_score",
                "itm_score",
            }
        },
    }
