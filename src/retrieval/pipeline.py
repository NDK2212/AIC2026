"""The online retrieval pipeline: query -> 4 parallel paths -> weighted fusion.

Design rules enforced here:

* the four paths run concurrently (they are all I/O bound);
* legacy BLIP/BGE path rerankers remain supported but are disabled by default;
* Qwen3-VL reranks fused candidates once using keyframe images and metadata;
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
from typing import Any, Callable, Mapping
from collections.abc import Sequence

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
from .rerank import BGEReranker, BLIP2Reranker, Qwen3VLReranker
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
        qwen3_vl_reranker: Qwen3VLReranker | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.text = text_searcher
        self.visual = visual_searcher
        self.blip2 = blip2_reranker or reranker
        self.bge = bge_reranker
        self.qwen3_vl = qwen3_vl_reranker

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
        video_ids: Sequence[str] | None = None,
        enabled_paths: Sequence[str] | None = None,
        fusion_method: str | None = None,
        fusion_weights: Mapping[str, float] | None = None,
        fusion_adaptive: bool | None = None,
        rrf_k: int | None = None,
        qwen_enabled: bool | None = None,
        qwen_top_n: int | None = None,
        qwen_weight: float | None = None,
    ) -> tuple[list[Candidate], DecomposeResult]:
        """Run retrieval with optional request-scoped UI overrides."""
        started = time.time()
        result = decompose_result or decompose(
            self.llm,
            query,
            adaptive_floor=self.cfg.fusion.adaptive_floor,
            default_weights=self.cfg.fusion.weights,
            use_cache=use_cache,
        )

        allowed_paths = tuple(enabled_paths) if enabled_paths is not None else None
        paths = self._run_paths(
            result, video_ids=video_ids, enabled_paths=allowed_paths
        )
        weights = self.resolve_weights(
            result,
            enabled_paths=allowed_paths,
            override_weights=fusion_weights,
            adaptive=fusion_adaptive,
        )
        effective_method = fusion_method or self.cfg.fusion.method
        if effective_method not in {"rrf", "weighted_rrf", "weighted_norm"}:
            effective_method = self.cfg.fusion.method
        effective_rrf_k = max(1, int(rrf_k or self.cfg.fusion.rrf_k))
        fused = fuse(
            paths,
            method=effective_method,
            weights=weights,
            k=effective_rrf_k,
        )
        should_qwen = qwen_enabled if qwen_enabled is not None else True
        if self.qwen3_vl is not None and self.qwen3_vl.enabled and should_qwen:
            reranker_cfg = getattr(self.qwen3_vl, "cfg", None)
            effective_top_n = qwen_top_n or getattr(reranker_cfg, "top_n", 25)
            log.info(
                "Qwen3-VL reranking top %d fused candidates",
                effective_top_n,
                extra={"progress": {
                    "phase": "rerank",
                    "status": "running",
                    "title": "Qwen3-VL đang rerank",
                    "detail": f"Top {effective_top_n} fused candidates trên GPU",
                }},
            )
            fused = self._enrich_qwen_candidates(fused, top_n=qwen_top_n)
            if qwen_top_n is None and qwen_weight is None:
                fused = self.qwen3_vl.rerank(query, fused)
            else:
                fused = self.qwen3_vl.rerank(
                    query, fused, top_n=qwen_top_n, weight=qwen_weight
                )

        ranked = fused[:topk]
        elapsed = time.time() - started
        log.info(
            "Retrieved %d candidates in %.2fs (paths: %s | weights: %s)",
            len(ranked), elapsed,
            {p: len(c) for p, c in paths.items()},
            {p: round(w, 3) for p, w in weights.items()},
            extra={"progress": {
                "phase": "retrieval",
                "status": "done",
                "title": "Retrieval và fusion hoàn tất",
                "detail": f"{len(ranked)} candidates trong {elapsed:.2f}s",
            }},
        )

        if write_trace:
            self._write_trace(
                query, result, paths, weights, ranked, elapsed, trace_name,
                fusion_method=effective_method, rrf_k=effective_rrf_k,
                fusion_adaptive=(
                    self.cfg.fusion.adaptive
                    if fusion_adaptive is None else fusion_adaptive
                ),
            )
        return ranked, result

    def _enrich_qwen_candidates(
        self, candidates: list[Candidate], *, top_n: int | None = None
    ) -> list[Candidate]:
        """Attach Description/OCR/ASR to the fused head before VL reranking."""
        if not candidates or not hasattr(self.text, "fetch_metadata"):
            return candidates
        reranker_cfg = getattr(self.qwen3_vl, "cfg", None)
        top_n = max(1, int(top_n or getattr(reranker_cfg, "top_n", 25)))
        head, tail = candidates[:top_n], candidates[top_n:]
        try:
            metadata = self.text.fetch_metadata(
                head, max_shot_gap=self.cfg.submission.shot_window
            )
        except Exception as exc:  # noqa: BLE001 - image-only reranking remains valid
            log.warning("Qwen3-VL metadata enrichment failed: %s", exc)
            return candidates
        enriched: list[Candidate] = []
        for candidate in head:
            values = metadata.get(candidate.key, {})
            extra = dict(candidate.extra or {})
            for source_key, extra_key in (
                ("description", "description_matched"),
                ("ocr", "ocr_matched"),
                ("asr", "asr_matched"),
            ):
                if values.get(source_key) and not extra.get(extra_key):
                    extra[extra_key] = values[source_key]
            enriched.append(candidate.replace(extra=extra))
        return enriched + tail

    # ------------------------------------------------------------------
    def _run_paths(
        self,
        result: DecomposeResult,
        *,
        video_ids: Sequence[str] | None = None,
        enabled_paths: Sequence[str] | None = None,
    ) -> dict[str, list[Candidate]]:
        """Fan the 4 retrieval paths out over a thread pool with symmetric reranking."""
        jobs: dict[str, Callable[[], list[Candidate]]] = {}
        scope = list(video_ids) if video_ids else None
        allowed = set(enabled_paths) if enabled_paths is not None else set(ALL_PATHS)

        if PATH_OCR in allowed and result.ocr_query:
            def _run_ocr() -> list[Candidate]:
                kwargs: dict[str, Any] = {
                    "size": self.cfg.elasticsearch.size,
                    "exact_text": result.exact_text,
                }
                if scope:
                    kwargs["video_ids"] = scope
                cands = self.text.search_ocr(result.ocr_query, result.ocr_terms, **kwargs)
                if self.bge is not None and self.bge.enabled and result.ocr_query:
                    cands = self.bge.rerank(result.ocr_query, cands)
                return cands

            jobs[PATH_OCR] = _run_ocr

        if PATH_ASR in allowed and result.asr_query:
            def _run_asr() -> list[Candidate]:
                kwargs: dict[str, Any] = {"size": self.cfg.elasticsearch.size}
                if scope:
                    kwargs["video_ids"] = scope
                cands = self.text.search_asr(result.asr_query, result.asr_terms, **kwargs)
                if self.bge is not None and self.bge.enabled and result.asr_query:
                    cands = self.bge.rerank(result.asr_query, cands)
                return cands

            jobs[PATH_ASR] = _run_asr

        if PATH_DESCRIPTION in allowed and result.description_query:
            def _run_desc() -> list[Candidate]:
                desc_q = result.description_query
                kwargs: dict[str, Any] = {"size": self.cfg.elasticsearch.size}
                if scope:
                    kwargs["video_ids"] = scope
                cands = self.text.search_description(
                    desc_q, result.description_terms, **kwargs
                )
                if self.bge is not None and self.bge.enabled and desc_q:
                    cands = self.bge.rerank(desc_q, cands)
                return cands

            jobs[PATH_DESCRIPTION] = _run_desc

        if PATH_VISUAL in allowed and result.image_query:
            def _run_visual() -> list[Candidate]:
                kwargs: dict[str, Any] = {"limit": self.cfg.qdrant.search_limit}
                if scope:
                    kwargs["video_ids"] = scope
                cands = self.visual.search(result.image_query, **kwargs)
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

    def resolve_weights(
        self,
        result: DecomposeResult,
        *,
        enabled_paths: Sequence[str] | None = None,
        override_weights: Mapping[str, float] | None = None,
        adaptive: bool | None = None,
    ) -> dict[str, float]:
        """Pick the fusion weights: adaptive (LLM) or the static config ones."""
        allowed = set(enabled_paths) if enabled_paths is not None else set(ALL_PATHS)
        active = [
            path for path in ALL_PATHS
            if path in allowed and result.query_for(path)
        ]
        if not active:
            return {}
        use_adaptive = self.cfg.fusion.adaptive if adaptive is None else adaptive
        if use_adaptive and result.modality_weights:
            return normalize_weights(
                result.path_weights(), active,
                floor=self.cfg.fusion.adaptive_floor, clamp_max=1.0,
            )
        # Static config weights are ratios (e.g. visual: 1.5) - never clamped.
        source_weights = override_weights or self.cfg.fusion.weights
        raw = {path: float(source_weights.get(path, 1.0)) for path in active}
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
        *,
        fusion_method: str | None = None,
        rrf_k: int | None = None,
        fusion_adaptive: bool | None = None,
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
                    "method": fusion_method or self.cfg.fusion.method,
                    "rrf_k": rrf_k or self.cfg.fusion.rrf_k,
                    "adaptive": (
                        self.cfg.fusion.adaptive
                        if fusion_adaptive is None else fusion_adaptive
                    ),
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
                "qwen3_vl_rerank_score",
                "pre_qwen3_vl_score",
            }
        },
    }
