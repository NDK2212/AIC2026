"""Qdrant retrieval for the visual path.

The image sub-query is embedded by the SigLIP *and* BEiT-3 text towers, then
sent to Qdrant as a single query with two prefetch branches fused server-side
(RRF or DBSF).  If the server cannot do that - or the config asks for manual
fusion - the two rankings come back separately and are fused client-side with
the very same :func:`src.retrieval.fusion.rrf`.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..clients.qdrant import QdrantWrapper
from ..config import Config
from ..logging_utils import get_logger
from ..schemas import Candidate, EncoderUnavailable, PATH_VISUAL
from ..utils.cache import DiskCache
from .fusion import rrf

log = get_logger(__name__)


class VisualSearcher:
    """Encodes a text query and searches the keyframe vector index."""

    def __init__(
        self,
        cfg: Config,
        qdrant: QdrantWrapper,
        cache: DiskCache | None = None,
    ) -> None:
        self.cfg = cfg
        self.qdrant = qdrant
        self.cache = cache
        self._encoders: dict[str, Any] | None = None
        self._degraded: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def encoders(self) -> dict[str, Any]:
        """Return the usable text encoders, keyed by Qdrant vector name.

        An encoder that cannot load is dropped with a warning rather than
        taking the whole visual path down - but if *none* load, the caller sees
        an empty dict and the path returns no candidates.
        """
        with self._lock:
            if self._encoders is not None:
                return self._encoders

            from ..embedding import get_encoder

            found: dict[str, Any] = {}
            for name in ("siglip", "beit3", "qwen"):
                enc_cfg = getattr(self.cfg.embedding, name, None)
                if enc_cfg is None or not enc_cfg.enabled:
                    log.info("embedding.%s is disabled or missing - skipping it", name)
                    continue
                if name not in self.cfg.qdrant.vector_names:
                    log.warning("Vector %s not configured in qdrant.vector_names - skipping", name)
                    continue
                vector_name = self.cfg.qdrant.vector_names[name]
                try:
                    encoder = get_encoder(enc_cfg, self.cache)
                    encoder.ensure_loaded()
                except EncoderUnavailable as exc:
                    self._degraded.add(name)
                    log.warning(
                        "%s text encoder unavailable - the visual path will run "
                        "without it. Reason: %s", name, exc,
                    )
                    continue
                found[vector_name] = encoder

            if not found:
                log.error(
                    "No visual text encoder could be loaded - the visual path is dead. "
                    "Check embedding.* in config.yaml."
                )
            self._encoders = found
            return found

    def verify_dimensions(self) -> dict[str, tuple[int, int]]:
        """Assert each encoder's width matches its Qdrant vector.

        Returns ``{vector_name: (encoder_dim, collection_dim)}``; raises
        :class:`~src.schemas.ConfigError` on any mismatch.
        """
        report: dict[str, tuple[int, int]] = {}
        collection_dims = self.qdrant.vector_dims()
        for vector_name, encoder in self.encoders().items():
            self.qdrant.assert_vector(vector_name, encoder.dim)
            report[vector_name] = (encoder.dim, collection_dims[vector_name])
        return report

    # ------------------------------------------------------------------
    def search(self, image_query: str | None, limit: int | None = None) -> list[Candidate]:
        """Retrieve keyframes for a visual sub-query."""
        if not image_query or not image_query.strip():
            log.debug("visual path disabled (no image sub-query)")
            return []

        limit = limit or self.cfg.qdrant.search_limit
        encoders = self.encoders()
        if not encoders:
            return []

        try:
            vectors = {
                name: encoder.encode_one(image_query)
                for name, encoder in encoders.items()
            }
        except Exception as exc:  # noqa: BLE001
            log.error("Query embedding failed: %s", exc)
            return []

        try:
            raw = self.qdrant.hybrid_search(
                vectors, limit=limit, prefetch_limit=self.cfg.qdrant.prefetch_limit
            )
        except Exception as exc:  # noqa: BLE001 - a dead path must not kill the run
            log.error("Qdrant search failed: %s", exc)
            return []

        candidates = self._to_candidates(raw, limit)
        log.info(
            "visual path -> %d candidates (vectors: %s)",
            len(candidates), ", ".join(sorted(vectors)),
        )
        return candidates

    # ------------------------------------------------------------------
    def _to_candidates(self, raw: list[dict[str, Any]], limit: int) -> list[Candidate]:
        """Parse Qdrant points, fusing client-side when results are per-vector."""
        video_field = self.cfg.qdrant.payload["video_id"]
        frame_field = self.cfg.qdrant.payload["frame_id"]

        by_vector: dict[str, list[Candidate]] = {}
        flat: list[Candidate] = []
        skipped = 0

        for position, point in enumerate(raw, start=1):
            payload = point.get("payload") or {}
            video_id = payload.get(video_field)
            if video_id is None:
                video_id = payload.get("video_name") or payload.get("video_id") or payload.get("video")

            frame_raw = payload.get(frame_field)
            if frame_raw is None:
                frame_raw = payload.get("frame_id") or payload.get("frame_idx") or payload.get("frame_id_ocr") or payload.get("frame")

            if video_id is None or frame_raw is None:
                skipped += 1
                continue
            try:
                frame_id = int(float(frame_raw))
            except (TypeError, ValueError):
                skipped += 1
                continue

            vector = point.get("_vector")
            rank = int(point.get("_rank") or position)
            candidate = Candidate(
                video_id=str(video_id).removesuffix(".mp4"),
                frame_id=frame_id,
                score=float(point.get("score") or 0.0),
                source=PATH_VISUAL,
                rank=rank,
                extra={
                    "point_id": point.get("id"),
                    **({"vector": vector} if vector else {}),
                },
            )
            if vector:
                by_vector.setdefault(str(vector), []).append(candidate)
            else:
                flat.append(candidate)

        if skipped:
            log.warning(
                "visual: skipped %d Qdrant points missing %s/%s",
                skipped, video_field, frame_field,
            )

        if by_vector:
            log.debug("Client-side RRF over %s", sorted(by_vector))
            fused = rrf(by_vector, k=self.cfg.fusion.rrf_k)
            merged = [
                c.replace(source=PATH_VISUAL, rank=i)
                for i, c in enumerate(fused[:limit], start=1)
            ]
            if not flat:
                return merged
            flat.extend(merged)

        # Deduplicate while keeping the best-ranked occurrence of each key.
        seen: dict[tuple[str, int], Candidate] = {}
        for candidate in flat:
            existing = seen.get(candidate.key)
            if existing is None or candidate.rank < existing.rank:
                seen[candidate.key] = candidate
        ordered = sorted(seen.values(), key=lambda c: (c.rank, c.video_id, c.frame_id))
        return [c.replace(rank=i) for i, c in enumerate(ordered[:limit], start=1)]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity helper used by the optional reranker."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0
