"""Qdrant access: connection, schema introspection and hybrid vector search.

The visual path issues **one** query with two prefetch branches (SigLIP and
BEiT-3) fused server-side by RRF or DBSF.  When the server is too old for the
Query API - or when ``qdrant.fusion: manual`` is configured - it degrades to two
separate searches fused client-side.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import QdrantConfig
from ..logging_utils import get_logger
from ..schemas import ConfigError

log = get_logger(__name__)

# Errors that mean "this server cannot do the Query API" rather than "bad input".
_UNSUPPORTED_MARKERS = (
    "not found", "404", "unknown route", "unsupported", "not supported",
    "no such method", "method not allowed", "405", "invalid path",
    "query_points", "unimplemented",
)


class QdrantWrapper:
    """Owns the Qdrant client and knows the collection's real schema."""

    def __init__(self, cfg: QdrantConfig) -> None:
        self.cfg = cfg
        self._client: Any | None = None
        self._models: Any | None = None
        self._schema: dict[str, int] | None = None
        self._query_api_ok: bool | None = None

    # ------------------------------------------------------------------
    @property
    def models(self) -> Any:
        """The ``qdrant_client.models`` module."""
        if self._models is None:
            try:
                from qdrant_client import models
            except ImportError as exc:  # pragma: no cover
                raise ConfigError(
                    "qdrant-client is not installed - run `pip install -r requirements.txt`"
                ) from exc
            self._models = models
        return self._models

    @property
    def client(self) -> Any:
        """Lazily connected ``QdrantClient``."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover
                raise ConfigError(
                    "qdrant-client is not installed - run `pip install -r requirements.txt`"
                ) from exc
            log.info("Connecting to Qdrant at %s", self.cfg.url)
            port = 443 if self.cfg.url.startswith("https") else None
            self._client = QdrantClient(
                url=self.cfg.url,
                port=port,
                api_key=self.cfg.api_key or None,
                timeout=self.cfg.timeout,
                prefer_grpc=False,
                check_compatibility=False,
            )
        return self._client

    # ------------------------------------------------------------------
    def vector_dims(self) -> dict[str, int]:
        """Map every named vector in the collection to its dimension.

        Raises :class:`ConfigError` when the collection does not exist.
        """
        if self._schema is not None:
            return self._schema

        try:
            info = self.client.get_collection(self.cfg.collection)
        except Exception as exc:  # noqa: BLE001 - client errors are untyped
            raise ConfigError(
                f"Cannot read Qdrant collection {self.cfg.collection!r} at "
                f"{self.cfg.url}: {exc}"
            ) from exc

        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        dims: dict[str, int] = {}
        if isinstance(vectors, dict):
            for name, spec in vectors.items():
                size = getattr(spec, "size", None)
                if size is None and isinstance(spec, dict):
                    size = spec.get("size")
                if size is not None:
                    dims[str(name)] = int(size)
        elif vectors is not None:  # single unnamed vector
            size = getattr(vectors, "size", None)
            if size is not None:
                dims[""] = int(size)

        self._schema = dims
        return dims

    def assert_vector(self, name: str, dim: int) -> None:
        """Fail loudly when an encoder's dimension disagrees with the index."""
        dims = self.vector_dims()
        if name not in dims:
            raise ConfigError(
                f"Qdrant collection {self.cfg.collection!r} has no named vector "
                f"{name!r}. Available: {sorted(dims) or '<none>'}. "
                "Fix qdrant.vector_names in config.yaml."
            )
        if dims[name] != dim:
            raise ConfigError(
                f"Dimension mismatch for vector {name!r}: the collection stores "
                f"{dims[name]}-d vectors but the configured encoder produces {dim}-d. "
                "The query encoder must be the same model used at indexing time."
            )

    def count(self) -> int | None:
        """Number of points in the collection, or ``None`` if unavailable."""
        try:
            return int(self.client.count(self.cfg.collection, exact=False).count)
        except Exception as exc:  # noqa: BLE001 # pragma: no cover
            log.debug("Qdrant count failed: %s", exc)
            return None

    def sample_payload(self, limit: int = 1) -> list[dict[str, Any]]:
        """A couple of payloads, used by ``cli inspect`` to show real fields."""
        try:
            points, _ = self.client.scroll(
                collection_name=self.cfg.collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [dict(p.payload or {}) for p in points]
        except Exception as exc:  # noqa: BLE001 # pragma: no cover
            log.debug("Qdrant scroll failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        vectors: dict[str, np.ndarray],
        limit: int,
        prefetch_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search several named vectors at once, fused inside Qdrant.

        ``vectors`` maps a *named vector* to its query embedding.  Returns a
        list of ``{"id", "score", "payload"}`` dicts, best first.
        """
        if not vectors:
            return []
        prefetch_limit = prefetch_limit or self.cfg.prefetch_limit

        if len(vectors) == 1:
            # A single branch needs no fusion at all - returning it untagged
            # preserves the real similarity scores instead of replacing them
            # with reciprocal-rank values.
            name, vector = next(iter(vectors.items()))
            return self._single_search(name, vector, limit)

        if self.cfg.fusion == "manual":
            log.info("qdrant.fusion=manual - running one search per vector")
            return self._multi_single_search(vectors, limit, prefetch_limit)

        if self._query_api_ok is not False:
            try:
                return self._fused_query(vectors, limit, prefetch_limit)
            except Exception as exc:  # noqa: BLE001
                if _looks_unsupported(exc):
                    self._query_api_ok = False
                    log.warning(
                        "Qdrant Query API unavailable (%s) - falling back to "
                        "per-vector search with client-side RRF", exc,
                    )
                else:
                    raise
        return self._multi_single_search(vectors, limit, prefetch_limit)

    def _fused_query(
        self, vectors: dict[str, np.ndarray], limit: int, prefetch_limit: int
    ) -> list[dict[str, Any]]:
        """One ``query_points`` call with N prefetch branches + server fusion."""
        models = self.models
        prefetch = [
            models.Prefetch(
                query=_as_list(vec),
                using=name,
                limit=prefetch_limit,
            )
            for name, vec in vectors.items()
        ]
        fusion = models.Fusion.DBSF if self.cfg.fusion == "dbsf" else models.Fusion.RRF
        response = self.client.query_points(
            collection_name=self.cfg.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=fusion),
            limit=limit,
            with_payload=True,
        )
        self._query_api_ok = True
        points = getattr(response, "points", response)
        log.debug(
            "Qdrant fused query (%s) over %s -> %d points",
            self.cfg.fusion, list(vectors), len(points),
        )
        return [_point_to_dict(p) for p in points]

    def _multi_single_search(
        self, vectors: dict[str, np.ndarray], limit: int, prefetch_limit: int
    ) -> list[dict[str, Any]]:
        """One search per named vector, returned separately for client fusion.

        The results are tagged with ``_vector`` so :mod:`src.retrieval.fusion`
        can RRF them; a single-vector call just returns its own ranking.
        """
        out: list[dict[str, Any]] = []
        for name, vec in vectors.items():
            hits = self._single_search(name, vec, max(limit, prefetch_limit))
            for rank, hit in enumerate(hits, start=1):
                hit["_vector"] = name
                hit["_rank"] = rank
            out.extend(hits)
        return out

    def _single_search(
        self, name: str, vector: np.ndarray, limit: int
    ) -> list[dict[str, Any]]:
        """Search one named vector, preferring the Query API when it works."""
        query = _as_list(vector)
        if self._query_api_ok is not False:
            try:
                response = self.client.query_points(
                    collection_name=self.cfg.collection,
                    query=query,
                    using=name,
                    limit=limit,
                    with_payload=True,
                )
                self._query_api_ok = True
                return [_point_to_dict(p) for p in getattr(response, "points", response)]
            except Exception as exc:  # noqa: BLE001
                if not _looks_unsupported(exc):
                    raise
                self._query_api_ok = False
                log.warning("Qdrant Query API unavailable (%s) - using legacy search", exc)

        points = self.client.search(
            collection_name=self.cfg.collection,
            query_vector=(name, query) if name else query,
            limit=limit,
            with_payload=True,
        )
        return [_point_to_dict(p) for p in points]


def _looks_unsupported(exc: Exception) -> bool:
    """True when the error means the server lacks the Query API."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _UNSUPPORTED_MARKERS)


def _as_list(vector: np.ndarray | Sequence[float]) -> list[float]:
    """Qdrant wants plain python floats, not numpy scalars."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return [float(x) for x in array]


def _point_to_dict(point: Any) -> dict[str, Any]:
    """Normalise a scored point into a plain dict."""
    return {
        "id": getattr(point, "id", None),
        "score": float(getattr(point, "score", 0.0) or 0.0),
        "payload": dict(getattr(point, "payload", None) or {}),
    }
