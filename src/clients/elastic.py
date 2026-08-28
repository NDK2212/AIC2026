"""Elasticsearch access for the OCR and ASR retrieval paths."""

from __future__ import annotations

from typing import Any

from ..config import ElasticConfig
from ..logging_utils import get_logger
from ..schemas import ConfigError

log = get_logger(__name__)


class ElasticWrapper:
    """Owns the ES client and exposes the one search shape the pipeline needs."""

    def __init__(self, cfg: ElasticConfig) -> None:
        self.cfg = cfg
        self._client: Any | None = None

    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        """Lazily connected ``Elasticsearch`` client."""
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as exc:  # pragma: no cover
                raise ConfigError(
                    "elasticsearch is not installed - run `pip install -r requirements.txt`"
                ) from exc

            kwargs: dict[str, Any] = {
                "hosts": self.cfg.hosts,
                "request_timeout": self.cfg.timeout,
                "verify_certs": self.cfg.verify_certs,
            }
            if self.cfg.api_key:
                kwargs["api_key"] = self.cfg.api_key
            elif self.cfg.user and self.cfg.password:
                kwargs["basic_auth"] = (self.cfg.user, self.cfg.password)
            log.info("Connecting to Elasticsearch at %s", self.cfg.hosts)
            self._client = Elasticsearch(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """True when the cluster answers."""
        try:
            return bool(self.client.ping())
        except Exception as exc:  # noqa: BLE001
            log.debug("Elasticsearch ping failed: %s", exc)
            return False

    def index_fields(self) -> dict[str, str]:
        """Field name -> ES type for the configured index."""
        try:
            mapping = self.client.indices.get_mapping(index=self.cfg.index)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(
                f"Cannot read Elasticsearch index {self.cfg.index!r}: {exc}"
            ) from exc

        out: dict[str, str] = {}
        for body in dict(mapping).values():
            props = (body or {}).get("mappings", {}).get("properties", {}) or {}
            for name, spec in props.items():
                out[str(name)] = str((spec or {}).get("type", "object"))
        return out

    def count(self) -> int | None:
        """Number of documents in the index, or ``None`` if unavailable."""
        try:
            return int(self.client.count(index=self.cfg.index)["count"])
        except Exception as exc:  # noqa: BLE001
            log.debug("Elasticsearch count failed: %s", exc)
            return None

    def sample_docs(self, size: int = 1) -> list[dict[str, Any]]:
        """A couple of documents, used by ``cli inspect``."""
        try:
            res = self.client.search(index=self.cfg.index, size=size, query={"match_all": {}})
            return [hit.get("_source", {}) for hit in res["hits"]["hits"]]
        except Exception as exc:  # noqa: BLE001
            log.debug("Elasticsearch sample failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    def search(self, body: dict[str, Any], size: int) -> list[dict[str, Any]]:
        """Run a search body and return the raw hits, best first.

        ``body`` is written in REST form (``_source``); elasticsearch-py 8.x
        takes those as keyword arguments and spells that one ``source``, so it
        is translated here rather than leaking the client's naming into the
        query builder.
        """
        params = dict(body)
        if "_source" in params:
            params["source"] = params.pop("_source")
        if "size" in params:
            size = params.pop("size")
        response = self.client.search(index=self.cfg.index, size=size, **params)
        hits = response["hits"]["hits"]
        log.debug("Elasticsearch returned %d hits", len(hits))
        return list(hits)
