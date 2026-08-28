"""Content-addressed disk cache for LLM answers and text embeddings.

Keys are ``sha256`` digests of the exact inputs, so a re-run costs no API quota
and produces byte-identical results.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)


def sha256_key(*parts: Any) -> str:
    """Stable digest over an arbitrary tuple of scalars."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


class DiskCache:
    """A tiny two-level (``namespace``/``sha``) cache on the filesystem."""

    def __init__(self, root: Path | str, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def _path(self, namespace: str, key: str, suffix: str) -> Path:
        bucket = self.root / namespace / key[:2]
        return bucket / f"{key}{suffix}"

    # -- JSON --------------------------------------------------------------
    def get_json(self, namespace: str, key: str) -> Any | None:
        """Return a cached JSON value, or ``None`` on a miss."""
        if not self.enabled:
            return None
        path = self._path(namespace, key, ".json")
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Dropping corrupt cache entry %s: %s", path, exc)
            return None

    def set_json(self, namespace: str, key: str, value: Any) -> None:
        """Persist a JSON-serialisable value."""
        if not self.enabled:
            return
        path = self._path(namespace, key, ".json")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
            tmp.replace(path)

    # -- numpy -------------------------------------------------------------
    def get_array(self, namespace: str, key: str) -> np.ndarray | None:
        """Return a cached vector, or ``None`` on a miss."""
        if not self.enabled:
            return None
        path = self._path(namespace, key, ".npy")
        if not path.is_file():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError) as exc:
            log.warning("Dropping corrupt cache entry %s: %s", path, exc)
            return None

    def set_array(self, namespace: str, key: str, value: np.ndarray) -> None:
        """Persist a numpy vector."""
        if not self.enabled:
            return
        path = self._path(namespace, key, ".npy")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                np.save(fh, value)
            tmp.replace(path)

    # -- misc --------------------------------------------------------------
    def clear(self, namespace: str | None = None) -> int:
        """Delete cached files; returns how many were removed."""
        target = self.root / namespace if namespace else self.root
        if not target.exists():
            return 0
        removed = 0
        for path in target.rglob("*"):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed
