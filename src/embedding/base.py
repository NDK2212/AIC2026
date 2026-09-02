"""Text-encoder abstraction plus a lazy, process-wide singleton registry.

Models are heavy, so they are only loaded when the visual path actually runs,
and only once per process.  Every encoder returns L2-normalised ``(B, dim)``
float32 arrays and caches individual vectors on disk.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..logging_utils import get_logger
from ..schemas import EncoderUnavailable
from ..utils.cache import DiskCache, sha256_key

if TYPE_CHECKING:  # pragma: no cover
    from ..config import EncoderConfig

log = get_logger(__name__)

_REGISTRY: dict[str, "TextEncoder"] = {}
_REGISTRY_LOCK = threading.Lock()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation that tolerates zero vectors."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


class TextEncoder(ABC):
    """Encodes text into the same space as the indexed image embeddings."""

    name: str = "encoder"

    def __init__(self, cfg: "EncoderConfig", cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self.name = cfg.name
        self._loaded = False
        self._lock = threading.Lock()

    # -- subclass contract -------------------------------------------------
    @property
    def dim(self) -> int:
        """Embedding dimension declared in the config (verified on load)."""
        return self.cfg.dim

    @abstractmethod
    def _load(self) -> None:
        """Load weights.  Raise :class:`EncoderUnavailable` when impossible."""

    @abstractmethod
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch, returning ``(len(texts), dim)`` unnormalised."""

    # -- public API --------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Load the model once; subsequent calls are no-ops."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            log.info(
                "Loading %s text encoder (%s) on %s",
                self.name,
                self.cfg.model_id,
                self.cfg.device,
                extra={"progress": {
                    "phase": "model",
                    "status": "running",
                    "title": f"Đang nạp {self.name} encoder",
                    "detail": f"{self.cfg.model_id} trên {self.cfg.device}",
                }},
            )
            self._load()
            self._loaded = True

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into an L2-normalised ``(B, dim)`` float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: list[np.ndarray | None] = [None] * len(texts)
        pending: list[int] = []
        for i, text in enumerate(texts):
            key = sha256_key(self.name, self.cfg.model_id, text)
            cached = self.cache.get_array("emb", key) if self.cache else None
            if cached is not None and cached.shape[-1] == self.dim:
                out[i] = cached.astype(np.float32).reshape(-1)
            else:
                pending.append(i)

        if pending:
            self.ensure_loaded()
            for start in range(0, len(pending), max(1, self.cfg.batch_size)):
                chunk = pending[start:start + max(1, self.cfg.batch_size)]
                vectors = l2_normalize(self._encode_batch([texts[i] for i in chunk]))
                if vectors.shape[1] != self.dim:
                    raise EncoderUnavailable(
                        f"{self.name} produced {vectors.shape[1]}-d vectors but "
                        f"embedding.{self.name}.dim says {self.dim}"
                    )
                for row, i in enumerate(chunk):
                    vector = vectors[row]
                    out[i] = vector
                    if self.cache is not None:
                        self.cache.set_array(
                            "emb", sha256_key(self.name, self.cfg.model_id, texts[i]), vector
                        )

        return np.stack([v for v in out if v is not None]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string into a ``(dim,)`` vector."""
        return self.encode([text])[0]


def get_encoder(cfg: "EncoderConfig", cache: DiskCache | None = None) -> TextEncoder:
    """Return the shared encoder instance for a config, building it on demand."""
    from .beit3 import BEiT3TextEncoder
    from .qwen import QwenTextEncoder
    from .siglip import SigLIPTextEncoder

    key = f"{cfg.name}:{cfg.model_id}:{cfg.backend}:{cfg.device}"
    with _REGISTRY_LOCK:
        encoder = _REGISTRY.get(key)
        if encoder is None:
            if cfg.name == "siglip":
                encoder = SigLIPTextEncoder(cfg, cache)
            elif cfg.name == "beit3":
                encoder = BEiT3TextEncoder(cfg, cache)
            elif cfg.name == "qwen":
                encoder = QwenTextEncoder(cfg, cache)
            else:  # pragma: no cover - guarded by config parsing
                raise EncoderUnavailable(f"Unknown encoder {cfg.name!r}")
            _REGISTRY[key] = encoder
        return encoder


def reset_encoders() -> None:
    """Drop every cached encoder (used by tests)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
