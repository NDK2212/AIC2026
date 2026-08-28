"""SigLIP text tower.

Two backends are supported so the query encoder can match whatever was used at
indexing time: HuggingFace ``transformers`` (default) and ``open_clip``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..logging_utils import get_logger
from ..schemas import EncoderUnavailable
from .base import TextEncoder

log = get_logger(__name__)


class SigLIPTextEncoder(TextEncoder):
    """Encodes queries with the SigLIP text tower."""

    def __init__(self, cfg: Any, cache: Any = None) -> None:
        super().__init__(cfg, cache)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._backend = cfg.backend

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise EncoderUnavailable(
                "torch is not installed - the visual retrieval path cannot run"
            ) from exc
        self._torch = torch

        if self._backend == "open_clip":
            self._load_open_clip()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise EncoderUnavailable(
                "transformers is not installed - run `pip install -r requirements.txt`"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
            dtype = self._torch.float16 if (self.cfg.device != "cpu" and self._torch.cuda.is_available()) else self._torch.float32
            model = AutoModel.from_pretrained(self.cfg.model_id, torch_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailable(
                f"Could not load SigLIP model {self.cfg.model_id!r}: {exc}"
            ) from exc

        model.eval().to(self.cfg.device)
        self._model = model

        hidden = _declared_text_dim(model)
        if hidden and hidden != self.cfg.dim:
            log.warning(
                "SigLIP text projection is %d-d but embedding.siglip.dim says %d - "
                "the configured value wins only if the model actually agrees",
                hidden, self.cfg.dim,
            )

    def _load_open_clip(self) -> None:
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise EncoderUnavailable(
                "open_clip_torch is not installed but embedding.siglip.backend is "
                "'open_clip' - install it or switch the backend to 'transformers'"
            ) from exc

        try:
            model, _, _ = open_clip.create_model_and_transforms(
                self.cfg.model_id, pretrained=self.cfg.pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(self.cfg.model_id)
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailable(
                f"Could not load open_clip model {self.cfg.model_id!r}: {exc}"
            ) from exc

        model.eval().to(self.cfg.device)
        self._model = model

    # ------------------------------------------------------------------
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            if self._backend == "open_clip":
                tokens = self._tokenizer(texts).to(self.cfg.device)
                features = self._model.encode_text(tokens)
            else:
                batch = self._tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                features = self._model.get_text_features(**batch)
        return features.detach().float().cpu().numpy()


def _declared_text_dim(model: Any) -> int | None:
    """Best-effort read of the text tower's output width."""
    for attr in ("text_config", "config"):
        node = getattr(model, "config", None)
        node = getattr(node, attr, None) if attr != "config" else node
        size = getattr(node, "hidden_size", None) if node is not None else None
        if isinstance(size, int):
            return size
    return None
