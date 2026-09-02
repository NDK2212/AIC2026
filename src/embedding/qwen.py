"""Qwen multimodal text tower.

Supports sentence-transformers (default) to match the Qwen embedding extraction
used during keyframe ingestion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..logging_utils import get_logger
from ..schemas import EncoderUnavailable
from .base import TextEncoder

log = get_logger(__name__)


class QwenTextEncoder(TextEncoder):
    """Encodes queries with the Qwen embedding model."""

    def __init__(self, cfg: Any, cache: Any = None) -> None:
        super().__init__(cfg, cache)
        self._model: Any = None
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

        if self._backend == "qwen3_vl_embedder":
            try:
                self._load_qwen3_vl_embedder()
                return
            except Exception as exc:
                raise EncoderUnavailable(
                    f"Could not load Qwen3VLEmbedder: {exc}"
                ) from exc

        if self._backend == "sentence_transformers":
            try:
                self._load_sentence_transformers()
                return
            except Exception as exc:
                raise EncoderUnavailable(
                    "Could not load Qwen via sentence-transformers. Install the "
                    f"image dependencies with `make install`: {exc}"
                ) from exc

        if self._backend == "transformers":
            self._load_transformers()
            return

        raise EncoderUnavailable(
            "embedding.qwen.backend must be sentence_transformers|"
            f"qwen3_vl_embedder|transformers, got {self._backend!r}"
        )

    def _load_qwen3_vl_embedder(self) -> None:
        import importlib.util
        from huggingface_hub import hf_hub_download

        try:
            script_path = hf_hub_download(self.cfg.model_id, filename="scripts/qwen3_vl_embedding.py")
            spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
            qwen_mod = importlib.util.module_from_spec(spec)
            import sys
            sys.modules["qwen3_vl_embedding"] = qwen_mod
            spec.loader.exec_module(qwen_mod)

            dtype = self._torch.float16 if (self.cfg.device != "cpu" and self._torch.cuda.is_available()) else self._torch.float32
            log.info("Loading Qwen3VLEmbedder (%s) on %s (dtype=%s)", self.cfg.model_id, self.cfg.device, dtype)
            self._embedder = qwen_mod.Qwen3VLEmbedder(
                self.cfg.model_id,
                torch_dtype=dtype,
            )
            self._backend = "qwen3_vl_embedder"
        except Exception as exc:
            raise EncoderUnavailable(f"Failed loading Qwen3VLEmbedder: {exc}") from exc

    def _load_sentence_transformers(self) -> None:
        import types
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.base.modules import Transformer
        from sentence_transformers.sentence_transformer.modules import Normalize, Pooling

        # The Qwen checkpoint references old leaf-module paths. Alias only
        # those leaves; replacing the parent packages removes Router and breaks
        # sentence-transformers 6.x during model-card registration.
        m_base_mod_trans = types.ModuleType("sentence_transformers.base.modules.transformer")
        m_base_mod_trans.Transformer = Transformer

        import sys
        sys.modules["sentence_transformers.base.modules.transformer"] = m_base_mod_trans

        m_st_mod_pool = types.ModuleType("sentence_transformers.sentence_transformer.modules.pooling")
        m_st_mod_pool.Pooling = Pooling
        m_st_mod_norm = types.ModuleType("sentence_transformers.sentence_transformer.modules.normalize")
        m_st_mod_norm.Normalize = Normalize

        sys.modules["sentence_transformers.sentence_transformer.modules.pooling"] = m_st_mod_pool
        sys.modules["sentence_transformers.sentence_transformer.modules.normalize"] = m_st_mod_norm

        model_kwargs = {}
        if self.cfg.device != "cpu" and self._torch.cuda.is_available():
            model_kwargs["torch_dtype"] = self._torch.float16
        else:
            model_kwargs["torch_dtype"] = getattr(self._torch, "bfloat16", self._torch.float32)

        try:
            log.info("Loading Qwen model via sentence-transformers: %s on %s (dtype=%s)", self.cfg.model_id, self.cfg.device, model_kwargs.get("torch_dtype"))
            self._model = SentenceTransformer(
                self.cfg.model_id,
                device=self.cfg.device,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            self._backend = "sentence_transformers"
        except Exception as exc:
            log.warning("Could not load Qwen via sentence_transformers with %s: %s. Retrying default...", model_kwargs.get("torch_dtype"), exc)
            try:
                self._model = SentenceTransformer(
                    self.cfg.model_id,
                    device=self.cfg.device,
                )
                self._backend = "sentence_transformers"
            except Exception as exc2:
                raise EncoderUnavailable(f"Could not load Qwen model via sentence_transformers: {exc2}") from exc2

    def _load_transformers(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise EncoderUnavailable(
                "transformers is not installed - run `pip install -r requirements.txt`"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
            dtype = self._torch.float16 if (self.cfg.device != "cpu" and self._torch.cuda.is_available()) else getattr(self._torch, "bfloat16", self._torch.float32)
            try:
                self._model = AutoModel.from_pretrained(self.cfg.model_id, torch_dtype=dtype, trust_remote_code=True).to(self.cfg.device).eval()
            except Exception:
                self._model = AutoModel.from_pretrained(self.cfg.model_id, torch_dtype=self._torch.float32, trust_remote_code=True).to(self.cfg.device).eval()
            self._backend = "transformers"
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailable(
                f"Could not load Qwen model {self.cfg.model_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        if self._backend == "qwen3_vl_embedder" and hasattr(self, "_embedder"):
            inputs = [{"text": t} for t in texts]
            embeddings = self._embedder.process(inputs, normalize=True)
            if hasattr(embeddings, "cpu"):
                return embeddings.cpu().float().numpy().astype(np.float32)
            return np.asarray(embeddings, dtype=np.float32)

        if self._backend == "sentence_transformers" and self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=len(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)

        # transformers fallback
        torch = self._torch
        import torch.nn.functional as F
        with torch.inference_mode():
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.cfg.device)
            outputs = self._model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                # last token pooling
                flipped_tensor = attention_mask.flip(dims=[1])
                last_one_positions = flipped_tensor.argmax(dim=1)
                col = attention_mask.shape[1] - last_one_positions - 1
                row = torch.arange(token_embeddings.shape[0], device=token_embeddings.device)
                features = token_embeddings[row, col]
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs[0][:, 0]
            features = F.normalize(features, p=2, dim=-1)
            return features.cpu().float().numpy().astype(np.float32)
