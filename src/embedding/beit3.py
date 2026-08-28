"""BEiT-3 text tower.

BEiT-3 has no first-class ``transformers`` implementation, so loading it needs
the original ``unilm/beit3`` code plus a local checkpoint.  When that is not
available this encoder raises :class:`EncoderUnavailable` with an actionable
message and the pipeline degrades to SigLIP-only retrieval - loudly, never
silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger
from ..schemas import EncoderUnavailable
from .base import TextEncoder

log = get_logger(__name__)

_HELP = (
    "Set embedding.beit3.checkpoint_path (and embedding.beit3.tokenizer_path to "
    "beit3.spm) in config/config.yaml, make the unilm/beit3 package importable, "
    "or set embedding.beit3.enabled: false to run the visual path on SigLIP alone."
)


class BEiT3TextEncoder(TextEncoder):
    """Encodes queries with the BEiT-3 text tower."""

    def __init__(self, cfg: Any, cache: Any = None) -> None:
        super().__init__(cfg, cache)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._mode = "torchscale"

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise EncoderUnavailable("torch is not installed") from exc
        self._torch = torch

        if self.cfg.backend == "transformers":
            self._load_transformers()
        else:
            self._load_torchscale()

    def _load_transformers(self) -> None:
        """Some forks publish BEiT-3 behind ``AutoModel``/``trust_remote_code``."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise EncoderUnavailable("transformers is not installed") from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_path or self.cfg.model_id, trust_remote_code=True
            )
            model = AutoModel.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailable(
                f"Could not load BEiT-3 via transformers ({self.cfg.model_id!r}): "
                f"{exc}. {_HELP}"
            ) from exc
        model.eval().to(self.cfg.device)
        self._model = model
        self._mode = "transformers"

    def _load_torchscale(self) -> None:
        """Load the reference BEiT-3 retrieval model from a local checkpoint."""
        checkpoint = self.cfg.checkpoint_path
        if not checkpoint or not Path(checkpoint).is_file():
            raise EncoderUnavailable(
                f"BEiT-3 checkpoint not found at {checkpoint!r}. {_HELP}"
            )

        torch = self._torch
        model = None

        # 1. Try loading via official BEiT3Wrapper from unilm repo
        try:
            import sys
            import types
            import math
            if "torch._six" not in sys.modules:
                six_mod = types.ModuleType("torch._six")
                six_mod.inf = math.inf
                six_mod.string_classes = (str, bytes)
                sys.modules["torch._six"] = six_mod

            for p in (Path(".hf_cache/unilm/beit3"), Path(".hf_cache/unilm"), Path("unilm/beit3"), Path("unilm")):
                if p.is_dir() and str(p.resolve()) not in sys.path:
                    sys.path.insert(0, str(p.resolve()))

            from modeling_utils import BEiT3Wrapper, _get_large_config
            with torch.device("cpu"):
                config = _get_large_config(img_size=224)
                model = BEiT3Wrapper(args=config)
        except Exception:
            model = None

        # 2. Fallback to timm model registry if BEiT3Wrapper not importable
        if model is None:
            try:
                import modeling_finetune  # noqa: F401
                import timm
                model_name = Path(str(self.cfg.model_id)).name.replace("-", "_")
                for name in ["beit3_large_patch16_224", f"beit3_{model_name}_patch16_224", f"beit3_{model_name}", model_name]:
                    try:
                        model = timm.create_model(name, pretrained=False)
                        break
                    except Exception:
                        pass
            except ImportError:
                pass

        if model is None:
            raise EncoderUnavailable(
                f"Could not instantiate BEiT-3 model architecture. {_HELP}"
            )

        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            weights = state.get("model", state) if isinstance(state, dict) else state
            missing, unexpected = model.load_state_dict(weights, strict=False)
            if missing:
                log.debug("BEiT-3 checkpoint is missing %d tensors", len(missing))
            if unexpected:
                log.debug("BEiT-3 checkpoint has %d unexpected tensors", len(unexpected))
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailable(
                f"Could not load BEiT-3 weights from {checkpoint}: {exc}. {_HELP}"
            ) from exc

        self._tokenizer = self._load_spm()
        if str(self.cfg.device).lower() not in ("cpu", "auto"):
            try:
                model.to(self.cfg.device)
            except Exception:
                pass
        model.eval()
        self._model = model
        self._mode = "torchscale"

    def _load_spm(self) -> Any:
        """Load the sentencepiece tokenizer BEiT-3 was trained with."""
        tokenizer_path = self.cfg.tokenizer_path
        if not tokenizer_path or not Path(tokenizer_path).is_file():
            raise EncoderUnavailable(
                f"BEiT-3 sentencepiece model not found at {tokenizer_path!r}. {_HELP}"
            )
        try:
            from transformers import XLMRobertaTokenizer

            return XLMRobertaTokenizer(str(tokenizer_path))
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailable(
                f"Could not load the BEiT-3 tokenizer: {exc}. {_HELP}"
            ) from exc

    # ------------------------------------------------------------------
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(self.cfg.device)
        attention_mask = batch["attention_mask"].to(self.cfg.device)
        padding_mask = (attention_mask == 0)

        with torch.inference_mode():
            if hasattr(self._model, "beit3"):
                out = self._model.beit3(
                    textual_tokens=input_ids,
                    text_padding_position=padding_mask,
                )
                features = _first_tensor(out)
            elif self._mode == "transformers":
                out = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                features = _first_tensor(out)
            else:
                out = self._model(
                    text_description=input_ids,
                    padding_mask=padding_mask,
                    only_infer=True,
                )
                features = _text_features(out)
        return features.detach().float().cpu().numpy()


def _first_tensor(output: Any) -> Any:
    """Pull an embedding tensor out of whatever shape the model returned."""
    if isinstance(output, dict):
        for k in ("encoder_out", "pooler_output", "last_hidden_state", "text_embeds"):
            if k in output and output[k] is not None:
                output = output[k]
                break
        else:
            output = list(output.values())[0]

    for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attr, None)
        if value is not None:
            return value[:, 0] if value.dim() == 3 else value
    if isinstance(output, (tuple, list)) and output:
        value = output[0]
        return value[:, 0] if getattr(value, "dim", lambda: 2)() == 3 else value
    if hasattr(output, "dim") and output.dim() == 3:
        return output[:, 0]
    return output


def _text_features(output: Any) -> Any:
    """BEiT-3 retrieval models return ``(vision_cls, language_cls)``."""
    if isinstance(output, (tuple, list)):
        return output[-1]
    return _first_tensor(output)
