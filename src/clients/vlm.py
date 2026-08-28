"""Vision-language model client used by the Q&A task.

Two providers are supported:

* ``nvidia`` - LangChain ``ChatNVIDIA`` with multimodal content blocks.
* ``openai-compatible`` - a plain ``/chat/completions`` POST.

Images are downscaled and base64-encoded before being sent, and every answer is
cached by ``sha256(model + prompt + image bytes)`` so re-runs are free.
"""

from __future__ import annotations

import base64
import io
import os
import random
import time
from pathlib import Path
from typing import Any

from ..config import VLMConfig
from ..logging_utils import get_logger
from ..schemas import ConfigError, PipelineError
from ..utils.cache import DiskCache, sha256_key

log = get_logger(__name__)

_RETRY_MARKERS = (
    "429", "500", "502", "503", "504", "rate limit", "too many requests",
    "timeout", "timed out", "connection", "overloaded",
)


def encode_image(path: Path | str, max_side: int = 768) -> tuple[str, str]:
    """Return ``(base64_jpeg, mime)`` for an image, downscaled to ``max_side``.

    Falls back to sending the original bytes when Pillow is unavailable.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise PipelineError(f"Keyframe image not found: {image_path}")

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return base64.b64encode(image_path.read_bytes()).decode("ascii"), "image/jpeg"

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        longest = max(img.size)
        if max_side and longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/jpeg"


class VLMClient:
    """Ask a vision model one question about one frame."""

    def __init__(self, cfg: VLMConfig, cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self._client: Any | None = None

    # ------------------------------------------------------------------
    def _get_client_for_key(self, api_key: str) -> Any:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "api_key": api_key,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        return ChatNVIDIA(**kwargs)

    @property
    def client(self) -> Any:
        """Lazily built ChatNVIDIA client (nvidia provider only)."""
        from .key_pool import GLOBAL_KEY_POOL
        try:
            key = GLOBAL_KEY_POOL.next_key()
        except ValueError:
            key = self._api_key()
        return self._get_client_for_key(key)

    # ------------------------------------------------------------------
    def ask(
        self,
        image_path: Path | str,
        system: str,
        question: str,
        *,
        use_cache: bool = True,
    ) -> str:
        """Return the raw model answer for one image + question."""
        b64, mime = encode_image(image_path, self.cfg.image_max_side)
        key = sha256_key(self.cfg.model, system, question, b64[:4096], len(b64))

        if use_cache and self.cache is not None:
            hit = self.cache.get_json("vlm", key)
            if isinstance(hit, dict) and "content" in hit:
                return str(hit["content"])

        content = self._call_with_retry(b64, mime, system, question)
        if use_cache and self.cache is not None:
            self.cache.set_json("vlm", key, {"content": content})
        return content

    # ------------------------------------------------------------------
    def _call_with_retry(self, b64: str, mime: str, system: str, question: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                if self.cfg.provider == "openai-compatible":
                    return self._call_openai(b64, mime, system, question)
                return self._call_nvidia(b64, mime, system, question)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                retryable = any(m in f"{exc}".lower() for m in _RETRY_MARKERS)
                if not retryable or attempt == self.cfg.max_retries:
                    raise
                delay = self.cfg.retry_backoff ** attempt + random.uniform(0, 0.5)
                log.warning(
                    "VLM call failed (attempt %d/%d): %s - retrying in %.1fs",
                    attempt, self.cfg.max_retries, exc, delay,
                )
                time.sleep(delay)
        raise PipelineError(f"VLM call failed: {last_exc}")  # pragma: no cover

    def _call_nvidia(self, b64: str, mime: str, system: str, question: str) -> str:
        """LangChain multimodal content blocks, with an NVIDIA-HTML fallback."""
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ]
        try:
            return _text_of(self.client.invoke(messages))
        except Exception as exc:  # noqa: BLE001
            if any(m in f"{exc}".lower() for m in _RETRY_MARKERS):
                raise
            # Some NVIDIA vision endpoints only accept an inline <img> tag.
            log.debug("Content-block call rejected (%s) - retrying with inline img", exc)
            inline = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f'{question} <img src="data:{mime};base64,{b64}" />',
                },
            ]
            return _text_of(self.client.invoke(inline))

    def _call_openai(self, b64: str, mime: str, system: str, question: str) -> str:
        """Plain HTTP POST against an OpenAI-compatible endpoint."""
        import requests

        if not self.cfg.base_url:
            raise ConfigError("vlm.base_url is required for the openai-compatible provider")
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                },
            ],
        }
        response = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Accept": "application/json",
            },
            timeout=120,
        )
        if response.status_code >= 400:
            raise PipelineError(f"VLM HTTP {response.status_code}: {response.text[:400]}")
        data = response.json()
        return str(data["choices"][0]["message"]["content"])


def _text_of(response: Any) -> str:
    """Normalise a LangChain response into a string."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
        )
    return str(content)
