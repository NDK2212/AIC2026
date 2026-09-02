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

from ..config import ModelFallbackConfig, VLMConfig
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
    """Ask a vision model about one or more frames plus textual metadata."""

    def __init__(self, cfg: VLMConfig, cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self._client: Any | None = None
        self._primary_unavailable = False

    # ------------------------------------------------------------------
    def _get_client_for_key(self, api_key: str) -> Any:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "api_key": api_key,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "timeout": self.cfg.timeout,
        }
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        client = ChatNVIDIA(**kwargs)
        for attr in ("_client", "_async_client"):
            transport = getattr(client, attr, None)
            if transport is not None and hasattr(transport, "timeout"):
                transport.timeout = float(self.cfg.timeout)
        return client

    @property
    def client(self) -> Any:
        """Lazily built ChatNVIDIA client (nvidia provider only)."""
        from .key_pool import GLOBAL_KEY_POOL
        try:
            key = GLOBAL_KEY_POOL.next_key()
        except ValueError:
            key = self._api_key(self.cfg.api_key_env, fallback_to_nvidia=True)
        if not key:
            raise ConfigError(
                f"{self.cfg.api_key_env} or NVIDIA_API_KEY is required for the NVIDIA VLM"
            )
        return self._get_client_for_key(key)

    @staticmethod
    def _api_key(api_key_env: str | None, *, fallback_to_nvidia: bool = False) -> str:
        key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
        if not key and fallback_to_nvidia:
            key = os.environ.get("NVIDIA_API_KEY", "").strip()
        return key

    # ------------------------------------------------------------------
    def ask(
        self,
        image_path: Path | str,
        system: str,
        question: str,
        *,
        use_cache: bool = True,
    ) -> str:
        """Backward-compatible single-image wrapper around :meth:`ask_many`."""
        return self.ask_many(
            [image_path], system, question, contexts=None, use_cache=use_cache
        )

    def ask_many(
        self,
        image_paths: list[Path | str],
        system: str,
        question: str,
        *,
        contexts: list[str] | None = None,
        use_cache: bool = True,
    ) -> str:
        """Answer from multiple images with optional per-image text context."""
        if not image_paths:
            raise PipelineError("VLM ask_many requires at least one image")
        if contexts is not None and len(contexts) != len(image_paths):
            raise PipelineError("VLM contexts must have the same length as image_paths")

        encoded: list[tuple[str, str, str]] = []
        fingerprints: list[Any] = []
        for index, image_path in enumerate(image_paths):
            b64, mime = encode_image(image_path, self.cfg.image_max_side)
            context = str(contexts[index] if contexts is not None else "").strip()
            encoded.append((b64, mime, context))
            fingerprints.extend((context, b64[:4096], len(b64)))
        key = sha256_key(
            self.cfg.model,
            self.cfg.fallback.model if self.cfg.fallback.enabled else "",
            system,
            question,
            *fingerprints,
        )

        if use_cache and self.cache is not None:
            hit = self.cache.get_json("vlm", key)
            if isinstance(hit, dict) and "content" in hit:
                return str(hit["content"])

        content = self._call_many_with_retry(encoded, system, question)
        if use_cache and self.cache is not None:
            self.cache.set_json("vlm", key, {"content": content})
        return content

    # ------------------------------------------------------------------
    def _call_with_retry(self, b64: str, mime: str, system: str, question: str) -> str:
        """Backward-compatible single-image transport wrapper."""
        return self._call_many_with_retry([(b64, mime, "")], system, question)

    def _call_many_with_retry(
        self,
        images: list[tuple[str, str, str]],
        system: str,
        question: str,
    ) -> str:
        fallback = self.cfg.fallback
        if self._primary_unavailable and fallback.enabled:
            return self._call_fallback_many_with_retry(
                images, system, question, fallback
            )
        try:
            return self._retry_provider(
                images=images,
                system=system,
                question=question,
                provider=self.cfg.provider,
                model=self.cfg.model,
                base_url=self.cfg.base_url,
                api_key_env=self.cfg.api_key_env,
                max_retries=self.cfg.max_retries,
                retry_backoff=self.cfg.retry_backoff,
                timeout=self.cfg.timeout,
                is_primary=True,
            )
        except Exception as primary_exc:  # noqa: BLE001
            if not fallback.enabled:
                raise
            self._primary_unavailable = True
            log.warning(
                "Primary VLM %s failed (%s) - switching to fallback %s at %s",
                self.cfg.model,
                primary_exc,
                fallback.model,
                fallback.base_url,
                extra={"progress": {
                    "phase": "answer",
                    "status": "running",
                    "title": "Kimi K3 lỗi, đang chuyển sang Qwen local",
                    "detail": f"Fallback: {fallback.model}",
                }},
            )
            try:
                return self._call_fallback_many_with_retry(
                    images, system, question, fallback
                )
            except Exception as fallback_exc:  # noqa: BLE001
                raise PipelineError(
                    "Both primary and fallback VLM calls failed. "
                    f"Primary: {primary_exc}. Fallback: {fallback_exc}"
                ) from fallback_exc

    def _call_fallback_with_retry(
        self,
        b64: str,
        mime: str,
        system: str,
        question: str,
        fallback: ModelFallbackConfig,
    ) -> str:
        """Backward-compatible single-image fallback wrapper."""
        return self._call_fallback_many_with_retry(
            [(b64, mime, "")], system, question, fallback
        )

    def _call_fallback_many_with_retry(
        self,
        images: list[tuple[str, str, str]],
        system: str,
        question: str,
        fallback: ModelFallbackConfig,
    ) -> str:
        return self._retry_provider(
            images=images,
            system=system,
            question=question,
            provider=fallback.provider,
            model=fallback.model,
            base_url=fallback.base_url,
            api_key_env=fallback.api_key_env,
            max_retries=fallback.max_retries,
            retry_backoff=fallback.retry_backoff,
            timeout=fallback.timeout,
            is_primary=False,
        )

    def _retry_provider(
        self,
        *,
        images: list[tuple[str, str, str]],
        system: str,
        question: str,
        provider: str,
        model: str,
        base_url: str | None,
        api_key_env: str | None,
        max_retries: int,
        retry_backoff: float,
        timeout: int,
        is_primary: bool,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                if provider == "openai-compatible":
                    return self._call_openai_many(
                        images,
                        system,
                        question,
                        model=model,
                        base_url=base_url,
                        api_key_env=api_key_env,
                        timeout=timeout,
                    )
                if not is_primary:
                    raise ConfigError(
                        "Only openai-compatible VLM fallback endpoints are currently supported"
                    )
                return self._call_nvidia_many(images, system, question)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                retryable = any(m in f"{exc}".lower() for m in _RETRY_MARKERS)
                if not retryable or attempt == max_retries:
                    raise
                delay = retry_backoff ** attempt + random.uniform(0, 0.5)
                log.warning(
                    "%s VLM call failed (attempt %d/%d): %s - retrying in %.1fs",
                    "Primary" if is_primary else "Fallback",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise PipelineError(f"VLM call failed: {last_exc}")  # pragma: no cover

    def _call_nvidia(self, b64: str, mime: str, system: str, question: str) -> str:
        """Backward-compatible single-image NVIDIA wrapper."""
        return self._call_nvidia_many([(b64, mime, "")], system, question)

    def _call_nvidia_many(
        self,
        images: list[tuple[str, str, str]],
        system: str,
        question: str,
    ) -> str:
        """LangChain multimodal blocks with interleaved metadata and images."""
        content = _multimodal_content(images, question)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        try:
            return _text_of(self.client.invoke(messages))
        except Exception as exc:  # noqa: BLE001
            if any(m in f"{exc}".lower() for m in _RETRY_MARKERS):
                raise
            # Some NVIDIA vision endpoints only accept an inline <img> tag.
            log.debug("Content-block call rejected (%s) - retrying with inline img", exc)
            inline_parts = [question]
            for b64, mime, context in images:
                if context:
                    inline_parts.append(context)
                inline_parts.append(f'<img src="data:{mime};base64,{b64}" />')
            inline = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": "\n\n".join(inline_parts),
                },
            ]
            return _text_of(self.client.invoke(inline))

    def _call_openai(
        self,
        b64: str,
        mime: str,
        system: str,
        question: str,
        *,
        model: str,
        base_url: str | None,
        api_key_env: str | None,
        timeout: int,
    ) -> str:
        """Backward-compatible single-image OpenAI wrapper."""
        return self._call_openai_many(
            [(b64, mime, "")],
            system,
            question,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout=timeout,
        )

    def _call_openai_many(
        self,
        images: list[tuple[str, str, str]],
        system: str,
        question: str,
        *,
        model: str,
        base_url: str | None,
        api_key_env: str | None,
        timeout: int,
    ) -> str:
        """POST interleaved text and images to an OpenAI-compatible endpoint."""
        import requests

        if not base_url:
            raise ConfigError("An OpenAI-compatible VLM requires base_url")
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _multimodal_content(images, question)},
            ],
        }
        headers = {"Accept": "application/json"}
        api_key = self._api_key(
            api_key_env,
            fallback_to_nvidia=bool(api_key_env == self.cfg.api_key_env),
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            raise PipelineError(f"VLM HTTP {response.status_code}: {response.text[:400]}")
        data = response.json()
        return str(data["choices"][0]["message"]["content"])


def _multimodal_content(
    images: list[tuple[str, str, str]], question: str
) -> list[dict[str, Any]]:
    """Interleave each frame's metadata immediately before its image."""
    content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    for index, (b64, mime, context) in enumerate(images, start=1):
        label = context or f"Frame {index}"
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    return content


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
