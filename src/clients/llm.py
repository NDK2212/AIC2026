"""ChatNVIDIA wrapper with defensive JSON extraction, retries and caching.

``enable_thinking=True`` means the raw completion may contain ``<think>``
blocks, markdown fences and chatter around the object we actually want.  The
extraction here is deliberately paranoid: strip reasoning, strip fences, then
scan for a *balanced* brace span rather than trusting a regex.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Iterable

from ..config import LLMConfig, ModelFallbackConfig
from ..logging_utils import get_logger
from ..schemas import ConfigError, LLMParseError
from ..utils.cache import DiskCache, sha256_key
from ..utils.text_norm import strip_code_fences, strip_think_blocks

log = get_logger(__name__)

_RETRY_MARKERS = (
    "429", "500", "502", "503", "504",
    "rate limit", "ratelimit", "too many requests",
    "timeout", "timed out", "temporarily unavailable",
    "connection", "overloaded", "service unavailable",
)

_JSON_RETRY_HINT = (
    "Your previous output was not valid JSON. "
    "Output ONLY the JSON object, with no explanation, no <think> block and no "
    "markdown fences."
)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first balanced top-level JSON object out of raw model output.

    Raises :class:`LLMParseError` when no parsable object exists.
    """
    if not text or not text.strip():
        raise LLMParseError("LLM returned an empty response")

    cleaned = strip_code_fences(strip_think_blocks(text)).strip()
    if not cleaned:
        cleaned = strip_code_fences(text).strip()

    # Fast path: the whole thing is already an object.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for span in _balanced_spans(cleaned):
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = cleaned[:400].replace("\n", " ")
    raise LLMParseError(f"No balanced JSON object found in LLM output: {preview!r}")


def _balanced_spans(text: str) -> Iterable[str]:
    """Yield balanced ``{...}`` substrings, longest (outermost) first.

    String literals and escapes are tracked so braces inside quoted values do
    not throw the scan off.
    """
    starts: list[int] = []
    spans: list[tuple[int, int]] = []
    in_string = False
    escaped = False
    quote = ""

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue
        if ch in "\"'":
            in_string, quote = True, ch
        elif ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            start = starts.pop()
            spans.append((start, i + 1))

    # Outermost objects first: widest span wins, then earliest position.
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    for start, end in spans:
        yield text[start:end]


class LLMClient:
    """Chat + JSON-mode access to the configured LLM."""

    def __init__(self, cfg: LLMConfig, cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self._client: Any | None = None
        self._primary_unavailable = False

    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        """Lazily construct the underlying ChatNVIDIA client."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _get_client_for_key(self, api_key: str) -> Any:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "api_key": api_key,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "timeout": self.cfg.timeout,
        }
        if self.cfg.enable_thinking:
            kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        try:
            client = ChatNVIDIA(**kwargs)
        except Exception as exc:
            if "chat_template_kwargs" in kwargs:
                kwargs.pop("chat_template_kwargs", None)
                client = ChatNVIDIA(**kwargs)
            else:
                raise
        self._set_transport_timeout(client)
        return client

    def _set_transport_timeout(self, client: Any) -> None:
        """ChatNVIDIA may leave its internal transport at the 60s default."""
        for attr in ("_client", "_async_client"):
            transport = getattr(client, attr, None)
            if transport is not None and hasattr(transport, "timeout"):
                transport.timeout = float(self.cfg.timeout)

    def _build_client(self) -> Any:
        from .key_pool import GLOBAL_KEY_POOL
        try:
            key = GLOBAL_KEY_POOL.next_key()
        except ValueError:
            key = os.environ.get(self.cfg.api_key_env, "").strip()
        if not key:
            raise ConfigError(
                f"{self.cfg.api_key_env} is not set. Fill in your API key in .env or via Web UI."
            )
        return self._get_client_for_key(key)

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, *, use_cache: bool = True) -> str:
        """Single completion, cached by ``sha256(model + system + user)``."""
        key = sha256_key(
            self.cfg.model,
            self.cfg.fallback.model if self.cfg.fallback.enabled else "",
            self.cfg.temperature,
            system,
            user,
        )
        if use_cache and self.cache is not None:
            hit = self.cache.get_json("llm", key)
            if isinstance(hit, dict) and "content" in hit:
                log.debug("LLM cache hit %s", key[:12])
                return str(hit["content"])

        content = self._invoke_with_retry(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}]
        )
        if use_cache and self.cache is not None:
            self.cache.set_json("llm", key, {"content": content})
        return content

    def chat_json(
        self,
        system: str,
        user: str,
        schema_hint: str | None = None,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Completion that must yield a JSON object; retries on parse failure."""
        key = sha256_key(
            self.cfg.model,
            self.cfg.fallback.model if self.cfg.fallback.enabled else "",
            self.cfg.temperature,
            system,
            user,
            schema_hint,
        )
        if use_cache and self.cache is not None:
            hit = self.cache.get_json("llm_json", key)
            if isinstance(hit, dict):
                log.debug("LLM JSON cache hit %s", key[:12])
                return hit

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if schema_hint:
            messages[0]["content"] = f"{system}\n\nExpected JSON schema:\n{schema_hint}"

        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(1, self.cfg.json_retries + 1):
            raw = self._invoke_with_retry(messages)
            last_raw = raw
            try:
                parsed = extract_json_object(raw)
            except LLMParseError as exc:
                last_error = exc
                log.warning(
                    "LLM JSON parse failed (attempt %d/%d): %s",
                    attempt, self.cfg.json_retries, exc,
                )
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw[-2000:]},
                    {"role": "user", "content": _JSON_RETRY_HINT},
                ]
                continue
            if use_cache and self.cache is not None:
                self.cache.set_json("llm_json", key, parsed)
            return parsed

        raise LLMParseError(
            f"LLM did not return valid JSON after {self.cfg.json_retries} attempts. "
            f"Last error: {last_error}. Raw output:\n{last_raw[:2000]}"
        )

    # ------------------------------------------------------------------
    def _invoke_with_retry(self, messages: list[dict[str, str]]) -> str:
        """Call the primary model, then switch to the configured local fallback."""
        fallback = self.cfg.fallback
        if self._primary_unavailable and fallback.enabled:
            return self._invoke_fallback_with_retry(messages, fallback)
        try:
            return self._invoke_primary_with_retry(messages)
        except Exception as primary_exc:  # noqa: BLE001
            if not fallback.enabled:
                raise
            self._primary_unavailable = True
            log.warning(
                "Primary LLM %s failed (%s) - switching to fallback %s at %s",
                self.cfg.model,
                primary_exc,
                fallback.model,
                fallback.base_url,
            )
            try:
                return self._invoke_fallback_with_retry(messages, fallback)
            except Exception as fallback_exc:  # noqa: BLE001
                raise LLMParseError(
                    "Both primary and fallback LLM calls failed. "
                    f"Primary: {primary_exc}. Fallback: {fallback_exc}"
                ) from fallback_exc

    def _invoke_primary_with_retry(self, messages: list[dict[str, str]]) -> str:
        """Call the primary provider with its retry policy."""
        if self.cfg.provider == "openai-compatible":
            return self._invoke_openai_with_retry(
                messages=messages,
                model=self.cfg.model,
                base_url=self.cfg.base_url,
                api_key_env=self.cfg.api_key_env,
                max_retries=self.cfg.max_retries,
                retry_backoff=self.cfg.retry_backoff,
                timeout=self.cfg.timeout,
                label="primary LLM",
            )

        from .key_pool import GLOBAL_KEY_POOL
        keys = GLOBAL_KEY_POOL.get_keys() or [os.environ.get(self.cfg.api_key_env, "").strip()]
        last_exc: Exception | None = None

        total_attempts = max(1, self.cfg.max_retries)
        for attempt in range(1, total_attempts + 1):
            key = keys[(attempt - 1) % len(keys)] if keys else ""
            if not key:
                raise ConfigError("No API key available")
            try:
                client = self._get_client_for_key(key)
                response = client.invoke(messages)
                return _response_text(response)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_retryable(exc) and attempt >= len(keys):
                    raise
                delay = min(self.cfg.retry_backoff ** attempt + random.uniform(0, 0.5), 10.0)
                log.warning(
                    "LLM call failed with key ...%s (attempt %d/%d): %s - switching key and retrying in %.1fs",
                    key[-4:] if len(key) > 4 else "key",
                    attempt,
                    total_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise LLMParseError(f"LLM call failed after {total_attempts} attempts: {last_exc}")

    def _invoke_fallback_with_retry(
        self,
        messages: list[dict[str, str]],
        fallback: ModelFallbackConfig,
    ) -> str:
        if fallback.provider != "openai-compatible":
            raise ConfigError(
                "Only openai-compatible LLM fallback endpoints are currently supported"
            )
        return self._invoke_openai_with_retry(
            messages=messages,
            model=fallback.model,
            base_url=fallback.base_url,
            api_key_env=fallback.api_key_env,
            max_retries=fallback.max_retries,
            retry_backoff=fallback.retry_backoff,
            timeout=fallback.timeout,
            label="fallback LLM",
        )

    def _invoke_openai_with_retry(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        base_url: str | None,
        api_key_env: str | None,
        max_retries: int,
        retry_backoff: float,
        timeout: int,
        label: str,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._call_openai(
                    messages=messages,
                    model=model,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == max_retries or not _is_retryable(exc):
                    raise
                delay = min(retry_backoff ** attempt + random.uniform(0, 0.5), 10.0)
                log.warning(
                    "%s call failed (attempt %d/%d): %s - retrying in %.1fs",
                    label,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise LLMParseError(f"{label} failed: {last_exc}")  # pragma: no cover

    def _call_openai(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        base_url: str | None,
        api_key_env: str | None,
        timeout: int,
    ) -> str:
        """POST one text completion to an OpenAI-compatible endpoint."""
        import requests

        if not base_url:
            raise ConfigError("An OpenAI-compatible LLM requires base_url")
        headers = {"Accept": "application/json"}
        api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "max_tokens": self.cfg.max_tokens,
            },
            headers=headers,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise LLMParseError(
                f"LLM HTTP {response.status_code}: {response.text[:400]}"
            )
        data = response.json()
        try:
            return _response_text(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMParseError(
                f"Invalid OpenAI-compatible LLM response: {str(data)[:400]}"
            ) from exc


def _is_retryable(exc: Exception) -> bool:
    """Heuristically classify provider errors as transient."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRY_MARKERS)


def _response_text(response: Any) -> str:
    """Normalise LangChain message objects into a plain string."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal / block-style responses
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)
