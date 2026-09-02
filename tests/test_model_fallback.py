from __future__ import annotations

from unittest.mock import MagicMock

from src.clients.llm import LLMClient
from src.clients.vlm import VLMClient
from src.config import Config, LLMConfig, ModelFallbackConfig, VLMConfig


def local_fallback() -> ModelFallbackConfig:
    return ModelFallbackConfig(
        enabled=True,
        provider="openai-compatible",
        model="qwen3-vl:2b-instruct",
        base_url="http://127.0.0.1:11434/v1",
        max_retries=2,
        timeout=180,
    )


def test_default_config_enables_local_llm_and_vlm_fallbacks():
    cfg = Config.load("config/config.yaml", no_cache=True)

    assert cfg.llm.max_retries == 1
    assert cfg.llm.json_retries == 2
    assert cfg.llm.timeout == 30
    assert cfg.llm.fallback.enabled is True
    assert cfg.llm.fallback.model == "qwen3-vl:2b-instruct"
    assert cfg.vlm.max_retries == 1
    assert cfg.vlm.timeout == 45
    assert cfg.vlm.fallback.enabled is True
    assert cfg.vlm.fallback.model == "qwen3-vl:2b-instruct"
    assert cfg.vqa.mode == "vision"
    assert cfg.vqa.vlm_images_per_video == 4


def test_llm_switches_to_fallback_after_primary_failure():
    client = LLMClient(LLMConfig(model="remote", fallback=local_fallback()))
    client._invoke_primary_with_retry = MagicMock(side_effect=TimeoutError("remote timeout"))
    client._invoke_fallback_with_retry = MagicMock(return_value="local answer")
    messages = [{"role": "user", "content": "hello"}]

    assert client._invoke_with_retry(messages) == "local answer"
    assert client._invoke_with_retry(messages) == "local answer"
    client._invoke_primary_with_retry.assert_called_once_with(messages)
    assert client._invoke_fallback_with_retry.call_count == 2


def test_vlm_switches_to_fallback_after_primary_failure():
    client = VLMClient(VLMConfig(model="remote", fallback=local_fallback()))
    client._retry_provider = MagicMock(side_effect=TimeoutError("remote timeout"))
    client._call_fallback_many_with_retry = MagicMock(return_value="local visual answer")

    result = client._call_with_retry("base64", "image/jpeg", "system", "question")
    second = client._call_with_retry("base64", "image/jpeg", "system", "question")

    assert result == "local visual answer"
    assert second == "local visual answer"
    client._retry_provider.assert_called_once()
    assert client._call_fallback_many_with_retry.call_count == 2


def test_local_vlm_request_does_not_require_fake_api_key(monkeypatch):
    fallback = local_fallback()
    client = VLMClient(VLMConfig(model="remote", fallback=fallback))
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": "a yellow animal"}}]
    }
    post = MagicMock(return_value=response)
    monkeypatch.setattr("requests.post", post)

    answer = client._call_openai(
        "base64",
        "image/jpeg",
        "system",
        "question",
        model=fallback.model,
        base_url=fallback.base_url,
        api_key_env=None,
        timeout=fallback.timeout,
    )

    assert answer == "a yellow animal"
    assert "Authorization" not in post.call_args.kwargs["headers"]


def test_multimodal_request_interleaves_each_context_with_its_image(monkeypatch):
    fallback = local_fallback()
    client = VLMClient(VLMConfig(model="remote", fallback=fallback))
    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": "cá chẽm"}}]}
    post = MagicMock(return_value=response)
    monkeypatch.setattr("requests.post", post)

    answer = client._call_openai_many(
        [
            ("image-one", "image/jpeg", "Description: stuffing four fish"),
            ("image-two", "image/jpeg", "OCR: Cá chẽm nướng"),
        ],
        "system",
        "Đây là loài cá gì?",
        model=fallback.model,
        base_url=fallback.base_url,
        api_key_env=None,
        timeout=fallback.timeout,
    )

    assert answer == "cá chẽm"
    content = post.call_args.kwargs["json"]["messages"][1]["content"]
    assert [block["type"] for block in content] == [
        "text", "text", "image_url", "text", "image_url"
    ]
    assert "stuffing four fish" in content[1]["text"]
    assert "Cá chẽm nướng" in content[3]["text"]
