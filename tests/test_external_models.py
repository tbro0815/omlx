# SPDX-License-Identifier: Apache-2.0
"""Tests for external (remote API) models: registry + RemoteOpenAIEngine."""

import json
import stat

import httpx
import pytest

from omlx.engine.remote import RemoteOpenAIEngine
from omlx.external_models import (
    ExternalModelRegistry,
    SelfEndpointError,
    make_model_id,
    validate_not_self_endpoint,
)


class TestMakeModelId:
    def test_flattens_slashes(self):
        assert make_model_id("openrouter", "deepseek/deepseek-v4") == (
            "ext.or.deepseek-deepseek-v4"
        )

    def test_sanitizes_odd_characters(self):
        mid = make_model_id("openai_compatible", "weird model:v2 (beta)")
        assert " " not in mid and ":" not in mid and "(" not in mid
        assert mid.startswith("ext.oai.")


class TestSelfEndpointGuard:
    def test_rejects_loopback_same_port(self):
        with pytest.raises(SelfEndpointError):
            validate_not_self_endpoint("http://127.0.0.1:8000/v1", 8000)

    def test_rejects_localhost_same_port(self):
        with pytest.raises(SelfEndpointError):
            validate_not_self_endpoint("http://localhost:8000/v1", 8000)

    def test_allows_loopback_other_port(self):
        validate_not_self_endpoint("http://127.0.0.1:1234/v1", 8000)

    def test_allows_remote_host(self):
        validate_not_self_endpoint("https://openrouter.ai/api/v1", 8000)

    def test_rejects_malformed_url(self):
        with pytest.raises(ValueError):
            validate_not_self_endpoint("ftp://openrouter.ai/api/v1", 8000)
        with pytest.raises(ValueError):
            validate_not_self_endpoint("not a url", 8000)


class TestRegistry:
    def test_add_list_remove_roundtrip(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            remote_model="deepseek/deepseek-v4",
            api_key="sk-test",
            server_port=8000,
            context_length=131072,
        )
        assert reg.get(m.model_id) is not None
        assert len(reg.list()) == 1

        # persisted across a fresh instance
        reg2 = ExternalModelRegistry(tmp_path)
        assert reg2.get(m.model_id) is not None
        assert reg2.get_api_key("https://openrouter.ai/api/v1") == "sk-test"

        assert reg2.remove(m.model_id) is True
        assert reg2.list() == []
        # endpoint key removed with the last model using it
        assert reg2.get_api_key("https://openrouter.ai/api/v1") is None

    def test_duplicate_add_rejected(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        reg.add(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            remote_model="x/y",
            server_port=8000,
        )
        with pytest.raises(ValueError):
            reg.add(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                remote_model="x/y",
                server_port=8000,
            )

    def test_no_secrets_in_records_file(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        reg.add(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            remote_model="x/y",
            api_key="sk-very-secret",
            server_port=8000,
        )
        records = (tmp_path / "external_models.json").read_text()
        assert "sk-very-secret" not in records

    def test_key_file_permissions(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        reg.set_api_key("https://openrouter.ai/api/v1", "sk-test")
        mode = stat.S_IMODE((tmp_path / "external_keys.json").stat().st_mode)
        assert mode == 0o600

    def test_self_endpoint_rejected_on_add(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        with pytest.raises(SelfEndpointError):
            reg.add(
                provider="openai_compatible",
                base_url="http://127.0.0.1:8000/v1",
                remote_model="x",
                server_port=8000,
            )


def _mock_engine(handler) -> RemoteOpenAIEngine:
    engine = RemoteOpenAIEngine(
        model_name="ext.or.test",
        base_url="https://mock.example/v1",
        remote_model="vendor/test-model",
        api_key="sk-test",
        provider="openrouter",
    )

    async def _start():
        engine._client = httpx.AsyncClient(
            base_url="https://mock.example/v1",
            transport=httpx.MockTransport(handler),
        )
        engine._loaded = True

    engine.start = _start
    return engine


class TestRemoteEngine:
    async def test_chat_maps_usage_and_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/chat/completions")
            body = json.loads(request.content)
            assert body["model"] == "vendor/test-model"
            assert body["messages"][0]["content"] == "Hola!"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "¡Hola! ¿Qué tal?"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                },
            )

        engine = _mock_engine(handler)
        await engine.start()
        out = await engine.chat(messages=[{"role": "user", "content": "Hola!"}])
        assert out.text == "¡Hola! ¿Qué tal?"
        assert out.prompt_tokens == 12
        assert out.completion_tokens == 7
        assert out.cached_tokens == 4
        assert out.finish_reason == "stop"
        await engine.stop()

    async def test_generate_uses_completions_for_openrouter(self):
        seen_paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "choices": [{"text": "def foo():\n    pass", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 6},
                },
            )

        engine = _mock_engine(handler)
        await engine.start()
        out = await engine.generate(prompt="Write foo()", max_tokens=32)
        assert out.text.startswith("def foo")
        assert seen_paths and seen_paths[0].endswith("/completions")
        assert not seen_paths[0].endswith("/chat/completions")
        await engine.stop()

    async def test_api_error_surfaces_detail(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402, json={"error": {"message": "Insufficient credits"}}
            )

        engine = _mock_engine(handler)
        await engine.start()
        with pytest.raises(RuntimeError, match="Insufficient credits"):
            await engine.chat(messages=[{"role": "user", "content": "hi"}])
        await engine.stop()

    async def test_stream_chat_accumulates_deltas(self):
        sse = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
            'data: {"usage":{"prompt_tokens":3,"completion_tokens":2},"choices":[]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse.encode(),
                headers={"Content-Type": "text/event-stream"},
            )

        engine = _mock_engine(handler)
        await engine.start()
        outputs = []
        async for out in engine.stream_chat(
            messages=[{"role": "user", "content": "hi"}]
        ):
            outputs.append(out)
        assert outputs[-1].finished is True
        assert outputs[-1].text == "Hello"
        assert outputs[-1].prompt_tokens == 3
        assert outputs[-1].finish_reason == "stop"
        await engine.stop()
