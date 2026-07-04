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

    async def test_thinking_toggle_maps_to_openrouter_reasoning(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "4"}}], "usage": {}},
            )

        engine = _mock_engine(handler)
        await engine.start()
        await engine.chat(
            messages=[{"role": "user", "content": "2+2?"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        assert seen.get("reasoning") == {"enabled": False}
        await engine.stop()

    async def test_stream_wraps_reasoning_deltas_in_think_tags(self):
        sse = (
            'data: {"choices":[{"delta":{"reasoning":"hmm"}}]}\n\n'
            'data: {"choices":[{"delta":{"reasoning":" more"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}\n\n'
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
        final = None
        async for out in engine.stream_chat(
            messages=[{"role": "user", "content": "hi"}]
        ):
            final = out
        assert final.text == "<think>hmm more</think>Answer"
        await engine.stop()


class TestAppleFMRegistry:
    def test_apple_fm_add_needs_no_endpoint_or_key(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="apple_fm",
            base_url="",
            remote_model="on-device",
            server_port=8000,
        )
        assert m.model_id == "ext.afm.on-device"
        assert m.base_url == "applefm://local"
        assert reg.get_api_key(m.base_url) is None
        # multimodal by default -> pool injects it as a vlm entry and the
        # chat UI enables image upload
        assert m.modality == "text+image"


class TestAppleFMEngine:
    def _engine_with_fake_sdk(self, chunks):
        """Build an AppleFMEngine wired to a fake apple_fm_sdk module."""
        import types

        from omlx.engine.apple_fm import AppleFMEngine

        fake = types.ModuleType("apple_fm_sdk")

        class FakeOptions:
            def __init__(self, sampling=None, temperature=None,
                         maximum_response_tokens=None):
                self.sampling = sampling
                self.temperature = temperature
                self.maximum_response_tokens = maximum_response_tokens

        class FakeSamplingMode:
            greedy = "greedy"

        class FakeSession:
            last = None

            def __init__(self, instructions=None, model=None, tools=None):
                self.instructions = instructions
                FakeSession.last = self

            async def respond(self, prompt, options=None, **kw):
                self.prompt = prompt
                self.options = options
                return "respuesta"

            async def stream_response(self, prompt, options=None):
                self.prompt = prompt
                for c in chunks:
                    yield c

        fake.GenerationOptions = FakeOptions
        fake.SamplingMode = FakeSamplingMode
        fake.LanguageModelSession = FakeSession

        engine = AppleFMEngine(model_name="ext.afm.on-device")
        engine._fm = fake
        engine._model = object()
        engine._loaded = True
        return engine, FakeSession

    async def test_chat_splits_system_into_instructions(self):
        engine, FakeSession = self._engine_with_fake_sdk([])
        out = await engine.chat(
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hola!"},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        assert out.text == "respuesta"
        assert FakeSession.last.instructions == "Be terse."
        assert FakeSession.last.prompt == "Hola!"
        # temperature 0 -> greedy sampling, max tokens mapped
        assert FakeSession.last.options.sampling == "greedy"
        assert FakeSession.last.options.maximum_response_tokens == 64

    async def test_multi_turn_flattening(self):
        engine, FakeSession = self._engine_with_fake_sdk([])
        await engine.chat(
            messages=[
                {"role": "user", "content": "One"},
                {"role": "assistant", "content": "Two"},
                {"role": "user", "content": "Three"},
            ],
        )
        prompt = FakeSession.last.prompt
        assert "User: One" in prompt and "Assistant: Two" in prompt
        assert prompt.endswith("Assistant:")

    async def test_stream_diffs_cumulative_snapshots(self):
        engine, _ = self._engine_with_fake_sdk(["1, 2", "1, 2, 3", "1, 2, 3."])
        outputs = []
        async for out in engine.stream_chat(
            messages=[{"role": "user", "content": "count"}]
        ):
            outputs.append(out)
        deltas = [o.new_text for o in outputs if not o.finished]
        assert deltas == ["1, 2", ", 3", "."]
        assert outputs[-1].finished is True
        assert outputs[-1].text == "1, 2, 3."


class TestAppleFMImages:
    def _engine_with_image_sdk(self):
        import types

        from omlx.engine.apple_fm import AppleFMEngine

        fake = types.ModuleType("apple_fm_sdk")

        class FakeOptions:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class FakeSamplingMode:
            greedy = "greedy"

        class FakeImageAttachment:
            def __init__(self, path, label=None):
                self.path = str(path)
                self.label = label

        class FakeSession:
            last = None

            def __init__(self, instructions=None, model=None, tools=None):
                self.instructions = instructions
                FakeSession.last = self

            async def respond(self, prompt, options=None, **kw):
                self.prompt = prompt
                return "a cat"

        fake.GenerationOptions = FakeOptions
        fake.SamplingMode = FakeSamplingMode
        fake.ImageAttachment = FakeImageAttachment
        fake.LanguageModelSession = FakeSession

        engine = AppleFMEngine(model_name="ext.afm.on-device")
        engine._fm = fake
        engine._model = object()
        engine._loaded = True
        return engine, FakeSession, FakeImageAttachment

    async def test_data_url_image_becomes_attachment(self):
        import base64
        import os

        engine, FakeSession, FakeImageAttachment = self._engine_with_image_sdk()
        png = base64.b64encode(b"\x89PNG fakebytes").decode()
        out = await engine.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png}"},
                        },
                    ],
                }
            ],
        )
        assert out.text == "a cat"
        prompt = FakeSession.last.prompt
        assert isinstance(prompt, list) and prompt[0] == "What is this?"
        assert isinstance(prompt[1], FakeImageAttachment)
        assert prompt[1].label == "image_1"
        # temp file cleaned up after the request
        assert not os.path.exists(prompt[1].path)

    async def test_history_images_become_placeholders(self):
        engine, FakeSession, FakeImageAttachment = self._engine_with_image_sdk()
        await engine.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "https://x/old.png"}},
                    ],
                },
                {"role": "assistant", "content": "seen"},
                {"role": "user", "content": "and now?"},
            ],
        )
        prompt = FakeSession.last.prompt
        # last user turn has no images -> plain text dialogue with placeholder
        assert isinstance(prompt, str)
        assert "[image]" in prompt and "and now?" in prompt
