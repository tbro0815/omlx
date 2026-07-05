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
        # endpoint key survives removal so delete-and-re-add (a normal
        # metadata-refresh flow) doesn't force re-entering the key
        assert reg2.get_api_key("https://openrouter.ai/api/v1") == "sk-test"

    def test_duplicate_add_upserts(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        reg.add(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            remote_model="x/y",
            server_port=8000,
        )
        # Re-adding the same provider+model refreshes metadata in place
        # (same derived id), it must not error or duplicate.
        m2 = reg.add(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            remote_model="x/y",
            server_port=8000,
            context_length=123_456,
        )
        assert len(reg.list()) == 1
        assert reg.get(m2.model_id).context_length == 123_456

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


# ── Provider B: Anthropic / OpenAI presets + subscription CLI relays ──


class TestProviderBIds:
    def test_new_provider_prefixes(self):
        assert make_model_id("anthropic", "claude-sonnet-5").startswith("ext.ant.")
        assert make_model_id("openai", "gpt-5.2").startswith("ext.openai.")
        assert make_model_id("claude_cli", "opus") == "ext.claude.opus"
        assert make_model_id("codex_cli", "default") == "ext.codex.default"


class TestPresetProviders:
    def test_anthropic_defaults_to_preset_endpoint(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="anthropic",
            base_url="",
            remote_model="claude-sonnet-5",
            api_key="sk-ant-test",
            server_port=8000,
        )
        assert m.base_url == "https://api.anthropic.com/v1"
        assert reg.get_api_key(m.base_url) == "sk-ant-test"

    def test_openai_defaults_to_preset_endpoint(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="openai", base_url="", remote_model="gpt-5.2",
            server_port=8000,
        )
        assert m.base_url == "https://api.openai.com/v1"

    def test_cli_provider_stores_no_key_and_needs_no_url(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="claude_cli",
            base_url="",
            remote_model="default",
            api_key="should-be-dropped",
            server_port=8000,
        )
        assert m.base_url == "cli://claude"
        assert reg.get_api_key(m.base_url) is None
        assert not (tmp_path / "external_keys.json").exists()


def _anthropic_mock_engine(handler) -> RemoteOpenAIEngine:
    engine = RemoteOpenAIEngine(
        model_name="ext.ant.test",
        base_url="https://api.anthropic.com/v1",
        remote_model="claude-sonnet-5",
        api_key="sk-ant-test",
        provider="anthropic",
    )

    async def _start():
        engine._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            transport=httpx.MockTransport(handler),
        )
        engine._loaded = True

    engine.start = _start
    return engine


class TestAnthropicEngine:
    async def test_start_sets_anthropic_headers(self):
        engine = RemoteOpenAIEngine(
            model_name="ext.ant.test",
            base_url="https://api.anthropic.com/v1",
            remote_model="claude-sonnet-5",
            api_key="sk-ant-test",
            provider="anthropic",
        )
        await engine.start()
        try:
            assert engine._client.headers["x-api-key"] == "sk-ant-test"
            assert engine._client.headers["anthropic-version"] == "2023-06-01"
            assert engine._client.headers["Authorization"] == "Bearer sk-ant-test"
        finally:
            await engine.stop()

    async def test_thinking_on_maps_to_native_extended_thinking(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            )

        engine = _anthropic_mock_engine(handler)
        await engine.start()
        await engine.chat(
            messages=[{"role": "user", "content": "hard question"}],
            max_tokens=8000,
            chat_template_kwargs={"enable_thinking": True},
        )
        think = seen.get("thinking")
        assert think and think["type"] == "enabled"
        assert 1024 <= think["budget_tokens"] < seen["max_tokens"]
        # Anthropic requires temperature 1 and no top_p with thinking on
        assert seen["temperature"] == 1.0
        assert "top_p" not in seen
        assert "reasoning" not in seen  # openrouter param must not leak
        await engine.stop()

    async def test_thinking_off_sends_no_thinking_params(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "4"}}], "usage": {}}
            )

        engine = _anthropic_mock_engine(handler)
        await engine.start()
        await engine.chat(
            messages=[{"role": "user", "content": "2+2?"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        assert "thinking" not in seen
        assert "reasoning" not in seen
        await engine.stop()


# ── CLI relay engines (fake claude / codex binaries on PATH) ──────────


import os
import stat as _stat
import sys
import textwrap

from omlx.engine.cli_relay import CLIRelayEngine, cli_available


def _install_fake_cli(tmp_path, monkeypatch, name: str, body: str) -> None:
    """Drop an executable python script named `name` onto PATH."""
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | _stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


_FAKE_CLAUDE = """
import json, os, sys
argv = sys.argv[1:]
args_file = os.environ.get("RELAY_ARGS_FILE")
if args_file:
    with open(args_file, "w") as f:
        json.dump(argv, f)
if "stream-json" in argv:
    events = [
        {"type": "stream_event",
         "event": {"delta": {"type": "thinking_delta", "thinking": "mull"}}},
        {"type": "stream_event",
         "event": {"delta": {"type": "text_delta", "text": "Hel"}}},
        {"type": "stream_event",
         "event": {"delta": {"type": "text_delta", "text": "lo"}}},
        {"type": "result", "is_error": False, "result": "Hello",
         "usage": {"input_tokens": 7, "output_tokens": 2,
                   "cache_read_input_tokens": 3}},
    ]
    for e in events:
        print(json.dumps(e))
else:
    print(json.dumps({
        "type": "result", "is_error": False, "result": "Hola desde claude",
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "cache_read_input_tokens": 4},
    }))
"""

_FAKE_CODEX = """
import json, os, sys
argv = sys.argv[1:]
args_file = os.environ.get("RELAY_ARGS_FILE")
if args_file:
    with open(args_file, "w") as f:
        json.dump(argv, f)
print(json.dumps({"type": "turn.started"}))
print(json.dumps({
    "type": "item.completed",
    "item": {"item_type": "agent_message", "text": "Hola desde codex"},
}))
print(json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 9, "cached_input_tokens": 2, "output_tokens": 6},
}))
"""


class TestCLIRelay:
    def test_cli_available_reports_missing_binary(self, monkeypatch, tmp_path):
        import omlx.engine.cli_relay as cli_relay

        monkeypatch.setenv("PATH", str(tmp_path))  # empty dir: nothing on PATH
        # the real machine may have the CLI in a standard install dir;
        # neutralize the fallback so "missing" is actually missing
        monkeypatch.setattr(cli_relay, "_EXTRA_BIN_DIRS", ())
        ok, detail = cli_available("claude_cli")
        assert ok is False and "claude" in detail

    async def test_claude_chat_parses_result_and_usage(
        self, tmp_path, monkeypatch
    ):
        _install_fake_cli(tmp_path, monkeypatch, "claude", _FAKE_CLAUDE)
        args_file = tmp_path / "args.json"
        monkeypatch.setenv("RELAY_ARGS_FILE", str(args_file))

        engine = CLIRelayEngine(
            model_name="ext.claude.opus",
            provider="claude_cli",
            remote_model="opus",
        )
        await engine.start()
        try:
            out = await engine.chat(
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hola!"},
                ]
            )
        finally:
            await engine.stop()

        assert out.text == "Hola desde claude"
        assert out.prompt_tokens == 14  # input + cache_read
        assert out.completion_tokens == 5
        assert out.cached_tokens == 4

        argv = json.loads(args_file.read_text())
        assert argv[0] == "-p" and argv[1] == "Hola!"
        assert "--model" in argv and argv[argv.index("--model") + 1] == "opus"
        assert "--max-turns" in argv
        assert "--append-system-prompt" in argv

    async def test_claude_default_model_omits_flag(self, tmp_path, monkeypatch):
        _install_fake_cli(tmp_path, monkeypatch, "claude", _FAKE_CLAUDE)
        args_file = tmp_path / "args.json"
        monkeypatch.setenv("RELAY_ARGS_FILE", str(args_file))

        engine = CLIRelayEngine(
            model_name="ext.claude.default",
            provider="claude_cli",
            remote_model="default",
        )
        await engine.start()
        try:
            await engine.chat(messages=[{"role": "user", "content": "hi"}])
        finally:
            await engine.stop()
        argv = json.loads(args_file.read_text())
        assert "--model" not in argv

    async def test_claude_stream_wraps_thinking_and_accumulates(
        self, tmp_path, monkeypatch
    ):
        _install_fake_cli(tmp_path, monkeypatch, "claude", _FAKE_CLAUDE)
        engine = CLIRelayEngine(
            model_name="ext.claude.opus",
            provider="claude_cli",
            remote_model="opus",
        )
        await engine.start()
        try:
            outputs = []
            async for out in engine.stream_chat(
                messages=[{"role": "user", "content": "hi"}]
            ):
                outputs.append(out)
        finally:
            await engine.stop()
        final = outputs[-1]
        assert final.finished is True
        assert final.text == "<think>mull</think>Hello"
        assert final.prompt_tokens == 10  # 7 + cache_read 3
        assert final.completion_tokens == 2

    async def test_codex_chat_parses_events(self, tmp_path, monkeypatch):
        _install_fake_cli(tmp_path, monkeypatch, "codex", _FAKE_CODEX)
        args_file = tmp_path / "args.json"
        monkeypatch.setenv("RELAY_ARGS_FILE", str(args_file))

        engine = CLIRelayEngine(
            model_name="ext.codex.default",
            provider="codex_cli",
            remote_model="default",
        )
        await engine.start()
        try:
            out = await engine.chat(
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hola!"},
                ]
            )
        finally:
            await engine.stop()

        assert out.text == "Hola desde codex"
        assert out.prompt_tokens == 9
        assert out.completion_tokens == 6
        assert out.cached_tokens == 2

        argv = json.loads(args_file.read_text())
        assert argv[0] == "exec" and "--json" in argv
        assert "--sandbox" in argv
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert "-m" not in argv  # default model: no flag
        # instructions are prepended to the prompt (no system flag)
        assert argv[-1].startswith("Be brief.")

    async def test_multi_turn_flattening(self):
        instructions, prompt = CLIRelayEngine._flatten_messages(
            [
                {"role": "system", "content": "Terse."},
                {"role": "user", "content": "One"},
                {"role": "assistant", "content": "Two"},
                {"role": "user", "content": "Three"},
            ]
        )
        assert instructions == "Terse."
        assert "user: One" in prompt and "assistant: Two" in prompt
        assert prompt.endswith("Reply directly to this message: Three")


class TestOpenAIProviderQuirks:
    def _openai_engine(self, handler) -> RemoteOpenAIEngine:
        engine = RemoteOpenAIEngine(
            model_name="ext.openai.gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            remote_model="gpt-5.4-mini",
            api_key="sk-test",
            provider="openai",
        )

        async def _start():
            engine._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                transport=httpx.MockTransport(handler),
            )
            engine._loaded = True

        engine.start = _start
        return engine

    async def test_openai_sends_max_completion_tokens(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            )

        engine = self._openai_engine(handler)
        await engine.start()
        await engine.chat(
            messages=[{"role": "user", "content": "hi"}], max_tokens=128
        )
        assert seen["max_completion_tokens"] == 128
        assert "max_tokens" not in seen
        await engine.stop()

    async def test_unsupported_param_is_dropped_and_retried(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "temperature" in body:
                return httpx.Response(
                    400,
                    json={"error": {"message": (
                        "Unsupported value: 'temperature' does not support "
                        "0.7 with this model. Only the default (1) is "
                        "supported."
                    )}},
                )
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            )

        engine = self._openai_engine(handler)
        await engine.start()
        out = await engine.chat(
            messages=[{"role": "user", "content": "hi"}], temperature=0.7
        )
        assert out.text == "ok"
        assert len(bodies) == 2
        assert "temperature" in bodies[0] and "temperature" not in bodies[1]
        await engine.stop()

    async def test_stream_adapts_max_tokens_rename(self):
        bodies = []
        sse = (
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "max_tokens" in body:
                return httpx.Response(
                    400,
                    json={"error": {"message": (
                        "Unsupported parameter: 'max_tokens' is not "
                        "supported with this model. Use "
                        "'max_completion_tokens' instead."
                    )}},
                )
            return httpx.Response(
                200, content=sse.encode(),
                headers={"Content-Type": "text/event-stream"},
            )

        # openrouter provider so max_tokens survives _sampling_payload and
        # the rename path is exercised end-to-end via the 400 handler
        engine = _mock_engine(handler)
        await engine.start()
        final = None
        async for out in engine.stream_chat(
            messages=[{"role": "user", "content": "hi"}], max_tokens=64
        ):
            final = out
        assert final.text == "ok"
        assert "max_completion_tokens" in bodies[-1]
        await engine.stop()


class TestAnthropicSamplingQuirks:
    async def test_anthropic_drops_top_p_proactively(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            )

        engine = _anthropic_mock_engine(handler)
        await engine.start()
        await engine.chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.9,
        )
        assert "top_p" not in seen
        assert seen["temperature"] == 0.7
        await engine.stop()

    async def test_mutually_exclusive_rejection_drops_top_p_and_retries(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "top_p" in body:
                return httpx.Response(
                    400,
                    json={"error": {"message": (
                        "`temperature` and `top_p` cannot both be specified "
                        "for this model. Please use only one."
                    )}},
                )
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            )

        # openrouter provider: both knobs survive _sampling_payload, so the
        # adaptive-retry path (not the proactive drop) is exercised
        engine = _mock_engine(handler)
        await engine.start()
        out = await engine.chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.9,
        )
        assert out.text == "ok"
        assert len(bodies) == 2
        assert "top_p" in bodies[0] and "top_p" not in bodies[1]
        assert bodies[1]["temperature"] == 0.7
        await engine.stop()


class TestAlwaysStreamingEndpointQuirks:
    """Servers like Apple's `fm serve` stream SSE even for stream=false."""

    SSE_BODY = (
        'data: {"id":"c1","choices":[{"delta":{"role":"assistant"}}],"model":"pcc"}\n\n'
        'data: {"id":"c1","choices":[{"delta":{"content":"PCC"}}],"model":"pcc"}\n\n'
        'data: {"id":"c1","choices":[{"delta":{"content":" works"}}],"model":"pcc"}\n\n'
        'data: {"id":"c1","choices":[{"finish_reason":"stop","delta":{}}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":2},"model":"pcc"}\n\n'
        "data: [DONE]\n\n"
    )

    def _sse_engine(self, body: str, content_type: str) -> RemoteOpenAIEngine:
        engine = RemoteOpenAIEngine(
            model_name="ext.oai.pcc",
            base_url="http://127.0.0.1:8181/v1",
            remote_model="pcc",
            api_key=None,
            provider="openai_compatible",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=body, headers={"content-type": content_type}
            )

        async def _start():
            engine._client = httpx.AsyncClient(
                base_url="http://127.0.0.1:8181/v1",
                transport=httpx.MockTransport(handler),
            )
            engine._loaded = True

        engine.start = _start
        return engine

    async def test_chat_aggregates_sse_response(self):
        engine = self._sse_engine(self.SSE_BODY, "text/event-stream")
        await engine.start()
        out = await engine.chat(messages=[{"role": "user", "content": "hi"}])
        assert out.text == "PCC works"
        assert out.finish_reason == "stop"
        assert out.prompt_tokens == 5
        assert out.completion_tokens == 2
        await engine.stop()

    async def test_chat_aggregates_sse_despite_json_content_type(self):
        engine = self._sse_engine(self.SSE_BODY, "application/json")
        await engine.start()
        out = await engine.chat(messages=[{"role": "user", "content": "hi"}])
        assert out.text == "PCC works"
        await engine.stop()

    def test_aggregate_merges_streamed_tool_calls(self):
        raw = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1",'
            '"function":{"name":"get_weather","arguments":"{\\"ci"}}]}}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"ty\\":\\"Rome\\"}"}}]}}]}\n\n'
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}\n\n'
            "data: [DONE]\n\n"
        )
        body = RemoteOpenAIEngine._aggregate_sse_body(raw)
        call = body["choices"][0]["message"]["tool_calls"][0]
        assert call["id"] == "t1"
        assert call["function"]["name"] == "get_weather"
        assert call["function"]["arguments"] == '{"city":"Rome"}'
        assert body["choices"][0]["finish_reason"] == "tool_calls"


class TestExternalModelMetadata:
    def test_anthropic_defaults_fill_ctx_output_and_vision(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="anthropic",
            base_url="",
            remote_model="claude-sonnet-4-6",
            server_port=8000,
        )
        assert m.context_length == 200_000
        assert m.max_output_tokens == 64_000
        assert m.modality == "text+image"

    def test_openai_defaults_to_vision_modality(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="openai", base_url="", remote_model="gpt-5.4-mini",
            server_port=8000,
        )
        assert m.modality == "text+image"

    def test_catalog_metadata_wins_over_defaults(self, tmp_path):
        reg = ExternalModelRegistry(tmp_path)
        m = reg.add(
            provider="anthropic",
            base_url="",
            remote_model="claude-sonnet-4-6",
            server_port=8000,
            context_length=1_000_000,
            max_output_tokens=32_000,
        )
        assert m.context_length == 1_000_000
        assert m.max_output_tokens == 32_000

    async def test_engine_clamps_max_tokens_to_provider_cap(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            )

        engine = RemoteOpenAIEngine(
            model_name="ext.or.capped",
            base_url="https://mock.example/v1",
            remote_model="vendor/capped",
            api_key="sk-test",
            provider="openrouter",
            max_output_tokens=1000,
        )

        async def _start():
            engine._client = httpx.AsyncClient(
                base_url="https://mock.example/v1",
                transport=httpx.MockTransport(handler),
            )
            engine._loaded = True

        engine.start = _start
        await engine.start()
        await engine.chat(
            messages=[{"role": "user", "content": "hi"}], max_tokens=32_768
        )
        assert seen["max_tokens"] == 1000
        await engine.stop()


class TestCLIResolution:
    def test_binary_found_in_extra_dir_when_not_on_path(
        self, tmp_path, monkeypatch
    ):
        import omlx.engine.cli_relay as cli_relay

        bindir = tmp_path / "landing"
        bindir.mkdir()
        script = bindir / "claude"
        script.write_text("#!/bin/sh\n")
        script.chmod(0o755)

        monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # nothing on PATH
        monkeypatch.setattr(cli_relay, "_EXTRA_BIN_DIRS", (str(bindir),))
        ok, path = cli_relay.cli_available("claude_cli")
        assert ok is True
        assert path == str(script)

    def test_missing_binary_mentions_search_locations(
        self, tmp_path, monkeypatch
    ):
        import omlx.engine.cli_relay as cli_relay

        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(cli_relay, "_EXTRA_BIN_DIRS", ("/nonexistent-dir",))
        ok, detail = cli_relay.cli_available("codex_cli")
        assert ok is False
        assert "codex" in detail and "/nonexistent-dir" in detail
