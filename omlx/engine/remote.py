# SPDX-License-Identifier: Apache-2.0
"""Remote OpenAI-compatible inference engine.

Serves external models (OpenRouter or any generic OpenAI-compatible
endpoint) through the standard BaseEngine interface, so chat, both
benchmark flows, and the OpenAI-compatible server treat them exactly like
local engines.

Design notes:
- ``chat``/``stream_chat`` pass messages straight through — the remote
  provider applies its own chat template.
- ``generate``/``stream_generate`` receive an already-templated prompt from
  local flows. For OpenRouter the raw ``/completions`` endpoint is used
  (benchmark fidelity); generic endpoints may not implement it, so the
  prompt is wrapped as a single user message on 404.
- No scheduler, no KV cache, no memory accounting: remote models occupy no
  local memory (EngineEntry.estimated_size == 0) and provider-side prompt
  cache hits are surfaced via ``GenerationOutput.cached_tokens``.
- Sampling params are mapped best-effort: temperature/top_p/max_tokens/
  stop/presence/frequency penalties are OpenAI-standard; top_k/min_p/
  repetition_penalty are forwarded only for OpenRouter (which supports
  them) and dropped otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3


class _ApproxTokenizer:
    """Minimal tokenizer stand-in for remote models.

    Local token counting is impossible without the provider's tokenizer;
    ~4 chars/token is the conventional estimate. Real counts come from the
    API's ``usage`` block and override these estimates in outputs.

    encode() returns ids that index an internal chunk table, so the
    ``encode -> slice -> decode`` pattern — used by the throughput
    benchmark's prompt synthesizer (admin/benchmark.py::_generate_prompt)
    — reconstructs the corresponding text instead of returning "". Without
    this, remote benchmark runs send empty prompts (OpenRouter answers
    'Input required: specify "prompt"').
    """

    eos_token_id = None
    _CHUNK = 4
    _MAX_CHUNKS = 1_000_000  # ~4 MB of text before the table resets

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def encode(self, text: str, **_kwargs) -> list[int]:
        if len(self._chunks) > self._MAX_CHUNKS:
            self._chunks = []
        base = len(self._chunks)
        pieces = [
            text[i : i + self._CHUNK] for i in range(0, len(text), self._CHUNK)
        ] or [""]
        self._chunks.extend(pieces)
        return list(range(base, base + len(pieces)))

    def decode(self, ids, **_kwargs) -> str:
        out = []
        for i in ids:
            i = int(i)
            if 0 <= i < len(self._chunks):
                out.append(self._chunks[i])
            else:
                out.append(" lorem")  # unknown id: harmless filler
        return "".join(out)


class RemoteOpenAIEngine(BaseEngine):
    """OpenAI-compatible HTTP engine (OpenRouter / generic endpoints)."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        remote_model: str,
        api_key: str | None = None,
        provider: str = "openai_compatible",
        timeout: float = 300.0,
        extra_headers: dict[str, str] | None = None,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._remote_model = remote_model
        self._api_key = api_key
        self._provider = provider
        self._timeout = timeout
        self._extra_headers = dict(extra_headers or {})
        self._client: httpx.AsyncClient | None = None
        self._tokenizer = _ApproxTokenizer()
        self._loaded = False
        self._supports_completions: bool | None = None
        # Activity tracking mirrors local engines so the pool's busy checks
        # and the dashboard behave.
        self._active_requests = 0

    # ── BaseEngine surface ────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        return "remote"

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def remote_model(self) -> str:
        return self._remote_model

    def has_active_requests(self) -> bool:
        return self._active_requests > 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "engine": "remote",
            "provider": self._provider,
            "remote_model": self._remote_model,
            "base_url": self._base_url,
            "active_requests": self._active_requests,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        # No local KV cache; provider-side prompt-cache hits are reported
        # per-request via GenerationOutput.cached_tokens.
        return None

    def _reset_activity_tracking(self) -> None:
        self._active_requests = 0

    async def start(self) -> None:
        if self._loaded:
            return
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._provider == "openrouter":
            # OpenRouter attribution headers (optional but polite).
            headers.setdefault("HTTP-Referer", "https://github.com/jundot/omlx")
            headers.setdefault("X-Title", "oMLX")
        headers.update(self._extra_headers)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout, connect=15.0),
        )
        self._loaded = True
        logger.info(
            f"RemoteOpenAIEngine started: {self._model_name} -> "
            f"{self._remote_model} @ {self._base_url}"
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._loaded = False
        logger.info(f"RemoteOpenAIEngine stopped: {self._model_name}")

    # ── request plumbing ──────────────────────────────────────────────

    def _sampling_payload(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        **kwargs,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._remote_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if presence_penalty:
            payload["presence_penalty"] = presence_penalty
        freq = kwargs.get("frequency_penalty", 0.0)
        if freq:
            payload["frequency_penalty"] = freq
        stop = kwargs.get("stop")
        if stop:
            payload["stop"] = stop
        if self._provider == "openrouter":
            # OpenRouter-specific sampling extensions.
            if top_k:
                payload["top_k"] = top_k
            if min_p:
                payload["min_p"] = min_p
            if repetition_penalty and repetition_penalty != 1.0:
                payload["repetition_penalty"] = repetition_penalty
        return payload

    async def _post_with_retries(self, path: str, payload: dict) -> httpx.Response:
        assert self._client is not None, "engine not started"
        last: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            resp = await self._client.post(path, json=payload)
            if resp.status_code < 400:
                return resp
            last = resp
            if resp.status_code not in _RETRYABLE_STATUS:
                break
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
            logger.warning(
                f"{self._model_name}: HTTP {resp.status_code} from "
                f"{path}, retry {attempt + 1}/{_MAX_RETRIES} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
        assert last is not None
        self._raise_api_error(last)
        raise RuntimeError("unreachable")

    def _raise_api_error(self, resp: httpx.Response) -> None:
        detail = ""
        try:
            body = resp.json()
            detail = (
                body.get("error", {}).get("message")
                if isinstance(body.get("error"), dict)
                else body.get("error")
            ) or ""
        except Exception:  # noqa: BLE001
            detail = resp.text[:300]
        raise RuntimeError(
            f"Remote API error {resp.status_code} for "
            f"{self._remote_model}: {detail}"
        )

    @staticmethod
    def _usage_fields(usage: dict | None) -> tuple[int, int, int]:
        usage = usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
        return prompt, completion, cached

    @staticmethod
    def _map_tool_calls(message: dict) -> Optional[List[Dict[str, Any]]]:
        calls = message.get("tool_calls")
        if not calls:
            return None
        mapped = []
        for c in calls:
            fn = c.get("function", {})
            mapped.append(
                {
                    "id": c.get("id", ""),
                    "type": c.get("type", "function"),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    },
                }
            )
        return mapped

    # ── chat (native passthrough) ─────────────────────────────────────

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> GenerationOutput:
        payload = self._sampling_payload(
            max_tokens, temperature, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, **kwargs,
        )
        payload["messages"] = messages
        if tools:
            payload["tools"] = tools

        self._active_requests += 1
        try:
            resp = await self._post_with_retries("/chat/completions", payload)
            body = resp.json()
        finally:
            self._active_requests -= 1

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        # Some providers put reasoning in a separate field; prepend as
        # thinking markup so oMLX's thinking extraction can pick it up.
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            text = f"<think>{reasoning}</think>{text}"
        prompt_t, completion_t, cached_t = self._usage_fields(body.get("usage"))
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            finish_reason=choice.get("finish_reason") or "stop",
            tool_calls=self._map_tool_calls(message),
            cached_tokens=cached_t,
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        payload = self._sampling_payload(
            max_tokens, temperature, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, **kwargs,
        )
        payload["messages"] = messages
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools

        async for out in self._stream("/chat/completions", payload):
            yield out

    # ── prompt-based generation (benchmarks, /v1/completions) ─────────

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        payload = self._sampling_payload(
            max_tokens, temperature, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, stop=stop, **kwargs,
        )
        self._active_requests += 1
        try:
            if await self._completions_supported():
                payload["prompt"] = prompt
                resp = await self._post_with_retries("/completions", payload)
                body = resp.json()
                choice = (body.get("choices") or [{}])[0]
                text = choice.get("text") or ""
                finish = choice.get("finish_reason") or "stop"
                tool_calls = None
            else:
                payload["messages"] = [{"role": "user", "content": prompt}]
                resp = await self._post_with_retries("/chat/completions", payload)
                body = resp.json()
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                finish = choice.get("finish_reason") or "stop"
                tool_calls = self._map_tool_calls(message)
        finally:
            self._active_requests -= 1

        prompt_t, completion_t, cached_t = self._usage_fields(body.get("usage"))
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            finish_reason=finish,
            tool_calls=tool_calls,
            cached_tokens=cached_t,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        payload = self._sampling_payload(
            max_tokens, temperature, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, stop=stop, **kwargs,
        )
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        if await self._completions_supported():
            payload["prompt"] = prompt
            path = "/completions"
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
            path = "/chat/completions"
        async for out in self._stream(path, payload):
            yield out

    async def _completions_supported(self) -> bool:
        """OpenRouter supports raw /completions; generic endpoints may not.

        Determined once, lazily: assume True for OpenRouter, False for
        generic endpoints (safest default — chat wrapping always works).
        """
        if self._supports_completions is None:
            self._supports_completions = self._provider == "openrouter"
        return self._supports_completions

    # ── SSE streaming core ────────────────────────────────────────────

    async def _stream(
        self, path: str, payload: dict
    ) -> AsyncIterator[GenerationOutput]:
        assert self._client is not None, "engine not started"
        accum = ""
        prompt_t = completion_t = cached_t = 0
        finish: str | None = None
        tool_call_parts: dict[int, dict] = {}

        self._active_requests += 1
        try:
            async with self._client.stream("POST", path, json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_api_error(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        prompt_t, completion_t, cached_t = self._usage_fields(
                            chunk["usage"]
                        )
                    choice = (chunk.get("choices") or [{}])[0]
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    new_text = delta.get("content") or choice.get("text") or ""
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        part = tool_call_parts.setdefault(
                            idx,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tc.get("id"):
                            part["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            part["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            part["function"]["arguments"] += fn["arguments"]
                    if new_text:
                        accum += new_text
                        yield GenerationOutput(
                            text=accum,
                            new_text=new_text,
                            finished=False,
                            finish_reason=None,
                            prompt_tokens=prompt_t,
                            completion_tokens=completion_t,
                            cached_tokens=cached_t,
                        )
        finally:
            self._active_requests -= 1

        yield GenerationOutput(
            text=accum,
            new_text="",
            finished=True,
            finish_reason=finish or "stop",
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            cached_tokens=cached_t,
            tool_calls=(
                [tool_call_parts[i] for i in sorted(tool_call_parts)] or None
            ),
        )
