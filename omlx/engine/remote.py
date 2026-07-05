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
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3

# OpenAI-style parameter rejections, e.g.:
#   "Unsupported parameter: 'max_tokens' is not supported with this model.
#    Use 'max_completion_tokens' instead."
#   "Unsupported value: 'temperature' does not support 0.7 with this model.
#    Only the default (1) is supported."
_UNSUPPORTED_PARAM_RE = re.compile(
    r"[Uu]nsupported (?:parameter|value): '([\w.]+)'"
)
# Anthropic-style mutual-exclusion rejections, e.g.:
#   "`temperature` and `top_p` cannot both be specified for this model.
#    Please use only one."
_MUTUALLY_EXCLUSIVE_RE = re.compile(
    r"`(\w+)` and `(\w+)` cannot both be specified"
)


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

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **_kwargs,
    ):
        """Naive template for local token *estimates* only.

        The remote provider renders its real chat template server-side;
        this exists so server-side stats/counting paths that reach for
        tokenizer.apply_chat_template don't crash on remote engines.
        """
        parts = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(b.get("text") or "")
                    for b in content
                    if isinstance(b, dict)
                )
            parts.append(f"{m.get('role', 'user')}: {content or ''}")
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\nassistant:"
        return self.encode(text) if tokenize else text


class RemoteOpenAIEngine(BaseEngine):
    """OpenAI-compatible HTTP engine (OpenRouter / generic endpoints)."""

    # Preserve image_url content parts in the server's chat path; vision
    # models behind OpenAI-compatible APIs consume them natively, and
    # text-only requests are unaffected.
    supports_multimodal_fallback = True

    def __init__(
        self,
        model_name: str,
        base_url: str,
        remote_model: str,
        api_key: str | None = None,
        provider: str = "openai_compatible",
        timeout: float = 300.0,
        extra_headers: dict[str, str] | None = None,
        max_output_tokens: int | None = None,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._remote_model = remote_model
        self._api_key = api_key
        self._provider = provider
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
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

    def count_tokens(self, text: str) -> int:
        """Approximate token count (~4 chars/token); real counts come from
        the API usage block per request."""
        return max(1, len(text or "") // 4)

    def count_chat_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        is_partial: bool | None = None,
    ) -> int:
        """Approximate prompt token count for chat messages.

        The provider applies its own chat template server-side, so an exact
        local count is impossible. Estimate from message content lengths
        plus per-message overhead; server paths use this only for stats and
        context-limit pre-checks, and the streamed usage block supplies the
        real numbers afterwards.
        """
        total = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(str(block.get("text") or ""))
            total += 16  # role + template overhead per message (chars)
        if tools:
            total += len(json.dumps(tools))
        return max(1, total // 4)

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
        elif self._provider == "anthropic" and self._api_key:
            # Anthropic's OpenAI-compat layer accepts Authorization: Bearer,
            # but supplying x-api-key too keeps native endpoints (e.g.
            # /v1/models) reachable through the same client.
            headers.setdefault("x-api-key", self._api_key)
            headers.setdefault("anthropic-version", "2023-06-01")
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
        # Clamp to the provider's documented output cap so oversized local
        # defaults (e.g. a global 32k max-tokens setting) don't 400.
        if self._max_output_tokens and max_tokens > self._max_output_tokens:
            max_tokens = self._max_output_tokens
        payload: dict[str, Any] = {
            "model": self._remote_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self._provider == "openai":
            # Current OpenAI models reject max_tokens on chat completions
            # in favor of max_completion_tokens (which older models also
            # accept), so send the new name unconditionally.
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        elif self._provider == "anthropic":
            # Current Claude models reject temperature and top_p supplied
            # together; temperature is the knob oMLX's flows drive, so
            # keep it and drop top_p.
            payload.pop("top_p", None)
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

        # Thinking toggle: local flows pass enable_thinking via
        # chat_template_kwargs (rendered into the template for local
        # models). Remote providers apply their own template, so the flag
        # must travel as an API parameter instead — otherwise reasoning
        # models (e.g. GLM) think even when the benchmark/chat toggle is
        # off. OpenRouter normalizes this across vendors via the unified
        # "reasoning" parameter; None (auto) leaves the provider default.
        ct_kwargs = kwargs.get("chat_template_kwargs") or {}
        enable_thinking = kwargs.get(
            "enable_thinking", ct_kwargs.get("enable_thinking")
        )
        if enable_thinking is not None and self._provider == "openrouter":
            payload["reasoning"] = {"enabled": bool(enable_thinking)}
        elif self._provider == "anthropic":
            # Anthropic's compat layer ignores reasoning_effort but honors
            # the native extended-thinking extra-body param. Budget must be
            # >= 1024 and strictly less than max_tokens, so bump max_tokens
            # when the caller's cap is too tight for thinking to fit.
            if enable_thinking:
                budget = max(1024, min(int(max_tokens * 0.75), 16_000))
                if budget >= max_tokens:
                    payload["max_tokens"] = budget + 1024
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                # Anthropic requires temperature == 1 with thinking enabled.
                payload["temperature"] = 1.0
                payload.pop("top_p", None)
        # Trace outgoing sampling params (no message content) at debug level;
        # invaluable when a provider seems to ignore a knob.
        logger.debug(
            "remote payload keys=%s reasoning=%r enable_thinking=%r "
            "ct_kwargs=%r",
            sorted(k for k in payload if k not in ("messages", "prompt")),
            payload.get("reasoning"),
            enable_thinking,
            ct_kwargs,
        )
        return payload

    def _adapt_payload_for_error(self, payload: dict, detail: str) -> bool:
        """Drop/rename a parameter the provider rejected; True if changed.

        Providers (notably OpenAI) reject requests outright when a model
        doesn't support a sampling parameter. Rather than hardcoding every
        model family's quirks, parse the rejection and retry once without
        the offending knob (or with the suggested replacement name).
        """
        mx = _MUTUALLY_EXCLUSIVE_RE.search(detail or "")
        if mx:
            first, second = mx.group(1), mx.group(2)
            # Prefer dropping the sampling-shape knob over temperature.
            drop = second if second != "temperature" else first
            if drop in payload:
                payload.pop(drop)
                logger.info(
                    f"{self._model_name}: {first!r}/{second!r} are mutually "
                    f"exclusive for {self._remote_model}; dropped {drop!r} "
                    f"and retrying"
                )
                return True
            return False
        m = _UNSUPPORTED_PARAM_RE.search(detail or "")
        if not m:
            return False
        param = m.group(1)
        if param == "max_tokens" and "max_completion_tokens" in detail:
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                logger.info(
                    f"{self._model_name}: provider wants "
                    f"max_completion_tokens; renamed and retrying"
                )
                return True
            return False
        if param in payload:
            payload.pop(param)
            logger.info(
                f"{self._model_name}: provider rejected {param!r} for "
                f"{self._remote_model}; dropped it and retrying"
            )
            return True
        return False

    async def _post_with_retries(self, path: str, payload: dict) -> httpx.Response:
        assert self._client is not None, "engine not started"
        last: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            resp = await self._client.post(path, json=payload)
            if resp.status_code < 400:
                return resp
            last = resp
            if resp.status_code == 400 and self._adapt_payload_for_error(
                payload, self._error_detail(resp)
            ):
                continue
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

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            return (
                body.get("error", {}).get("message")
                if isinstance(body.get("error"), dict)
                else body.get("error")
            ) or ""
        except Exception:  # noqa: BLE001
            return resp.text[:300]

    def _raise_api_error(self, resp: httpx.Response) -> None:
        detail = self._error_detail(resp)
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

    @classmethod
    def _parse_completion_body(cls, resp: httpx.Response) -> Dict[str, Any]:
        """Parse a chat-completions response, tolerating SSE bodies.

        Some OpenAI-compatible servers (e.g. Apple's ``fm serve``) stream
        SSE chunks even when the request did not ask for streaming.
        """
        content_type = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" not in content_type:
            try:
                return resp.json()
            except ValueError:
                if not resp.text.lstrip().startswith("data:"):
                    raise
        return cls._aggregate_sse_body(resp.text)

    @staticmethod
    def _aggregate_sse_body(raw: str) -> Dict[str, Any]:
        """Reassemble a non-streaming completion body from SSE chunks."""
        content: List[str] = []
        reasoning: List[str] = []
        tool_calls: List[dict] = []
        finish_reason: Optional[str] = None
        usage: Optional[dict] = None
        for line in raw.splitlines():
            data = line.strip()
            if not data.startswith("data:"):
                continue
            data = data[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            usage = chunk.get("usage") or usage
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or choice.get("message") or {}
            if delta.get("content"):
                content.append(delta["content"])
            step = delta.get("reasoning") or delta.get("reasoning_content")
            if step:
                reasoning.append(step)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(tool_calls))
                while len(tool_calls) <= idx:
                    tool_calls.append(
                        {"type": "function", "function": {"name": "", "arguments": ""}}
                    )
                slot = tool_calls[idx]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            finish_reason = choice.get("finish_reason") or finish_reason
        message: Dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            message["reasoning"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "choices": [{"message": message, "finish_reason": finish_reason or "stop"}],
            "usage": usage,
        }

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
            body = self._parse_completion_body(resp)
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
        in_think = False  # wrapping provider reasoning deltas in <think> tags

        self._active_requests += 1
        try:
            # Mirror _post_with_retries' parameter adaptation: a 400 for an
            # unsupported sampling param arrives before any SSE data, so it
            # is safe to fix the payload and reopen the stream.
            for _adapt_attempt in range(_MAX_RETRIES):
                probe = await self._client.send(
                    self._client.build_request("POST", path, json=payload),
                    stream=True,
                )
                if probe.status_code == 400:
                    await probe.aread()
                    await probe.aclose()
                    if self._adapt_payload_for_error(
                        payload, self._error_detail(probe)
                    ):
                        continue
                    self._raise_api_error(probe)
                break
            resp = probe
            try:
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
                    content_text = delta.get("content") or choice.get("text") or ""
                    reasoning_text = (
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or ""
                    )
                    # Providers stream reasoning in a separate delta field;
                    # wrap it in <think> tags so oMLX's thinking extraction
                    # renders it like local reasoning models.
                    new_text = ""
                    if reasoning_text:
                        if not in_think:
                            new_text += "<think>"
                            in_think = True
                        new_text += reasoning_text
                    if content_text:
                        if in_think:
                            new_text += "</think>"
                            in_think = False
                        new_text += content_text
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
                await resp.aclose()
        finally:
            self._active_requests -= 1

        if in_think:
            # Stream ended inside a reasoning block (e.g. length-capped).
            accum += "</think>"

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
