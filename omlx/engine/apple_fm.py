# SPDX-License-Identifier: Apache-2.0
"""Apple Foundation Models engine (on-device Apple Intelligence model).

Serves Apple's on-device foundation model through the standard BaseEngine
interface via the official ``apple-fm-sdk`` Python bindings, so chat and
both benchmark flows treat it like any other model.

Verified SDK surface (apple-fm-sdk 0.1.0, macOS 26):
- ``SystemLanguageModel(use_case, guardrails)`` + ``is_available()``
- ``LanguageModelSession(instructions=None, model=None, tools=None)``
- ``await session.respond(prompt, options=GenerationOptions(...)) -> str``
- ``session.stream_response(prompt, options) -> AsyncIterator[str]`` where
  chunks are CUMULATIVE snapshots of the full response so far.
- ``GenerationOptions(sampling: SamplingMode, temperature, maximum_response_tokens)``
- Errors include ConcurrentRequestsError -> requests are serialized here.

Design notes:
- No auth: prerequisites are machine-level (macOS 26+, Xcode 26+ with the
  SDK agreement accepted, Apple Intelligence enabled). ``is_available()``
  reasons are surfaced verbatim.
- Sessions are stateful in the SDK; oMLX passes full history per request,
  so a fresh session is created per request: system messages become the
  session ``instructions``, prior turns are flattened into the prompt.
- Generation runs in-process, so benchmark timing is REAL local-silicon
  timing (model_type is not "remote"; per-token metrics stay measured).
- Cloud variants (AFM 3 Cloud / Cloud Pro) are NOT exposed by the Python
  SDK; the variant hook exists so they can be added if Apple opens them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import BaseEngine, GenerationOutput
from .remote import _ApproxTokenizer

logger = logging.getLogger(__name__)

AFM_VARIANTS = {
    "on-device": "Apple Intelligence (on-device AFM)",
}


class AppleFMEngine(BaseEngine):
    """On-device Apple Foundation Models engine."""

    def __init__(
        self,
        model_name: str,
        variant: str = "on-device",
        permissive_guardrails: bool = False,
    ):
        self._model_name = model_name
        self._variant = variant
        self._permissive_guardrails = permissive_guardrails
        self._fm = None  # apple_fm_sdk module
        self._model = None
        self._tokenizer = _ApproxTokenizer()
        self._loaded = False
        # The on-device model rejects concurrent requests
        # (ConcurrentRequestsError); serialize everything.
        self._request_lock = asyncio.Lock()
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
        # NOT "remote": generation is in-process, so benchmark per-token
        # timing is genuine local-silicon measurement.
        return "apple_fm"

    @property
    def provider(self) -> str:
        return "apple_fm"

    @property
    def remote_model(self) -> str:
        return self._variant

    def has_active_requests(self) -> bool:
        return self._active_requests > 0

    def _reset_activity_tracking(self) -> None:
        self._active_requests = 0

    def count_tokens(self, text: str) -> int:
        return max(1, len(text or "") // 4)

    def count_chat_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        is_partial: bool | None = None,
    ) -> int:
        instructions, prompt = self._flatten_messages(messages)
        return self.count_tokens((instructions or "") + prompt)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "engine": "apple_fm",
            "variant": self._variant,
            "active_requests": self._active_requests,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        return None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._loaded:
            return
        if self._variant not in AFM_VARIANTS:
            raise RuntimeError(
                f"Unknown Apple FM variant {self._variant!r}. The Python "
                f"SDK currently exposes only the on-device model "
                f"(AFM 3 Cloud / Cloud Pro are not developer-accessible)."
            )
        try:
            import apple_fm_sdk as fm
        except ImportError as e:
            raise RuntimeError(
                "apple-fm-sdk is not installed in the oMLX environment. "
                "Install with: "
                "\"$(brew --prefix omlx)/libexec/bin/pip\" install apple-fm-sdk "
                "(requires macOS 26+, Xcode 26+ with the SDK agreement "
                "accepted, and Apple Intelligence enabled)."
            ) from e

        guardrails = (
            fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
            if self._permissive_guardrails
            else fm.SystemLanguageModelGuardrails.DEFAULT
        )
        model = fm.SystemLanguageModel(guardrails=guardrails)
        is_available, reason = model.is_available()
        if not is_available:
            reason_name = getattr(reason, "name", str(reason))
            hints = {
                "APPLE_INTELLIGENCE_NOT_ENABLED": (
                    "Enable it in System Settings -> Apple Intelligence & Siri."
                ),
                "MODEL_NOT_READY": (
                    "The model is still downloading; try again shortly."
                ),
                "DEVICE_NOT_ELIGIBLE": (
                    "This Mac does not support Apple Intelligence."
                ),
            }
            raise RuntimeError(
                f"Apple Foundation Models unavailable: {reason_name}. "
                f"{hints.get(reason_name, '')}"
            )
        self._fm = fm
        self._model = model
        self._loaded = True
        logger.info(
            f"AppleFMEngine started: {self._model_name} "
            f"(variant={self._variant}, guardrails="
            f"{'permissive' if self._permissive_guardrails else 'default'})"
        )

    async def stop(self) -> None:
        self._model = None
        self._fm = None
        self._loaded = False
        logger.info(f"AppleFMEngine stopped: {self._model_name}")

    # ── prompt/options mapping ────────────────────────────────────────

    @staticmethod
    def _flatten_messages(
        messages: List[Dict[str, Any]],
    ) -> tuple[Optional[str], str]:
        """Split messages into (instructions, prompt).

        System messages become session instructions. Multi-turn history is
        flattened into a role-tagged dialogue; a single user message passes
        through untagged (the common benchmark case).
        """

        def _text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(b.get("text") or "")
                    for b in content
                    if isinstance(b, dict)
                )
            return str(content or "")

        system_parts = [
            _text(m.get("content"))
            for m in messages
            if m.get("role") == "system"
        ]
        turns = [m for m in messages if m.get("role") != "system"]

        instructions = "\n\n".join(p for p in system_parts if p) or None

        if len(turns) == 1 and turns[0].get("role") == "user":
            prompt = _text(turns[0].get("content"))
        else:
            labels = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
            lines = [
                f"{labels.get(m.get('role', 'user'), m.get('role', 'user'))}: "
                f"{_text(m.get('content'))}"
                for m in turns
            ]
            lines.append("Assistant:")
            prompt = "\n".join(lines)
        return instructions, prompt

    def _build_options(self, max_tokens: int, temperature: float):
        fm = self._fm
        kwargs: dict[str, Any] = {}
        if max_tokens:
            kwargs["maximum_response_tokens"] = int(max_tokens)
        # Greedy sampling for temperature 0 (benchmark determinism);
        # otherwise pass temperature through.
        if temperature is not None and temperature <= 0:
            sampling = getattr(fm.SamplingMode, "greedy", None)
            try:
                if callable(sampling):
                    sampling = sampling()
                if sampling is not None:
                    kwargs["sampling"] = sampling
            except Exception:  # noqa: BLE001 - sampling stays unset
                pass
        elif temperature is not None:
            kwargs["temperature"] = float(temperature)
        try:
            return fm.GenerationOptions(**kwargs)
        except TypeError:
            # Older/newer SDK without one of the kwargs: retry minimal.
            kwargs.pop("sampling", None)
            return fm.GenerationOptions(
                **{k: v for k, v in kwargs.items() if k != "sampling"}
            )

    def _map_error(self, e: Exception) -> tuple[str, str]:
        """Return (finish_reason, message) for SDK exceptions."""
        fm = self._fm
        name = type(e).__name__
        if fm is not None:
            if isinstance(e, getattr(fm, "GuardrailViolationError", ())) or (
                isinstance(e, getattr(fm, "RefusalError", ()))
            ):
                return "content_filter", f"[Apple FM {name}] {e}"
            if isinstance(e, getattr(fm, "ExceededContextWindowSizeError", ())):
                return "length", f"[Apple FM {name}] {e}"
        return "error", f"[Apple FM {name}] {e}"

    def _make_session(self, instructions: Optional[str]):
        return self._fm.LanguageModelSession(
            instructions=instructions, model=self._model
        )

    # ── generation ────────────────────────────────────────────────────

    async def _respond_once(
        self,
        instructions: Optional[str],
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationOutput:
        options = self._build_options(max_tokens, temperature)
        prompt_est = self.count_tokens((instructions or "") + prompt)
        async with self._request_lock:
            self._active_requests += 1
            try:
                session = self._make_session(instructions)
                text = await session.respond(prompt=prompt, options=options)
                text = str(text)
                finish = "stop"
            except Exception as e:  # noqa: BLE001 - map SDK errors
                finish, text = self._map_error(e)
                if finish == "error":
                    raise RuntimeError(text) from e
            finally:
                self._active_requests -= 1
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_est,
            completion_tokens=self.count_tokens(text),
            finish_reason=finish,
        )

    async def _stream_once(
        self,
        instructions: Optional[str],
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[GenerationOutput]:
        options = self._build_options(max_tokens, temperature)
        prompt_est = self.count_tokens((instructions or "") + prompt)
        accum = ""
        finish = "stop"
        async with self._request_lock:
            self._active_requests += 1
            try:
                session = self._make_session(instructions)
                async for snapshot in session.stream_response(
                    prompt=prompt, options=options
                ):
                    snapshot = str(snapshot)
                    # SDK streams CUMULATIVE snapshots; diff to get deltas.
                    if snapshot.startswith(accum):
                        new_text = snapshot[len(accum):]
                    else:
                        new_text = snapshot  # defensive: treat as delta
                        snapshot = accum + snapshot
                    accum = snapshot
                    if new_text:
                        yield GenerationOutput(
                            text=accum,
                            new_text=new_text,
                            finished=False,
                            finish_reason=None,
                            prompt_tokens=prompt_est,
                            completion_tokens=self.count_tokens(accum),
                        )
            except Exception as e:  # noqa: BLE001 - map SDK errors
                finish, msg = self._map_error(e)
                if finish == "error":
                    raise RuntimeError(msg) from e
                accum = accum or msg
            finally:
                self._active_requests -= 1
        yield GenerationOutput(
            text=accum,
            new_text="",
            finished=True,
            finish_reason=finish,
            prompt_tokens=prompt_est,
            completion_tokens=self.count_tokens(accum),
        )

    # BaseEngine methods -------------------------------------------------

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
        return await self._respond_once(None, prompt, max_tokens, temperature)

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
        async for out in self._stream_once(None, prompt, max_tokens, temperature):
            yield out

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
        instructions, prompt = self._flatten_messages(messages)
        return await self._respond_once(
            instructions, prompt, max_tokens, temperature
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
        instructions, prompt = self._flatten_messages(messages)
        async for out in self._stream_once(
            instructions, prompt, max_tokens, temperature
        ):
            yield out
