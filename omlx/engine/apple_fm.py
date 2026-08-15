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

    # Tell the server's chat path to preserve image_url content parts
    # (otherwise extract_text_content strips them and the model answers
    # blind). Same mechanism DFlash uses for its VLM fallback.
    supports_multimodal_fallback = True

    # Inference runs on this machine's Neural Engine / GPU via the local
    # SDK bridge — no network hop. Benchmarks may treat per-token timing
    # as measured (unlike HTTP engines, whose tokens arrive in network
    # bursts), and results represent this device's real throughput.
    local_inference = True

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
        instructions, prompt, _images = self._flatten_messages(messages)
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
            from ..utils.install import get_venv_pip_command

            raise RuntimeError(
                "apple-fm-sdk is not installed in the oMLX environment. "
                f"Install with: {get_venv_pip_command()} install apple-fm-sdk "
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
    def _image_refs(content: Any) -> list[str]:
        """Extract image URL/data-URL refs from an OpenAI content list."""
        refs: list[str] = []
        if isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in ("image_url", "input_image", "image"):
                    iu = b.get("image_url") or b.get("url") or b.get("image")
                    if isinstance(iu, dict):
                        iu = iu.get("url")
                    if isinstance(iu, str) and iu:
                        refs.append(iu)
        return refs

    @staticmethod
    def _flatten_messages(
        messages: List[Dict[str, Any]],
    ) -> tuple[Optional[str], str, list[str]]:
        """Split messages into (instructions, prompt_text, image_refs).

        System messages become session instructions. Multi-turn history is
        flattened into a role-tagged dialogue; a single user message passes
        through untagged (the common benchmark case).

        Images: only the LAST user message's images are attached (older
        images are represented as "[image]" placeholders in the flattened
        history) — matching the fast-classifier / single-shot vision use
        case without ballooning session state.
        """

        def _text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") in ("image_url", "input_image", "image"):
                        parts.append("[image]")
                    else:
                        t = b.get("text")
                        if t:
                            parts.append(str(t))
                return " ".join(parts)
            return str(content or "")

        system_parts = [
            _text(m.get("content"))
            for m in messages
            if m.get("role") == "system"
        ]
        turns = [m for m in messages if m.get("role") != "system"]

        instructions = "\n\n".join(p for p in system_parts if p) or None

        # Images from the last user turn only.
        image_refs: list[str] = []
        for m in reversed(turns):
            if m.get("role") == "user":
                image_refs = AppleFMEngine._image_refs(m.get("content"))
                break

        if len(turns) == 1 and turns[0].get("role") == "user":
            content = turns[0].get("content")
            if isinstance(content, str):
                prompt = content
            else:
                # keep only the text; images travel as attachments
                prompt = " ".join(
                    str(b.get("text") or "")
                    for b in (content or [])
                    if isinstance(b, dict) and b.get("text")
                )
        else:
            labels = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
            lines = [
                f"{labels.get(m.get('role', 'user'), m.get('role', 'user'))}: "
                f"{_text(m.get('content'))}"
                for m in turns
            ]
            lines.append("Assistant:")
            prompt = "\n".join(lines)
        return instructions, prompt, image_refs

    async def _materialize_images(
        self, image_refs: list[str], temp_paths: list
    ) -> list:
        """Turn image refs into fm.ImageAttachment objects.

        Supports data URLs (base64), http(s) URLs, and existing local file
        paths. Downloaded/decoded images land in temp files that the caller
        removes after the request (tracked via ``temp_paths``).
        """
        import base64
        import re as _re
        import tempfile
        from pathlib import Path as _Path

        attachments = []
        for idx, ref in enumerate(image_refs):
            path: Optional[str] = None
            if ref.startswith("data:"):
                m = _re.match(r"data:image/(\w+);base64,(.*)", ref, _re.DOTALL)
                if not m:
                    raise RuntimeError(
                        "Unsupported image data URL (expected base64 image/*)"
                    )
                suffix = "." + (m.group(1).lower() or "png").replace("jpeg", "jpg")
                data = base64.b64decode(m.group(2))
                tf = tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix="omlx-afm-"
                )
                tf.write(data)
                tf.close()
                path = tf.name
                temp_paths.append(path)
            elif ref.startswith(("http://", "https://")):
                import httpx

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(ref)
                    resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "image/png")
                suffix = "." + ctype.split("/")[-1].split(";")[0].replace(
                    "jpeg", "jpg"
                )
                tf = tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix="omlx-afm-"
                )
                tf.write(resp.content)
                tf.close()
                path = tf.name
                temp_paths.append(path)
            elif ref.startswith("file://"):
                path = ref[7:]
            elif _Path(ref).exists():
                path = ref
            else:
                raise RuntimeError(f"Unsupported image reference: {ref[:80]}")
            attachments.append(
                self._fm.ImageAttachment(_Path(path), label=f"image_{idx + 1}")
            )
        return attachments

    @staticmethod
    def _cleanup_temp(temp_paths: list) -> None:
        import os as _os

        for p in temp_paths:
            try:
                _os.unlink(p)
            except OSError:
                pass

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
        if name in ("ImagePromptError", "PromptError") and (
            "macOS 27" in str(e) or "does not support attachment" in str(e)
        ):
            return "error", (
                "On-device vision is unavailable on this system: image "
                "attachments in Apple's Foundation Models framework require "
                "macOS 27 at runtime (and an SDK build compiled with the "
                "macOS 27 SDK). Text generation is unaffected. "
                f"[{name}: {e}]"
            )
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
        image_refs: Optional[list[str]] = None,
    ) -> GenerationOutput:
        options = self._build_options(max_tokens, temperature)
        prompt_est = self.count_tokens((instructions or "") + prompt)
        temp_paths: list = []
        async with self._request_lock:
            self._active_requests += 1
            try:
                session = self._make_session(instructions)
                if image_refs:
                    attachments = await self._materialize_images(
                        image_refs, temp_paths
                    )
                    fm_prompt: Any = [prompt, *attachments]
                else:
                    fm_prompt = prompt
                text = await session.respond(prompt=fm_prompt, options=options)
                text = str(text)
                finish = "stop"
            except Exception as e:  # noqa: BLE001 - map SDK errors
                finish, text = self._map_error(e)
                if finish == "error":
                    raise RuntimeError(text) from e
            finally:
                self._active_requests -= 1
                self._cleanup_temp(temp_paths)
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
        image_refs: Optional[list[str]] = None,
    ) -> AsyncIterator[GenerationOutput]:
        options = self._build_options(max_tokens, temperature)
        prompt_est = self.count_tokens((instructions or "") + prompt)
        accum = ""
        finish = "stop"
        temp_paths: list = []
        async with self._request_lock:
            self._active_requests += 1
            try:
                session = self._make_session(instructions)
                if image_refs:
                    attachments = await self._materialize_images(
                        image_refs, temp_paths
                    )
                    fm_prompt: Any = [prompt, *attachments]
                else:
                    fm_prompt = prompt
                async for snapshot in session.stream_response(
                    prompt=fm_prompt, options=options
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
                self._cleanup_temp(temp_paths)
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
        instructions, prompt, image_refs = self._flatten_messages(messages)
        return await self._respond_once(
            instructions, prompt, max_tokens, temperature, image_refs
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
        instructions, prompt, image_refs = self._flatten_messages(messages)
        if image_refs:
            # The SDK's streaming bridge hangs (instead of raising) when
            # attachment composition fails; respond() fails cleanly and
            # vision answers are short, so serve image requests
            # non-streaming and emit a single final chunk.
            out = await self._respond_once(
                instructions, prompt, max_tokens, temperature, image_refs
            )
            yield GenerationOutput(
                text=out.text,
                new_text=out.text,
                finished=True,
                finish_reason=out.finish_reason,
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
            )
            return
        async for out in self._stream_once(
            instructions, prompt, max_tokens, temperature, image_refs
        ):
            yield out
