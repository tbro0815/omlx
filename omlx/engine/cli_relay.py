# SPDX-License-Identifier: Apache-2.0
"""Subscription CLI relay engine (Claude Code / Codex CLI).

Relays inference through the vendor's own locally installed CLI so users
can benchmark and chat with models covered by their Claude Pro/Max or
ChatGPT subscription:

- ``claude_cli`` -> ``claude -p`` (Claude Code print mode)
- ``codex_cli``  -> ``codex exec`` (non-interactive Codex)

Terms-of-service posture (deliberate design constraints):
- Auth stays entirely inside the vendor CLI (OAuth handled by ``claude`` /
  ``codex login``); oMLX never sees, stores, or proxies tokens.
- Only official, documented automation surfaces are used (print/exec
  modes) — no chat-UI scraping, no reverse-engineered endpoints.
- Requests are serialized (concurrency 1): subscription plans are personal
  interactive plans, so oMLX never multiplexes parallel load onto them.

Operational notes:
- The CLIs are agents; we run them as pure text generators: tools are not
  granted, one turn only, and the subprocess cwd is a scratch directory so
  no project context (CLAUDE.md, git state) leaks into prompts.
- Sampling knobs (temperature/top_p/...) are not exposed by the CLIs and
  are silently ignored; benchmarks already report external models with
  ``generation_measured=False`` cosmetics.
- Claude supports incremental streaming via ``--output-format
  stream-json``; Codex emits whole-message events, so codex streams arrive
  as a single final chunk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import BaseEngine, GenerationOutput
from .remote import _ApproxTokenizer

logger = logging.getLogger(__name__)

_CLI_BINARIES = {"claude_cli": "claude", "codex_cli": "codex"}

# Environment variables scrubbed from the child process. These override
# the CLI's own subscription login (e.g. ANTHROPIC_API_KEY makes claude
# bill the API key instead of the claude.ai session — the opposite of
# what a *subscription* relay promises) and can wedge the CLI entirely.
_SCRUBBED_ENV = {
    "claude_cli": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
    ),
    "codex_cli": (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ),
}

# Friendly hints for the most common failure modes.
_LOGIN_HINTS = {
    "claude_cli": 'Run "claude login" (or "claude") in a terminal to sign in.',
    "codex_cli": 'Run "codex login" in a terminal to sign in.',
}


# Server processes launched by launchd / brew services get a minimal PATH
# (/usr/bin:/bin:...), so the user's shell finding the CLI doesn't mean we
# will. Probe the standard install locations too.
_EXTRA_BIN_DIRS = (
    "~/.local/bin",
    "~/.claude/local",
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "~/bin",
    "~/.npm-global/bin",
    "~/.bun/bin",
)


def _resolve_binary(binary: str) -> str | None:
    path = shutil.which(binary)
    if path:
        return path
    import os

    for d in _EXTRA_BIN_DIRS:
        candidate = os.path.join(os.path.expanduser(d), binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def cli_available(provider: str) -> tuple[bool, str]:
    """Locate the relay CLI for a provider (PATH + standard install dirs).

    Returns (True, absolute path) or (False, user-facing explanation).
    """
    binary = _CLI_BINARIES.get(provider)
    if binary is None:
        return False, f"Unknown CLI provider {provider!r}"
    path = _resolve_binary(binary)
    if path is None:
        dirs = ", ".join(_EXTRA_BIN_DIRS)
        return False, (
            f'The "{binary}" CLI was not found on the server\'s PATH or in '
            f"the usual install locations ({dirs}). The server runs with a "
            f"minimal environment, so a CLI visible in your terminal may "
            f"still be missed — symlink it into /usr/local/bin or "
            f"~/.local/bin if it lives somewhere unusual."
        )
    return True, path


class CLIRelayEngine(BaseEngine):
    """Text-generation relay through the claude / codex CLI."""

    # Text only: CLI print modes take a prompt string. Image relay would
    # require file staging per CLI and is intentionally out of scope.
    supports_multimodal_fallback = False

    def __init__(
        self,
        model_name: str,
        provider: str,
        remote_model: str,
        timeout: float = 600.0,
    ):
        if provider not in _CLI_BINARIES:
            raise ValueError(f"Unknown CLI relay provider: {provider!r}")
        self._model_name = model_name
        self._provider = provider
        self._binary = _CLI_BINARIES[provider]
        self._remote_model = remote_model
        self._timeout = timeout
        self._tokenizer = _ApproxTokenizer()
        self._loaded = False
        self._active_requests = 0
        self._scratch_dir: str | None = None
        # Serialize: subscription plans are personal; never run parallel
        # requests against them (also matches bench concurrency=1 guidance).
        self._lock = asyncio.Lock()

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
        return max(1, len(text or "") // 4)

    def count_chat_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        is_partial: bool | None = None,
    ) -> int:
        total = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(str(block.get("text") or ""))
            total += 16
        return max(1, total // 4)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "engine": "cli_relay",
            "provider": self._provider,
            "remote_model": self._remote_model,
            "binary": self._binary,
            "active_requests": self._active_requests,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        return None

    def _reset_activity_tracking(self) -> None:
        self._active_requests = 0

    async def start(self) -> None:
        if self._loaded:
            return
        ok, detail = cli_available(self._provider)
        if not ok:
            raise RuntimeError(detail)
        # Use the resolved absolute path: the bare name may not be on the
        # server process's minimal PATH even though the CLI is installed.
        self._binary = detail
        self._scratch_dir = tempfile.mkdtemp(prefix="omlx-cli-relay-")
        self._loaded = True
        logger.info(
            f"CLIRelayEngine started: {self._model_name} -> "
            f"{self._binary} ({self._remote_model})"
        )

    async def stop(self) -> None:
        if self._scratch_dir:
            shutil.rmtree(self._scratch_dir, ignore_errors=True)
            self._scratch_dir = None
        self._loaded = False
        logger.info(f"CLIRelayEngine stopped: {self._model_name}")

    # ── message flattening ────────────────────────────────────────────

    @staticmethod
    def _flatten_messages(
        messages: List[Dict[str, Any]],
    ) -> tuple[str, str]:
        """Collapse a chat transcript into (instructions, prompt).

        The CLIs accept one prompt string per invocation; multi-turn
        history is rendered as a plain transcript. Image/content-part
        blocks are reduced to their text parts.
        """

        def _text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        t = block.get("text")
                        if t:
                            parts.append(str(t))
                        elif block.get("type") == "image_url":
                            parts.append("[image omitted]")
                return " ".join(parts)
            return str(content or "")

        instructions: list[str] = []
        turns: list[tuple[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            text = _text(m.get("content"))
            if role in ("system", "developer"):
                if text:
                    instructions.append(text)
            else:
                turns.append((role, text))

        if len(turns) <= 1:
            prompt = turns[0][1] if turns else ""
        else:
            transcript = "\n".join(f"{r}: {t}" for r, t in turns[:-1])
            prompt = (
                "Conversation so far:\n"
                f"{transcript}\n\n"
                f"Reply directly to this message: {turns[-1][1]}"
            )
        return "\n".join(instructions), prompt

    # ── subprocess plumbing ───────────────────────────────────────────

    def _build_command(
        self, prompt: str, instructions: str, stream: bool
    ) -> list[str]:
        use_model = self._remote_model not in ("", "default")
        if self._provider == "claude_cli":
            cmd = [
                self._binary,
                "-p",
                prompt,
                "--max-turns",
                "1",
            ]
            if use_model:
                cmd += ["--model", self._remote_model]
            if instructions:
                cmd += ["--append-system-prompt", instructions]
            if stream:
                cmd += [
                    "--output-format",
                    "stream-json",
                    "--verbose",
                    "--include-partial-messages",
                ]
            else:
                cmd += ["--output-format", "json"]
            return cmd
        # codex_cli: instructions are prepended (no system-prompt flag).
        full_prompt = (
            f"{instructions}\n\n{prompt}" if instructions else prompt
        )
        cmd = [self._binary, "exec", "--json"]
        if use_model:
            cmd += ["-m", self._remote_model]
        cmd += [
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            full_prompt,
        ]
        return cmd

    async def _spawn(self, cmd: list[str]) -> asyncio.subprocess.Process:
        # Log the invocation shape (flags only — argv[-1]/-p payloads are
        # prompt content) so hangs are diagnosable from the server log.
        logger.debug(
            "cli_relay spawn: %s",
            " ".join(c if len(c) < 40 else f"<{len(c)} chars>" for c in cmd),
        )
        # stdin MUST be detached: the CLIs probe stdin for piped input and
        # can block forever on an inherited descriptor that never EOFs
        # (observed as a bench stuck in 'starting' under the service
        # environment).
        scrub = _SCRUBBED_ENV.get(self._provider, ())
        env = {k: v for k, v in os.environ.items() if k not in scrub}
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._scratch_dir,
            env=env,
        )

    def _friendly_error(self, returncode: int, stderr: str) -> RuntimeError:
        tail = (stderr or "").strip()[-500:]
        low = tail.lower()
        if any(
            s in low
            for s in ("not logged in", "login", "unauthorized", "authenticate")
        ):
            hint = _LOGIN_HINTS.get(self._provider, "")
            return RuntimeError(
                f"{self._binary} is not authenticated. {hint} [{tail}]"
            )
        return RuntimeError(
            f"{self._binary} exited with code {returncode}: {tail}"
        )

    # ── result parsing ────────────────────────────────────────────────

    @staticmethod
    def _claude_usage(usage: dict | None) -> tuple[int, int, int]:
        usage = usage or {}
        prompt_t = int(usage.get("input_tokens") or 0) + int(
            usage.get("cache_read_input_tokens") or 0
        ) + int(usage.get("cache_creation_input_tokens") or 0)
        completion_t = int(usage.get("output_tokens") or 0)
        cached_t = int(usage.get("cache_read_input_tokens") or 0)
        return prompt_t, completion_t, cached_t

    @staticmethod
    def _codex_usage(usage: dict | None) -> tuple[int, int, int]:
        usage = usage or {}
        prompt_t = int(usage.get("input_tokens") or 0)
        completion_t = int(usage.get("output_tokens") or 0)
        cached_t = int(usage.get("cached_input_tokens") or 0)
        return prompt_t, completion_t, cached_t

    def _parse_codex_events(
        self, lines: list[str]
    ) -> tuple[str, tuple[int, int, int]]:
        """Extract the final agent message + usage from codex --json output.

        Codex's NDJSON event schema has shifted across releases; handle
        both the item-based ("item.completed" with an agent_message item)
        and the msg-based ({"msg": {"type": "agent_message"}}) shapes.
        """
        text = ""
        usage = (0, 0, 0)
        for line in lines:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type") or ""
            item = event.get("item") or {}
            msg = event.get("msg") or {}
            if etype == "item.completed" and (
                item.get("item_type") == "agent_message"
                or item.get("type") == "agent_message"
            ):
                text = item.get("text") or item.get("message") or text
            elif msg.get("type") == "agent_message":
                text = msg.get("message") or msg.get("text") or text
            elif etype in ("turn.completed", "turn_complete"):
                usage = self._codex_usage(event.get("usage") or msg.get("usage"))
            elif msg.get("type") == "token_count":
                info = msg.get("info") or {}
                total = info.get("total_token_usage") or {}
                if total:
                    usage = self._codex_usage(total)
        return text, usage

    # ── generation ────────────────────────────────────────────────────

    async def _run_once(
        self, messages: List[Dict[str, Any]]
    ) -> GenerationOutput:
        instructions, prompt = self._flatten_messages(messages)
        cmd = self._build_command(prompt, instructions, stream=False)
        async with self._lock:
            self._active_requests += 1
            try:
                proc = await self._spawn(cmd)
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        proc.communicate(), timeout=self._timeout
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    raise RuntimeError(
                        f"{self._binary} timed out after {self._timeout:.0f}s"
                    )
            finally:
                self._active_requests -= 1

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise self._friendly_error(proc.returncode, stderr or stdout)

        if self._provider == "claude_cli":
            text = ""
            usage = (0, 0, 0)
            try:
                body = json.loads(stdout)
                if body.get("is_error"):
                    raise RuntimeError(
                        f"claude returned an error result: "
                        f"{str(body.get('result'))[:300]}"
                    )
                text = body.get("result") or ""
                usage = self._claude_usage(body.get("usage"))
            except json.JSONDecodeError:
                text = stdout.strip()
        else:
            text, usage = self._parse_codex_events(stdout.splitlines())
            if not text:
                # Older codex versions print the final message to stdout
                # as plain text after the event stream.
                tail = [
                    ln for ln in stdout.splitlines() if not ln.startswith("{")
                ]
                text = "\n".join(tail).strip()

        prompt_t, completion_t, cached_t = usage
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_t or self.count_chat_tokens(messages),
            completion_tokens=completion_t or self.count_tokens(text),
            finish_reason="stop",
            cached_tokens=cached_t,
        )

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
        return await self._run_once(messages)

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
        if self._provider != "claude_cli":
            # Codex has no partial-delta stream: emit the final result once.
            out = await self._run_once(messages)
            yield GenerationOutput(
                text=out.text,
                new_text=out.text,
                finished=False,
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                cached_tokens=out.cached_tokens,
            )
            yield GenerationOutput(
                text=out.text,
                new_text="",
                finished=True,
                finish_reason=out.finish_reason,
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                cached_tokens=out.cached_tokens,
            )
            return

        instructions, prompt = self._flatten_messages(messages)
        cmd = self._build_command(prompt, instructions, stream=True)
        accum = ""
        usage = (0, 0, 0)
        in_think = False
        async with self._lock:
            self._active_requests += 1
            try:
                proc = await self._spawn(cmd)
                assert proc.stdout is not None
                while True:
                    try:
                        line_b = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=self._timeout
                        )
                    except asyncio.TimeoutError:
                        proc.kill()
                        raise RuntimeError(
                            f"{self._binary} stream timed out after "
                            f"{self._timeout:.0f}s"
                        )
                    if not line_b:
                        break
                    line = line_b.decode("utf-8", errors="replace").strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "stream_event":
                        inner = event.get("event") or {}
                        delta = inner.get("delta") or {}
                        new_text = ""
                        if delta.get("type") == "text_delta":
                            if in_think:
                                new_text += "</think>"
                                in_think = False
                            new_text += delta.get("text") or ""
                        elif delta.get("type") == "thinking_delta":
                            # Surface CLI thinking like other remote
                            # engines: wrapped for the UI's extractor.
                            t = delta.get("thinking") or ""
                            if t:
                                if not in_think:
                                    new_text += "<think>"
                                    in_think = True
                                new_text += t
                        if new_text:
                            accum += new_text
                            yield GenerationOutput(
                                text=accum,
                                new_text=new_text,
                                finished=False,
                            )
                    elif etype == "result":
                        if event.get("is_error"):
                            raise RuntimeError(
                                "claude returned an error result: "
                                f"{str(event.get('result'))[:300]}"
                            )
                        if in_think:
                            accum += "</think>"
                            in_think = False
                        if not accum:
                            accum = event.get("result") or ""
                        usage = self._claude_usage(event.get("usage"))
                rc = await proc.wait()
                if rc != 0:
                    stderr = b""
                    if proc.stderr is not None:
                        stderr = await proc.stderr.read()
                    raise self._friendly_error(
                        rc, stderr.decode("utf-8", errors="replace")
                    )
            finally:
                self._active_requests -= 1

        if in_think:
            accum += "</think>"
        prompt_t, completion_t, cached_t = usage
        yield GenerationOutput(
            text=accum,
            new_text="",
            finished=True,
            finish_reason="stop",
            prompt_tokens=prompt_t or self.count_chat_tokens(messages),
            completion_tokens=completion_t or self.count_tokens(accum),
            cached_tokens=cached_t,
        )

    # ── prompt-based generation (benchmark flows) ─────────────────────

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
        return await self._run_once([{"role": "user", "content": prompt}])

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
        async for out in self.stream_chat(
            [{"role": "user", "content": prompt}]
        ):
            yield out
