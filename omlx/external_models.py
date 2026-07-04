# SPDX-License-Identifier: Apache-2.0
"""Registry for external (remote API) models.

External models are entries in the EnginePool that are not backed by a local
model directory. They are served by ``omlx.engine.remote.RemoteOpenAIEngine``
via an OpenAI-compatible HTTP API (primary target: OpenRouter; any generic
OpenAI-compatible endpoint works).

Persistence:
- ``{base_path}/external_models.json``      — model records (no secrets)
- ``{base_path}/external_keys.json`` (0600) — API keys, keyed by endpoint

Security notes:
- API keys never live in the main records file so the records can be shared
  or committed without leaking credentials.
- ``validate_not_self_endpoint`` refuses endpoints that resolve back to the
  running oMLX server itself: benchmarking oMLX through oMLX recurses —
  every remote request would spawn another local request.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PROVIDERS = ("openrouter", "openai_compatible", "apple_fm")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_ID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SelfEndpointError(ValueError):
    """Raised when an external endpoint points back at this oMLX server."""


@dataclass
class ExternalModel:
    """One external model record (no secrets)."""

    model_id: str  # oMLX-side id, e.g. "ext.openrouter.deepseek-v4"
    provider: str  # "openrouter" | "openai_compatible"
    base_url: str  # e.g. https://openrouter.ai/api/v1
    remote_model: str  # provider-side model id, e.g. "deepseek/deepseek-v4"
    display_name: str = ""
    # Optional metadata from the provider catalog (context length etc.).
    context_length: int | None = None
    modality: str | None = None  # e.g. "text", "text+image"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExternalModel":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def make_model_id(provider: str, remote_model: str) -> str:
    """Build a filesystem/URL-safe oMLX model id for an external model.

    oMLX model ids double as URL path segments and dict keys, so slashes in
    provider model ids (e.g. "deepseek/deepseek-v4") are flattened.
    """
    flat = _ID_SANITIZE_RE.sub("-", remote_model.replace("/", "-"))
    prefix = {
        "openrouter": "ext.or",
        "apple_fm": "ext.afm",
    }.get(provider, "ext.oai")
    return f"{prefix}.{flat}"


def validate_not_self_endpoint(base_url: str, server_port: int | None) -> None:
    """Reject endpoints that point back at this oMLX server.

    Blocks any URL whose host resolves to a local interface address AND whose
    port matches the running server's port (an explicit product decision:
    wiring oMLX to its own OpenAI endpoint creates request recursion).
    Also rejects obviously malformed URLs.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid endpoint URL: {base_url!r}")

    if server_port is None:
        return

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port != server_port:
        return

    host = parsed.hostname
    candidates: set[str] = set()
    try:
        infos = socket.getaddrinfo(host, None)
        candidates.update(info[4][0] for info in infos)
    except OSError:
        # Unresolvable now — let the connection attempt surface the error.
        return

    local_addrs: set[str] = {"127.0.0.1", "::1", "0.0.0.0", "::"}
    try:
        hostname = socket.gethostname()
        local_addrs.update(
            info[4][0] for info in socket.getaddrinfo(hostname, None)
        )
    except OSError:
        pass

    for addr in candidates:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_loopback or addr in local_addrs:
            raise SelfEndpointError(
                f"Endpoint {base_url!r} resolves to this machine on port "
                f"{port}, which is the running oMLX server. Wiring oMLX to "
                f"its own endpoint is not allowed (request recursion)."
            )


class ExternalModelRegistry:
    """CRUD + persistence for external models and their endpoint API keys."""

    def __init__(self, base_path: Path):
        self._base_path = Path(base_path)
        self._models_file = self._base_path / "external_models.json"
        self._keys_file = self._base_path / "external_keys.json"
        self._models: dict[str, ExternalModel] = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._models_file.exists():
                data = json.loads(self._models_file.read_text())
                for rec in data.get("models", []):
                    try:
                        m = ExternalModel.from_dict(rec)
                        self._models[m.model_id] = m
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Skipping bad external model record: {e}")
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Could not read {self._models_file}: {e}")

    def _save(self) -> None:
        self._base_path.mkdir(parents=True, exist_ok=True)
        payload = {"models": [m.to_dict() for m in self._models.values()]}
        self._models_file.write_text(json.dumps(payload, indent=2))

    def _load_keys(self) -> dict[str, str]:
        try:
            if self._keys_file.exists():
                return json.loads(self._keys_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Could not read {self._keys_file}: {e}")
        return {}

    def _save_keys(self, keys: dict[str, str]) -> None:
        self._base_path.mkdir(parents=True, exist_ok=True)
        # Write then chmod: never leave the file world-readable.
        self._keys_file.write_text(json.dumps(keys, indent=2))
        try:
            os.chmod(self._keys_file, 0o600)
        except OSError as e:
            logger.warning(f"Could not chmod {self._keys_file}: {e}")

    # ── key management (keyed by endpoint, shared across its models) ──

    @staticmethod
    def _key_ref(base_url: str) -> str:
        p = urlparse(base_url)
        return f"{p.scheme}://{p.netloc}"

    def set_api_key(self, base_url: str, api_key: str) -> None:
        keys = self._load_keys()
        keys[self._key_ref(base_url)] = api_key
        self._save_keys(keys)

    def get_api_key(self, base_url: str) -> Optional[str]:
        return self._load_keys().get(self._key_ref(base_url))

    def delete_api_key(self, base_url: str) -> None:
        keys = self._load_keys()
        if keys.pop(self._key_ref(base_url), None) is not None:
            self._save_keys(keys)

    # ── CRUD ──────────────────────────────────────────────────────────

    def list(self) -> list[ExternalModel]:
        return list(self._models.values())

    def get(self, model_id: str) -> Optional[ExternalModel]:
        return self._models.get(model_id)

    def add(
        self,
        provider: str,
        base_url: str,
        remote_model: str,
        display_name: str = "",
        api_key: str | None = None,
        server_port: int | None = None,
        context_length: int | None = None,
        modality: str | None = None,
    ) -> ExternalModel:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; expected {PROVIDERS}")
        if provider == "apple_fm":
            # In-process SDK: no endpoint, no key, nothing to validate.
            base_url = "applefm://local"
            api_key = None
            # AFM 3 is multimodal (image input); default the modality so
            # the chat UI enables image upload (model_type becomes "vlm").
            modality = modality or "text+image"
        else:
            base_url = base_url.rstrip("/")
            validate_not_self_endpoint(base_url, server_port)
        if not remote_model or not remote_model.strip():
            raise ValueError("remote_model must not be empty")

        model = ExternalModel(
            model_id=make_model_id(provider, remote_model),
            provider=provider,
            base_url=base_url,
            remote_model=remote_model.strip(),
            display_name=display_name or remote_model,
            context_length=context_length,
            modality=modality,
        )
        if model.model_id in self._models:
            raise ValueError(f"External model already exists: {model.model_id}")
        self._models[model.model_id] = model
        self._save()
        if api_key:
            self.set_api_key(base_url, api_key)
        logger.info(
            f"Added external model {model.model_id} "
            f"({provider} -> {remote_model} @ {base_url})"
        )
        return model

    def remove(self, model_id: str) -> bool:
        model = self._models.pop(model_id, None)
        if model is None:
            return False
        self._save()
        # Drop the endpoint key only when no other model uses that endpoint.
        if not any(
            self._key_ref(m.base_url) == self._key_ref(model.base_url)
            for m in self._models.values()
        ):
            self.delete_api_key(model.base_url)
        logger.info(f"Removed external model {model_id}")
        return True
