# SPDX-License-Identifier: Apache-2.0
"""DSpark prefill tap-capture on the mlx-vlm Qwen3.5 inner model.

``_patch_inner_model_capture`` wraps ``Qwen3_5Model.__call__`` and, for a
DSpark host, re-issues the forward with ``capture_layer_ids`` / ``hidden_sink``
so the drafter's context can be primed from prefill.

The subtlety these tests exist for: mlx-vlm's ``LanguageModel.__call__``
forwards ``capture_layer_ids``, ``hidden_sink`` and ``gdn_sink`` *explicitly*
on every call, so on a plain prefill they arrive as keyword arguments whose
value happens to be ``None``. A "is it None?" eligibility check therefore says
yes while the keys are still sitting in ``**kwargs`` — and re-supplying them
raises ``TypeError: got multiple values for keyword argument``, killing the
engine loop mid-generation.

No mlx-vlm import: the patcher only touches ``q35_lang.Qwen3_5Model``, so a
stub class exercises the whole wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_lm_mtp import qwen35_dspark
from omlx.patches.mlx_vlm_mtp import qwen35_vlm_runtime

HIDDEN = 8
TAPS = (1, 3)
N_LAYERS = 5


class _InnerModel:
    """Stands in for ``mlx_vlm.models.qwen3_5.language.Qwen3_5Model``."""

    def __init__(self):
        self.layers = [object()] * N_LAYERS
        self.calls: list[dict] = []

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
        gdn_sink=None,
    ):
        self.calls.append(
            {
                "capture_layer_ids": capture_layer_ids,
                "hidden_sink": hidden_sink,
                "gdn_sink": gdn_sink,
                "position_ids": position_ids,
            }
        )
        if hidden_sink is not None and capture_layer_ids:
            for i in sorted(capture_layer_ids):
                hidden_sink.append(mx.full((1, int(inputs.shape[1]), HIDDEN), i + 1.0))
        return mx.zeros((1, int(inputs.shape[1]), HIDDEN))


def _patched_inner():
    """A freshly patched stub class (the patcher is idempotent per class)."""

    cls = type("Qwen3_5Model", (_InnerModel,), {})
    qwen35_vlm_runtime._patch_inner_model_capture(SimpleNamespace(Qwen3_5Model=cls))
    return cls


class _Host:
    def __init__(self):
        self.args = SimpleNamespace(
            hidden_size=HIDDEN,
            vocab_size=32,
            rms_norm_eps=1e-6,
            max_position_embeddings=1024,
            dspark_block_size=4,
            dspark_target_layer_ids=list(TAPS),
            dspark_markov_rank=4,
            dspark_noise_token_id=3,
            dspark_num_hidden_layers=2,
            dspark_num_attention_heads=2,
            dspark_num_key_value_heads=1,
            dspark_head_dim=4,
            dspark_intermediate_size=16,
            dspark_rms_norm_eps=1e-6,
            dspark_rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            dspark_confidence_head=False,
            dspark_confidence_with_markov=False,
        )
        self._omlx_dspark_decode_enabled = True
        self.captured: list = []

    # The wrapper only needs these two off the host.
    def make_dspark_cache(self):
        return []

    def dspark_append_context(self, taps, cache, *, start_offset=None):
        self.captured.append((taps, start_offset))


def _prefill_kwargs():
    """Exactly what ``LanguageModel.__call__`` forwards on a plain prefill."""
    return {
        "position_ids": None,
        "capture_layer_ids": None,
        "hidden_sink": None,
        "gdn_sink": None,
    }


def test_plain_prefill_with_explicit_none_kwargs_does_not_raise(monkeypatch):
    """Regression: the engine loop died here with 'multiple values for ...'."""
    inner = _patched_inner()()
    host = _Host()
    inner._omlx_mtp_prime_host = lambda: host
    monkeypatch.setattr(qwen35_dspark, "capture_prompt", lambda *a, **k: None)

    out = inner(
        mx.zeros((1, 6), dtype=mx.uint32),
        None,
        None,
        [SimpleNamespace(offset=6)],
        **_prefill_kwargs(),
    )

    assert out.shape == (1, 6, HIDDEN)
    assert len(inner.calls) == 1
    # The wrapper's own capture set replaced the caller's None.
    assert inner.calls[0]["capture_layer_ids"] == sorted(TAPS)
    assert inner.calls[0]["hidden_sink"] is not None
    # gdn_sink must stay None: an ordinary prefill is not a verify forward.
    assert inner.calls[0]["gdn_sink"] is None


def test_taps_are_concatenated_in_config_order(monkeypatch):
    inner = _patched_inner()()
    host = _Host()
    inner._omlx_mtp_prime_host = lambda: host
    seen = {}

    def _capture(h, inputs, taps, cache):
        seen["taps"] = taps

    monkeypatch.setattr(qwen35_dspark, "capture_prompt", _capture)
    inner(
        mx.zeros((1, 2), dtype=mx.uint32),
        None,
        None,
        [SimpleNamespace(offset=2)],
        **_prefill_kwargs(),
    )

    taps = seen["taps"]
    assert taps.shape == (1, 2, HIDDEN * len(TAPS))
    # The stub fills each captured layer with layer_idx + 1.
    for slot, layer in enumerate(TAPS):
        chunk = taps[:, :, slot * HIDDEN : (slot + 1) * HIDDEN]
        assert float(chunk.min()) == layer + 1.0


def test_verify_forward_is_left_alone(monkeypatch):
    """A forward that already asks for captures is the MTP cycle's own."""
    inner = _patched_inner()()
    host = _Host()
    inner._omlx_mtp_prime_host = lambda: host
    called = []
    monkeypatch.setattr(
        qwen35_dspark, "capture_prompt", lambda *a, **k: called.append(1)
    )

    sink: list = []
    inner(
        mx.zeros((1, 3), dtype=mx.uint32),
        None,
        None,
        [SimpleNamespace(offset=3)],
        capture_layer_ids=[4],
        hidden_sink=sink,
        gdn_sink=[],
    )
    assert inner.calls[0]["capture_layer_ids"] == [4]
    assert not called


def test_kill_switch_falls_back_to_no_capture(monkeypatch):
    inner = _patched_inner()()
    host = _Host()
    inner._omlx_mtp_prime_host = lambda: host
    monkeypatch.setenv("OMLX_DSPARK_VLM_PREFILL_CAPTURE", "0")
    called = []
    monkeypatch.setattr(
        qwen35_dspark, "capture_prompt", lambda *a, **k: called.append(1)
    )

    inner(
        mx.zeros((1, 4), dtype=mx.uint32),
        None,
        None,
        [SimpleNamespace(offset=4)],
        **_prefill_kwargs(),
    )
    assert inner.calls[0]["capture_layer_ids"] is None
    assert not called


def test_image_forward_is_skipped(monkeypatch):
    """``inputs_embeds`` means a recursive / multimodal call."""
    inner = _patched_inner()()
    host = _Host()
    inner._omlx_mtp_prime_host = lambda: host
    called = []
    monkeypatch.setattr(
        qwen35_dspark, "capture_prompt", lambda *a, **k: called.append(1)
    )

    inner(
        mx.zeros((1, 2), dtype=mx.uint32),
        mx.zeros((1, 2, HIDDEN)),
        None,
        [SimpleNamespace(offset=2)],
        **_prefill_kwargs(),
    )
    assert inner.calls[0]["capture_layer_ids"] is None
    assert not called
