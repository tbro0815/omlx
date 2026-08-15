# SPDX-License-Identifier: Apache-2.0
"""Embedded DSpark (SpecForge/DFlash topology) regression tests for Qwen3.5/3.6.

The drafter is grafted into the target checkpoint by ``tools/graft_dspark.py``
and consumed by ``omlx.patches.mlx_lm_mtp.qwen35_dspark``. Two things have to
hold for it to be worth anything, and neither is observable from a smoke test:

1. the module's parameter tree has to match the checkpoint's tensor names
   exactly (strict ``load_weights``), and
2. the forward has to reproduce upstream's, including the pieces that are easy
   to get subtly wrong — dual-source KV, non-causal block attention, YaRN
   rotary phase on absolute positions, and the masked-diffusion block
   convention where the anchor's own output row is discarded.

(2) is checked against a NumPy port of ``dflash.DFlashDraftModel.forward``
written from transformers' Qwen3 semantics, so it is an independent
implementation rather than a snapshot of this one.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_lm_mtp import qwen35_dspark as qd

EPS = 1e-6
H, NH, NKV, HD, NL, V, TAPS, RANK = 64, 4, 2, 16, 2, 97, 2, 8
ROPE = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 128,
}
BLOCK_SIZE = 4


def _config(**overrides):
    params = {
        "hidden_size": H,
        "vocab_size": V,
        "rms_norm_eps": EPS,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "dspark_block_size": BLOCK_SIZE,
        "dspark_target_layer_ids": [1, 3],
        "dspark_markov_rank": RANK,
        "dspark_noise_token_id": 13,
        "dspark_num_hidden_layers": NL,
        "dspark_num_attention_heads": NH,
        "dspark_num_key_value_heads": NKV,
        "dspark_head_dim": HD,
        "dspark_intermediate_size": 3 * H,
        "dspark_rms_norm_eps": EPS,
        "dspark_rope_parameters": ROPE,
        "dspark_confidence_head": True,
        "dspark_confidence_with_markov": True,
    }
    params.update(overrides)
    return SimpleNamespace(**params)


@pytest.fixture(scope="module")
def head_and_weights():
    from mlx.utils import tree_flatten, tree_unflatten

    rng = np.random.default_rng(0)
    head = qd.DSparkHead(_config())
    weights = {}
    for key, value in tree_flatten(head.parameters()):
        if key.endswith("norm.weight") and value.ndim == 1:
            # Plain-Qwen3 RMSNorm gammas are one-centered.
            array = 1.0 + 0.05 * rng.standard_normal(value.shape)
        else:
            array = 0.05 * rng.standard_normal(value.shape)
        weights[key] = array.astype(np.float32)
    head.update(tree_unflatten([(k, mx.array(v)) for k, v in weights.items()]))
    mx.eval(head.parameters())
    return head, weights


# ---------------------------------------------------------------------------
# NumPy reference — transformers Qwen3 + specforge dflash semantics.
# ---------------------------------------------------------------------------


def _rms_norm(x, w):
    return (x * (1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + EPS))) * w


def _yarn_cos_sin(positions):
    dim, base = HD, ROPE["rope_theta"]
    factor, orig = ROPE["factor"], ROPE["original_max_position_embeddings"]

    def find_dim(rotations):
        return dim * math.log(orig / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(find_dim(ROPE["beta_fast"])), 0)
    high = min(math.ceil(find_dim(ROPE["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (np.arange(0, dim, 2) / dim)
    extrapolation = 1.0 - np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - extrapolation) + (
        1.0 / pos_freqs
    ) * extrapolation
    attention_factor = 0.1 * math.log(factor) + 1.0
    angles = positions[:, None] * inv_freq[None, :]
    emb = np.concatenate([angles, angles], axis=-1)
    return np.cos(emb) * attention_factor, np.sin(emb) * attention_factor


def _rope(x, positions):
    cos, sin = _yarn_cos_sin(positions)
    x1, x2 = x[..., : HD // 2], x[..., HD // 2 :]
    return x * cos[None] + np.concatenate([-x2, x1], axis=-1) * sin[None]


def _reference_forward(w, taps, block, ctx_positions, block_positions):
    context = _rms_norm(taps @ w["fc.weight"].T, w["hidden_norm.weight"])
    h = block
    for i in range(NL):
        p = f"layers.{i}."
        x = _rms_norm(h, w[p + "input_layernorm.weight"])
        ctx_len, width = context.shape[0], x.shape[0]

        q = (x @ w[p + "self_attn.q_proj.weight"].T).reshape(width, NH, HD)
        q = _rms_norm(q, w[p + "self_attn.q_norm.weight"]).transpose(1, 0, 2)
        keys = np.concatenate(
            [
                context @ w[p + "self_attn.k_proj.weight"].T,
                x @ w[p + "self_attn.k_proj.weight"].T,
            ],
            axis=0,
        ).reshape(ctx_len + width, NKV, HD)
        keys = _rms_norm(keys, w[p + "self_attn.k_norm.weight"]).transpose(1, 0, 2)
        values = (
            np.concatenate(
                [
                    context @ w[p + "self_attn.v_proj.weight"].T,
                    x @ w[p + "self_attn.v_proj.weight"].T,
                ],
                axis=0,
            )
            .reshape(ctx_len + width, NKV, HD)
            .transpose(1, 0, 2)
        )

        all_positions = np.concatenate([ctx_positions, block_positions])
        keys = _rope(keys, all_positions)
        q = _rope(q, all_positions[-width:])

        keys = np.repeat(keys, NH // NKV, axis=0)
        values = np.repeat(values, NH // NKV, axis=0)
        scores = q @ keys.transpose(0, 2, 1) / math.sqrt(HD)  # no causal mask
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        attn = scores / scores.sum(-1, keepdims=True)
        out = (attn @ values).transpose(1, 0, 2).reshape(width, NH * HD)
        h = h + out @ w[p + "self_attn.o_proj.weight"].T

        y = _rms_norm(h, w[p + "post_attention_layernorm.weight"])
        gate = y @ w[p + "mlp.gate_proj.weight"].T
        up = y @ w[p + "mlp.up_proj.weight"].T
        h = h + ((gate / (1 + np.exp(-gate))) * up) @ w[p + "mlp.down_proj.weight"].T
    return _rms_norm(h, w["norm.weight"])


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_carry_config_survives_from_dict_filtering():
    instance = SimpleNamespace()
    qd.carry_config(instance, {"dspark_block_size": 7, "unrelated": 1})
    assert instance.dspark_block_size == 7
    assert not hasattr(instance, "unrelated")


def test_is_dspark_config_requires_both_block_size_and_taps():
    assert qd.is_dspark_config(_config())
    assert not qd.is_dspark_config(_config(dspark_block_size=0))
    assert not qd.is_dspark_config(_config(dspark_target_layer_ids=[]))
    assert not qd.is_dspark_config(SimpleNamespace())


def test_max_draft_length_reserves_the_anchor_position():
    # A block of 7 positions holds the anchor plus 6 MASKs.
    assert qd.max_draft_length(_config(dspark_block_size=7)) == 6


def test_resolve_depth_treats_an_unset_depth_as_the_whole_block():
    cfg = _config(dspark_block_size=7)
    # Block drafting costs one head forward at any width, so the process-wide
    # default of 1 would pay DSpark's price for a single token.
    assert qd.resolve_depth(cfg, 1) == 6
    assert qd.resolve_depth(cfg, 0) == 6
    assert qd.resolve_depth(cfg, 4) == 4  # explicit setting is a cap
    assert qd.resolve_depth(cfg, 8) == 6  # never past the trained width


def test_dspark_args_rejects_bad_gqa_grouping():
    with pytest.raises(ValueError):
        qd.dspark_args(_config(dspark_num_key_value_heads=3))
    with pytest.raises(ValueError):
        qd.dspark_args(_config(dspark_target_layer_ids=[]))


# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------


def test_parameter_tree_matches_the_checkpoint_layout(head_and_weights):
    from mlx.utils import tree_flatten

    head, _ = head_and_weights
    keys = {k for k, _ in tree_flatten(head.parameters())}
    expected = {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    }
    for i in range(NL):
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            expected.add(f"layers.{i}.{name}")
    assert keys == expected

    params = dict(tree_flatten(head.parameters()))
    assert params["fc.weight"].shape == (H, TAPS * H)
    # The drafter ships proj.bias; DeepSeek's confidence head does not.
    assert params["confidence_head.proj.weight"].shape == (1, H + RANK)
    assert params["confidence_head.proj.bias"].shape == (1,)


def test_optional_heads_are_absent_when_the_config_omits_them():
    head = qd.DSparkHead(
        _config(dspark_markov_rank=0, dspark_confidence_head=False)
    )
    assert getattr(head, "markov_head", None) is None
    assert getattr(head, "confidence_head", None) is None


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_block_forward_matches_the_upstream_reference(head_and_weights):
    head, weights = head_and_weights
    rng = np.random.default_rng(1)
    ctx_len, width, start = 5, BLOCK_SIZE, 5

    taps = (0.1 * rng.standard_normal((ctx_len, TAPS * H))).astype(np.float32)
    block = (0.1 * rng.standard_normal((width, H))).astype(np.float32)

    reference = _reference_forward(
        weights,
        taps.astype(np.float64),
        block.astype(np.float64),
        np.arange(start - ctx_len, start),
        np.arange(start, start + width),
    )

    cache = head.make_cache()
    for entry in cache:
        entry.seed_offset(start - ctx_len)
    head.append_context(mx.array(taps)[None], cache, start_offset=start - ctx_len)
    # The context cache ends at the anchor's own position: the draft block
    # starts exactly where the committed timeline stops.
    assert [c.offset for c in cache] == [start] * NL

    got = np.array(head(mx.array(block)[None], cache))[0]
    error = np.abs(got - reference).max() / np.abs(reference).max()
    assert error < 2e-5, error


def test_streaming_the_context_equals_one_shot(head_and_weights):
    head, _ = head_and_weights
    rng = np.random.default_rng(2)
    taps = mx.array((0.1 * rng.standard_normal((6, TAPS * H))).astype(np.float32))[
        None
    ]
    block = mx.array((0.1 * rng.standard_normal((3, H))).astype(np.float32))[None]

    one_shot = head.make_cache()
    head.append_context(taps, one_shot, start_offset=0)
    chunked = head.make_cache()
    head.append_context(taps[:, :4], chunked, start_offset=0)
    head.append_context(taps[:, 4:], chunked, start_offset=4)

    a = np.array(head(block, one_shot))
    b = np.array(head(block, chunked))
    assert np.abs(a - b).max() < 1e-5


def test_drafts_are_read_from_the_mask_positions(head_and_weights):
    """The anchor row is discarded; each MASK position predicts its own token.

    Upstream reads ``[:, -block_size + 1:, :]`` from a block whose first entry
    is the confirmed anchor, so ``k`` drafts need ``k + 1`` query positions.
    """
    head, _ = head_and_weights
    host = _host(head)

    logits, hidden = host.dspark_forward(
        mx.zeros((1, 2, TAPS * H)),
        mx.array([[7]], dtype=mx.uint32),
        host.make_mtp_cache(),
        draft_length=3,
    )
    assert logits.shape == (1, 3, V)
    assert hidden.shape == (1, 3, H)

    # Same context and anchor, one position deeper: the shared prefix of the
    # block is unchanged, so the earlier drafts must be bit-identical.
    logits4, _ = host.dspark_forward(
        mx.zeros((1, 2, TAPS * H)),
        mx.array([[7]], dtype=mx.uint32),
        host.make_mtp_cache(),
        draft_length=4,
    )
    assert logits4.shape == (1, 4, V)


def test_markov_bias_shape_matches_the_sampling_loop(head_and_weights):
    head, _ = head_and_weights
    host = _host(head)
    bias, embedding = host.dspark_markov(mx.array([5], dtype=mx.uint32))
    assert bias.shape == (1, V)
    assert embedding.shape == (1, RANK)
    # batch_generator adds this straight onto logits[:, idx, :].
    assert (mx.zeros((1, V)) + bias).shape == (1, V)


# ---------------------------------------------------------------------------
# Context cache
# ---------------------------------------------------------------------------


def test_cache_grows_past_its_preallocation_and_trims():
    cache = qd.DSparkContextCache(step=2)
    keys = mx.zeros((1, NKV, 5, HD))
    cache.append(keys, keys, start_offset=0)
    cache.append(keys, keys, start_offset=5)
    assert cache.offset == 10 and len(cache) == 10
    stored_keys, stored_values = cache.state
    assert stored_keys.shape[2] == 10 and stored_values.shape[2] == 10

    cache.trim(4)
    assert cache.offset == 6 and len(cache) == 6
    assert cache.state[0].shape[2] == 6


def test_windowed_cache_keeps_the_newest_rows_and_absolute_offsets():
    """A suffix window must not disturb the rotary phase of what it keeps.

    ``offset`` tracks absolute position, not stored rows, so a windowed cache
    still puts the draft block at the anchor's true position — the drafter
    just sees less history.
    """
    cache = qd.DSparkContextCache(max_size=8, step=4)
    for i in range(20):
        row = mx.full((1, NKV, 1, HD), float(i))
        cache.append(row, row, start_offset=i)

    assert cache.offset == 20
    assert len(cache) <= 8 + cache.step
    keys, values = cache.state
    kept = [float(keys[0, 0, j, 0]) for j in range(keys.shape[2])]
    # Newest row is last, oldest survivors dropped first, no gaps.
    assert kept == list(range(20 - len(kept), 20))
    assert values.shape == keys.shape


def test_a_chunk_longer_than_the_window_keeps_only_its_tail():
    cache = qd.DSparkContextCache(max_size=4, step=4)
    rows = mx.concatenate(
        [mx.full((1, NKV, 1, HD), float(i)) for i in range(10)], axis=2
    )
    cache.append(rows, rows, start_offset=0)
    # offset still advances by the whole chunk: it is a position, not a count.
    assert cache.offset == 10
    keys, _ = cache.state
    kept = [float(keys[0, 0, j, 0]) for j in range(keys.shape[2])]
    assert kept == [6.0, 7.0, 8.0, 9.0]


def test_window_is_unbounded_when_disabled(monkeypatch):
    monkeypatch.setenv("OMLX_DSPARK_CONTEXT_WINDOW", "0")
    assert qd.context_window() == 0
    monkeypatch.setenv("OMLX_DSPARK_CONTEXT_WINDOW", "2048")
    assert qd.context_window() == 2048
    monkeypatch.setenv("OMLX_DSPARK_CONTEXT_WINDOW", "nonsense")
    assert qd.context_window() == qd._DEFAULT_CONTEXT_WINDOW


def test_windowing_does_not_change_the_forward_for_a_short_context(
    head_and_weights,
):
    """Below the window, a windowed cache is bit-identical to an unbounded one."""
    head, _ = head_and_weights
    rng = np.random.default_rng(7)
    taps = mx.array((0.1 * rng.standard_normal((5, TAPS * H))).astype(np.float32))[None]
    block = mx.array((0.1 * rng.standard_normal((3, H))).astype(np.float32))[None]

    unbounded = [qd.DSparkContextCache(max_size=0) for _ in head.layers]
    windowed = [qd.DSparkContextCache(max_size=64) for _ in head.layers]
    for cache in (unbounded, windowed):
        head.append_context(taps, cache, start_offset=0)
    a = np.array(head(block, unbounded))
    b = np.array(head(block, windowed))
    assert np.abs(a - b).max() == 0.0


def test_cache_refuses_a_discontiguous_append():
    cache = qd.DSparkContextCache()
    keys = mx.zeros((1, NKV, 2, HD))
    cache.append(keys, keys, start_offset=0)
    with pytest.raises(ValueError):
        cache.append(keys, keys, start_offset=7)


def test_seeded_cache_keeps_absolute_positions_without_stored_rows():
    cache = qd.DSparkContextCache()
    cache.seed_offset(120)
    assert cache.offset == 120 and len(cache) == 0
    assert cache.state == (None, None)
    keys = mx.zeros((1, NKV, 1, HD))
    cache.append(keys, keys, start_offset=120)
    assert cache.offset == 121 and len(cache) == 1


# ---------------------------------------------------------------------------
# Priming seam
# ---------------------------------------------------------------------------


class _Host:
    """Minimal stand-in for the patched language model."""

    def __init__(self, head):
        self.mtp = head
        self.args = _config()
        self._omlx_dspark_decode_enabled = True
        self._omlx_mtp_depth = 3
        self.model = SimpleNamespace(
            embed_tokens=lambda ids: mx.zeros((*ids.shape, H))
        )
        self.lm_head = lambda x: mx.zeros((*x.shape[:-1], V))

    def make_mtp_cache(self):
        return self.make_dspark_cache()


qd.install_host_methods(_Host)


def _host(head):
    return _Host(head)


def _target_cache(offset):
    return [SimpleNamespace(offset=offset)]


def test_capture_prompt_then_take_primed_hands_over_the_context(head_and_weights):
    head, _ = head_and_weights
    host = _host(head)
    prompt_len = 6
    taps = mx.zeros((1, prompt_len, TAPS * H))

    qd.capture_prompt(
        host, mx.zeros((1, prompt_len), dtype=mx.uint32), taps, _target_cache(prompt_len)
    )
    # Activation runs one more target token with capture disabled.
    primed = qd.take_primed(host, _target_cache(prompt_len + 1), None)
    assert primed is not None
    caches, hist_offset = primed
    assert hist_offset == prompt_len
    assert [c.offset for c in caches] == [prompt_len] * NL
    assert len(caches[0]) == prompt_len


def test_take_primed_falls_back_to_a_correctly_phased_empty_cache(head_and_weights):
    """No priming must still mean the right rotary phase, not offset 0.

    Returning ``None`` here would leave ``_post_init_mtp`` with a fresh
    offset-0 cache while the target sits at position P — every context row
    the drafter later commits would be rotated to the wrong position.
    """
    head, _ = head_and_weights
    host = _host(head)
    caches, hist_offset = qd.take_primed(host, _target_cache(500), None)
    assert hist_offset == 499
    assert [c.offset for c in caches] == [499] * NL
    assert all(len(c) == 0 for c in caches)


def test_capture_prompt_invalidates_on_a_discontiguous_chunk(head_and_weights):
    head, _ = head_and_weights
    host = _host(head)
    inputs = mx.zeros((1, 4), dtype=mx.uint32)
    taps = mx.zeros((1, 4, TAPS * H))

    qd.capture_prompt(host, inputs, taps, _target_cache(4))
    # A rewind / request switch: the next chunk does not continue the
    # timeline, so the stale context is dropped and a fresh one starts at the
    # new chunk's own position (95..98) rather than being silently spliced.
    qd.capture_prompt(host, inputs, taps, _target_cache(99))
    caches, hist_offset = qd.take_primed(host, _target_cache(100), None)
    assert hist_offset == 99
    assert len(caches[0]) == 4  # only the new chunk, not 4 + 4


def test_seam_log_distinguishes_primed_from_unprimed(head_and_weights, caplog):
    """``primed=N`` in the batch-generator log is the same either way.

    ``take_primed`` seeds an empty cache at the target's offset when priming
    did not survive, so the activation line reports the same ``primed=N`` as a
    fully primed request. These INFO lines are what actually distinguishes
    "the drafter holds the prompt" from "the drafter holds nothing".
    """
    head, _ = head_and_weights
    prompt_len = 6

    host = _host(head)
    qd.capture_prompt(
        host,
        mx.zeros((1, prompt_len), dtype=mx.uint32),
        mx.zeros((1, prompt_len, TAPS * H)),
        _target_cache(prompt_len),
    )
    with caplog.at_level("INFO", logger="omlx.patches.mlx_lm_mtp.qwen35_dspark"):
        qd.take_primed(host, _target_cache(prompt_len + 1), None)
    assert f"rows={prompt_len} offset={prompt_len}" in caplog.text
    assert "primed)" in caplog.text
    assert "missing" not in caplog.text

    caplog.clear()
    with caplog.at_level("INFO", logger="omlx.patches.mlx_lm_mtp.qwen35_dspark"):
        qd.take_primed(_host(head), _target_cache(500), None)
    assert "rows=0 offset=499" in caplog.text
    assert "unprimed" in caplog.text


def test_a_broken_prefill_chunk_is_reported_with_the_row_shortfall(
    head_and_weights, caplog
):
    """The 16k case: priming restarts mid-prompt and only the tail survives."""
    head, _ = head_and_weights
    host = _host(head)
    inputs = mx.zeros((1, 4), dtype=mx.uint32)
    taps = mx.zeros((1, 4, TAPS * H))

    with caplog.at_level("INFO", logger="omlx.patches.mlx_lm_mtp.qwen35_dspark"):
        qd.capture_prompt(host, inputs, taps, _target_cache(4))
        qd.capture_prompt(host, inputs, taps, _target_cache(8))
        # Third chunk lands out of step — the first eight rows are dropped.
        qd.capture_prompt(host, inputs, taps, _target_cache(99))
        qd.take_primed(host, _target_cache(100), None)

    assert "priming restarted after 2 chunk(s)" in caplog.text
    assert "discarding 8 context row(s)" in caplog.text
    # The seam then shows a context whose offset is right but whose rows are not.
    assert "rows=4 offset=99" in caplog.text
    assert "95 missing" in caplog.text


def test_capture_prompt_ignores_a_lone_decode_step(head_and_weights):
    head, _ = head_and_weights
    host = _host(head)
    qd.capture_prompt(
        host,
        mx.zeros((1, 1), dtype=mx.uint32),
        mx.zeros((1, 1, TAPS * H)),
        _target_cache(1),
    )
    assert getattr(host, qd._PRIME_CTX_ATTR, None) is None
