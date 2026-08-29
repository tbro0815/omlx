# SPDX-License-Identifier: Apache-2.0
"""Qwen4 QSA prefix-cache block handling (text vs MRoPE position channels).

Regression cover for the failure that made every turn of a long Qwen3.8
Flash-Next session re-prefill its whole prompt:

    omlx.cache.prefix_cache - WARNING - Failed to reconstruct cache:
    [concatenate] All the input array dimensions must match exactly except
    for the concatenation axis. However, the provided shapes are
    (1,1,2048), ... (1,3,2048), ... and the concatenation axis is 2.

``_serialize_qsa_positions`` pins the *rank* of stored positions at
``[B, C, S]`` but not ``C``: a block stored while the request used plain
text positions has ``C == 1``, one stored under MRoPE has ``C == 3``. A
sequence spanning both could not be concatenated, so reconstruct raised and
the prefix cache returned ``reused 0`` forever while continuing to write
snapshots that could never be read.

These tests need real ``mlx`` -- they run on Apple Silicon, not in CI
containers. They allocate a handful of tiny tensors and never load a model.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.cache.type_handlers import (  # noqa: E402
    Qwen4BatchQSAKVCacheHandler,
    Qwen4QSAKVCacheHandler,
    Qwen4QSAQuantizedKVCacheHandler,
    _align_qsa_position_channels,
    _deserialize_qsa_positions,
    _serialize_qsa_positions,
)

B, S, HEADS, DIM, IDX_DIM = 1, 4, 2, 8, 6


def _text_positions(value, seq=S):
    """Model-facing text positions: [B, S]."""
    return mx.full((B, seq), value, dtype=mx.int32)


def _mrope_positions(value, seq=S):
    """Model-facing MRoPE positions: [3, B, S], one row per t/h/w."""
    return mx.stack(
        [mx.full((B, seq), value + offset, dtype=mx.int32) for offset in (0, 100, 200)],
        axis=0,
    )


def _block(positions, fill=0.0, seq=S):
    """One serialized cache block, shaped like Qwen4QSAKVCacheHandler emits.

    ``positions`` must already span ``seq`` tokens; a block whose indexer
    length disagrees with its keys is malformed, and building one by accident
    is how the first draft of these tests produced a false failure.
    """
    handler = Qwen4QSAKVCacheHandler()
    assert positions.shape[-1] == seq, "positions must span the same tokens as keys"
    keys = mx.full((B, HEADS, seq, DIM), fill)
    values = mx.full((B, HEADS, seq, DIM), fill)
    index_keys = mx.full((B, seq, IDX_DIM), fill)
    elements = (keys, values, index_keys, _serialize_qsa_positions(positions))
    return {
        **{
            info.name: element
            for info, element in zip(handler.get_state_axis_info(), elements)
        },
        "states": elements,
        "cache_type": handler.cache_type.value,
    }


# --------------------------------------------------------------------------
# _align_qsa_position_channels
# --------------------------------------------------------------------------


def test_serialize_layouts_differ_in_channel_count():
    """The premise: identical rank, different C. Guards the whole fix."""
    assert _serialize_qsa_positions(_text_positions(0)).shape == (B, 1, S)
    assert _serialize_qsa_positions(_mrope_positions(0)).shape == (B, 3, S)


def test_mixed_blocks_are_unconcatenable_without_alignment():
    """Reproduce the raw failure, so the fix is demonstrably load-bearing."""
    blocks = [
        _serialize_qsa_positions(_text_positions(0)),
        _serialize_qsa_positions(_mrope_positions(1)),
    ]
    with pytest.raises(Exception):
        mx.eval(mx.concatenate(blocks, axis=2))


def test_align_widens_text_to_mrope():
    aligned = _align_qsa_position_channels(
        [
            _serialize_qsa_positions(_text_positions(7)),
            _serialize_qsa_positions(_mrope_positions(0)),
        ]
    )
    assert [tuple(a.shape) for a in aligned] == [(B, 3, S), (B, 3, S)]

    joined = mx.concatenate(aligned, axis=2)
    mx.eval(joined)
    assert tuple(joined.shape) == (B, 3, 2 * S)

    # The text position is replicated across t/h/w, matching what
    # _append_indexer_positions does at runtime.
    for channel in range(3):
        assert mx.all(joined[0, channel, :S] == 7).item()

    # MRoPE channels survive intact: 0 / 100 / 200.
    for channel, expected in enumerate((0, 100, 200)):
        assert mx.all(joined[0, channel, S:] == expected).item()


def test_align_is_a_noop_for_uniform_blocks():
    """Uniform input must be returned unchanged, without reallocating."""
    text = [_serialize_qsa_positions(_text_positions(i)) for i in range(3)]
    assert _align_qsa_position_channels(text) is text

    mrope = [_serialize_qsa_positions(_mrope_positions(i)) for i in range(3)]
    assert _align_qsa_position_channels(mrope) is mrope


def test_align_never_narrows_or_invents_channels():
    """A combination with no defined widening is handed back untouched.

    Better a loud concatenate failure than silently fabricated channels.
    """
    two = mx.zeros((B, 2, S), dtype=mx.int32)
    three = _serialize_qsa_positions(_mrope_positions(0))
    out = _align_qsa_position_channels([two, three])
    assert [tuple(a.shape) for a in out] == [(B, 2, S), (B, 3, S)]


@pytest.mark.parametrize(
    "elements",
    [
        [],
        [mx.zeros((B, 1, S), dtype=mx.int32)],
        [mx.zeros((B, 1, S), dtype=mx.int32), None],
        [mx.zeros((B, S), dtype=mx.int32), mx.zeros((B, 1, S), dtype=mx.int32)],
    ],
    ids=["empty", "single", "with-none", "unexpected-rank"],
)
def test_align_passes_degenerate_input_through(elements):
    assert _align_qsa_position_channels(elements) is elements


# --------------------------------------------------------------------------
# concatenate_states -- the path that actually raised
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler_cls",
    [Qwen4QSAKVCacheHandler, Qwen4QSAQuantizedKVCacheHandler],
    ids=["float", "quantized"],
)
def test_concatenate_states_spans_text_and_mrope_blocks(handler_cls):
    """The quantized handler inherits concatenate_states, so cover both."""
    handler = handler_cls()
    blocks = [_block(_text_positions(i), fill=i) for i in range(20)]
    blocks += [_block(_mrope_positions(i), fill=20 + i) for i in range(5)]

    merged = handler.concatenate_states(blocks)
    positions = merged["index_position_ids"]
    mx.eval(positions)

    assert tuple(positions.shape) == (B, 3, 25 * S)
    assert tuple(merged["keys"].shape) == (B, HEADS, 25 * S, DIM)
    assert tuple(merged["index_keys"].shape) == (B, 25 * S, IDX_DIM)


def test_concatenate_states_preserves_all_text_layout():
    """No needless widening when the whole sequence was text."""
    handler = Qwen4QSAKVCacheHandler()
    merged = handler.concatenate_states([_block(_text_positions(i)) for i in range(3)])
    assert tuple(merged["index_position_ids"].shape) == (B, 1, 3 * S)


def test_concatenated_positions_round_trip_to_model_layout():
    """After joining, deserialize must yield the model-facing [3, B, S]."""
    handler = Qwen4QSAKVCacheHandler()
    merged = handler.concatenate_states(
        [_block(_text_positions(1)), _block(_mrope_positions(2))]
    )
    restored = _deserialize_qsa_positions(merged["index_position_ids"])
    mx.eval(restored)
    assert tuple(restored.shape) == (3, B, 2 * S)


def test_text_only_round_trip_returns_two_dimensional_positions():
    handler = Qwen4QSAKVCacheHandler()
    merged = handler.concatenate_states([_block(_text_positions(1))] * 2)
    restored = _deserialize_qsa_positions(merged["index_position_ids"])
    mx.eval(restored)
    assert tuple(restored.shape) == (B, 2 * S)


# --------------------------------------------------------------------------
# slice_state -- per-element clamping
# --------------------------------------------------------------------------


def test_slice_state_clamps_each_element_to_its_own_length():
    """A short index tensor must not be sliced against the keys length.

    QSAKVCache.state returns index tensors unsliced while keys are cut to
    ``offset``, so the two can disagree. MLX clamps an out-of-range slice
    silently, which used to yield a block that concatenated without error
    but no longer lined up with keys.
    """
    handler = Qwen4QSAKVCacheHandler()
    state = _block(_text_positions(0), seq=S)
    # Indexer tensors two tokens shorter than keys.
    short = S - 2
    elements = (
        state["keys"],
        state["values"],
        mx.zeros((B, short, IDX_DIM)),
        mx.zeros((B, 1, short), dtype=mx.int32),
    )
    state = {
        **{
            info.name: element
            for info, element in zip(handler.get_state_axis_info(), elements)
        },
        "states": elements,
        "cache_type": handler.cache_type.value,
    }

    sliced = handler.slice_state(state, 0, S)
    assert sliced is not None
    assert tuple(sliced["keys"].shape) == (B, HEADS, S, DIM)
    assert tuple(sliced["index_keys"].shape) == (B, short, IDX_DIM)
    assert tuple(sliced["index_position_ids"].shape) == (B, 1, short)


def test_slice_state_drops_elements_that_start_past_their_end():
    handler = Qwen4QSAKVCacheHandler()
    elements = (
        mx.zeros((B, HEADS, S, DIM)),
        mx.zeros((B, HEADS, S, DIM)),
        mx.zeros((B, 1, IDX_DIM)),
        mx.zeros((B, 1, 1), dtype=mx.int32),
    )
    state = {
        **{
            info.name: element
            for info, element in zip(handler.get_state_axis_info(), elements)
        },
        "states": elements,
        "cache_type": handler.cache_type.value,
    }
    sliced = handler.slice_state(state, 2, S)
    assert sliced is not None
    assert tuple(sliced["keys"].shape) == (B, HEADS, 2, DIM)
    assert sliced["index_keys"] is None
    assert sliced["index_position_ids"] is None


def test_slice_then_concatenate_round_trips():
    """Slicing into blocks and rejoining must restore the original length."""
    handler = Qwen4QSAKVCacheHandler()
    whole = _block(_mrope_positions(3, seq=8), seq=8)
    first = handler.slice_state(whole, 0, 4)
    second = handler.slice_state(whole, 4, 8)
    merged = handler.concatenate_states([first, second])
    mx.eval(merged["index_position_ids"])
    assert tuple(merged["keys"].shape) == tuple(whole["keys"].shape)
    assert tuple(merged["index_keys"].shape) == tuple(whole["index_keys"].shape)
    assert tuple(merged["index_position_ids"].shape) == (B, 3, 8)


def test_concatenate_drops_index_state_when_a_block_lacks_it():
    """A partial indexer tensor must not be joined against full keys.

    Found by an earlier draft of test_slice_then_concatenate_round_trips,
    which accidentally built a block whose positions were shorter than its
    keys. slice_state correctly emitted None for the out-of-range part, and
    concatenate_states then dropped the None and returned 8 tokens of keys
    beside 4 of positions -- indexer state describing the wrong tokens.
    Absent state is recoverable; misaligned state is not.
    """
    handler = Qwen4QSAKVCacheHandler()
    complete = _block(_mrope_positions(0), fill=1.0)
    partial = dict(complete)
    elements = list(complete["states"])
    elements[2] = None  # index_keys
    elements[3] = None  # index_position_ids
    partial["states"] = tuple(elements)
    partial["index_keys"] = None
    partial["index_position_ids"] = None

    merged = handler.concatenate_states([complete, partial])
    assert tuple(merged["keys"].shape) == (B, HEADS, 2 * S, DIM)
    assert merged["index_keys"] is None
    assert merged["index_position_ids"] is None


def test_concatenate_keeps_index_state_when_every_block_has_it():
    handler = Qwen4QSAKVCacheHandler()
    merged = handler.concatenate_states(
        [_block(_mrope_positions(0)), _block(_mrope_positions(1))]
    )
    assert merged["index_keys"] is not None
    assert tuple(merged["index_position_ids"].shape) == (B, 3, 2 * S)


# --------------------------------------------------------------------------
# The batch handler is deliberately untouched by stage 1
# --------------------------------------------------------------------------


def test_batch_handler_still_returns_last_block_only():
    """Documents current behaviour so a later change is a conscious one."""
    handler = Qwen4BatchQSAKVCacheHandler()
    assert handler.supports_block_slicing is False
    first, second = {"cache_type": "a"}, {"cache_type": "b"}
    assert handler.concatenate_states([first, second]) is second
