# SPDX-License-Identifier: Apache-2.0
"""BatchQSAKVCache row joins across mixed text/MRoPE position ranks.

Stage 2 of the QSA cache repair. Stage 1 fixed stored prefix-cache blocks
disagreeing on the position *channel* count; this covers the live batching
side, where whole rows disagree on position *rank*:

    text-only request   -> index_position_ids is 2-D [B, S]
    request with images -> index_position_ids is 3-D [3, B, S]

``extend`` (a request joining an in-flight batch) and ``merge`` (building a
batch from singletons) both picked one ``sample_positions``, padded every row
at its own rank, then concatenated on an axis derived from the sample. Two
rows at different ranks therefore joined on the wrong axis or raised.

The fix normalizes inside ``_pad_index`` -- the single point every row passes
through -- promoting 2-D up to 3-D exactly as ``_append_indexer_positions``
does at runtime, and picks the sample as the *widest* operand so there is
always something to promote to.

Needs real ``mlx``; allocates tiny tensors, never loads a model.
"""

import pytest

mx = pytest.importorskip("mlx.core")

pytest.importorskip("mlx_vlm")

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache  # noqa: E402

S, IDX_DIM = 4, 6


def _text_positions(value, rows=1, seq=S):
    """2-D text positions: [B, S]."""
    return mx.full((rows, seq), value, dtype=mx.int32)


def _mrope_positions(value, rows=1, seq=S):
    """3-D MRoPE positions: [3, B, S]."""
    return mx.stack(
        [
            mx.full((rows, seq), value + offset, dtype=mx.int32)
            for offset in (0, 100, 200)
        ],
        axis=0,
    )


def _row(positions, fill=1.0, rows=1, seq=S):
    """A one-row BatchQSAKVCache carrying only indexer state.

    The KV half is left empty on purpose: these tests are about the indexer
    arrays, and BatchKVCache.extend is upstream's own well-covered path.
    """
    cache = BatchQSAKVCache([0] * rows)
    cache.index_keys = mx.full((rows, seq, IDX_DIM), fill)
    cache.index_position_ids = positions
    cache.index_offset = seq
    return cache


# --------------------------------------------------------------------------
# _widest_positions
# --------------------------------------------------------------------------


def test_widest_positions_prefers_mrope():
    text, mrope = _text_positions(0), _mrope_positions(0)
    assert BatchQSAKVCache._widest_positions([text, mrope]) is mrope
    assert BatchQSAKVCache._widest_positions([mrope, text]) is mrope


def test_widest_positions_handles_none_and_empty():
    text = _text_positions(0)
    assert BatchQSAKVCache._widest_positions([None, text]) is text
    assert BatchQSAKVCache._widest_positions([None, None]) is None
    assert BatchQSAKVCache._widest_positions([]) is None


# --------------------------------------------------------------------------
# _pad_index promotion
# --------------------------------------------------------------------------


def test_pad_index_promotes_text_row_to_mrope_sample():
    row = _row(_text_positions(7))
    sample = _mrope_positions(0)
    _, positions = BatchQSAKVCache._pad_index(row, S, row.index_keys, sample)
    mx.eval(positions)
    assert tuple(positions.shape) == (3, 1, S)
    # The single text position is replicated across t/h/w.
    for channel in range(3):
        assert mx.all(positions[channel] == 7).item()


def test_pad_index_leaves_matching_rank_alone():
    row = _row(_mrope_positions(5))
    _, positions = BatchQSAKVCache._pad_index(
        row, S, row.index_keys, _mrope_positions(0)
    )
    mx.eval(positions)
    assert tuple(positions.shape) == (3, 1, S)
    for channel, expected in enumerate((5, 105, 205)):
        assert mx.all(positions[channel] == expected).item()


def test_pad_index_promotes_then_pads_on_the_right_axis():
    """Promotion must happen before padding, or pad picks the 2-D branch."""
    row = _row(_text_positions(2))
    target = S + 3
    _, positions = BatchQSAKVCache._pad_index(
        row, target, row.index_keys, _mrope_positions(0)
    )
    mx.eval(positions)
    assert tuple(positions.shape) == (3, 1, target)
    # Left-padded with zeros, original values pushed right.
    assert mx.all(positions[:, :, :3] == 0).item()
    assert mx.all(positions[:, :, 3:] == 2).item()


def test_pad_index_all_text_stays_two_dimensional():
    row = _row(_text_positions(1))
    _, positions = BatchQSAKVCache._pad_index(
        row, S, row.index_keys, _text_positions(0)
    )
    mx.eval(positions)
    assert tuple(positions.shape) == (1, S)


# --------------------------------------------------------------------------
# extend -- a request joining an in-flight batch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left_positions", "right_positions"),
    [
        (_mrope_positions, _text_positions),
        (_text_positions, _mrope_positions),
    ],
    ids=["mrope-joined-by-text", "text-joined-by-mrope"],
)
def test_extend_across_mixed_ranks(left_positions, right_positions):
    """Either join order must produce one coherent 3-D batch."""
    left = _row(left_positions(1), fill=1.0)
    right = _row(right_positions(2), fill=2.0)
    left.extend(right)
    mx.eval(left.index_position_ids)

    assert tuple(left.index_position_ids.shape) == (3, 2, S)
    assert tuple(left.index_keys.shape) == (2, S, IDX_DIM)
    assert left.index_offset == S


def test_extend_uniform_text_rows_stay_two_dimensional():
    left, right = _row(_text_positions(1)), _row(_text_positions(2))
    left.extend(right)
    mx.eval(left.index_position_ids)
    assert tuple(left.index_position_ids.shape) == (2, S)


def test_extend_uniform_mrope_rows_preserve_channels():
    left, right = _row(_mrope_positions(0)), _row(_mrope_positions(1))
    left.extend(right)
    mx.eval(left.index_position_ids)
    assert tuple(left.index_position_ids.shape) == (3, 2, S)
    for channel, expected in enumerate((0, 100, 200)):
        assert mx.all(left.index_position_ids[channel, 0] == expected).item()
    for channel, expected in enumerate((1, 101, 201)):
        assert mx.all(left.index_position_ids[channel, 1] == expected).item()


def test_extend_pads_shorter_row_to_the_longer_one():
    left = _row(_mrope_positions(1, seq=S), seq=S)
    right = _row(_text_positions(2, seq=S - 2), seq=S - 2)
    left.extend(right)
    mx.eval(left.index_position_ids)
    assert tuple(left.index_position_ids.shape) == (3, 2, S)
    assert left.index_offset == S


def test_extend_rejects_foreign_cache_types():
    with pytest.raises(TypeError):
        _row(_text_positions(0)).extend(object())


# --------------------------------------------------------------------------
# merge -- building a batch from singleton rows
# --------------------------------------------------------------------------


def test_merge_sample_selection_prefers_mrope_regardless_of_order():
    """merge() shares _pad_index, so its risk is only which sample it picks.

    Both are covered directly above; this pins the ordering property that
    merge() depends on -- a text row listed first must not become the sample
    and strand a later MRoPE row at a rank nothing promotes to.

    Note what is NOT covered here: merge() end to end. It calls
    BatchKVCache.merge first, which needs populated KV state, so exercising
    it means building realistic caches rather than the indexer-only rows
    these tests use. The mixed-rank behaviour it relies on is the same
    _pad_index path that test_pad_index_promotes_text_row_to_mrope_sample
    and the extend tests cover.
    """
    text, mrope = _text_positions(1), _mrope_positions(2)
    assert BatchQSAKVCache._widest_positions([text, mrope]) is mrope
    assert BatchQSAKVCache._widest_positions([mrope, text]) is mrope
