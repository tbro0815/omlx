# SPDX-License-Identifier: Apache-2.0
"""QSAKVCache must never hand out its allocated indexer tail.

Successor to the omni ``test_qsa_offset_slicing.py``, rewritten for upstream
0.6.4's redesign. The omni fix clamped ``index_keys`` / ``index_position_ids``
to ``self.offset`` -- the *KV* offset -- inside ``state`` and ``extract``.
Upstream replaced that with ``_QSAIndexerCache``, where the two arrays live in
a step-allocated buffer and the public attributes are properties already cut
to ``_index_offset``, the indexer's own write cursor. The clamp is now
structural rather than per-call site, so the omni patch is gone.

That is a real semantic change, not just a refactor: the indexer offset is
deliberately decoupled from the KV offset, reconciled on ``trim`` via
``_trim_indexer(self.offset)`` rather than held equal at all times. These
tests pin the parts of the old audit item that still hold under the new
design, plus the reconciliation that replaced the rest -- none of which
upstream's own suite covers.

Needs real ``mlx``; tiny tensors, no model load.
"""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import QSAKVCache  # noqa: E402

HEADS, DIM, IDX_DIM = 2, 8, 6
WRITTEN = 5


def _positions(rows, seq, ndim, start=0):
    """Text positions [rows, seq], or MRoPE [3, rows, seq].

    The three MRoPE channels are offset by 0 / 100 / 200 so a test can tell
    them apart -- a slice that collapses or transposes them shows up as a
    wrong value, not just a wrong shape.
    """
    flat = mx.arange(start, start + seq, dtype=mx.int32)
    if ndim == 3:
        channels = mx.stack([flat + c * 100 for c in range(3)])
        return mx.broadcast_to(channels[:, None, :], (3, rows, seq))
    return mx.broadcast_to(flat[None], (rows, seq))


def _cache(rows=1, positions_ndim=2, written=WRITTEN):
    """A cache whose indexer buffer is far longer than what was written.

    ``update_indexer`` is the only path that grows the buffer while tracking
    the write cursor, so it is what creates the allocated-tail condition the
    old test built by hand. ``index_step`` is 8192, so ``written`` tokens sit
    in an 8192-wide allocation.
    """
    cache = QSAKVCache()
    cache.keys = mx.zeros((rows, HEADS, written, DIM))
    cache.values = mx.zeros((rows, HEADS, written, DIM))
    cache.offset = written
    cache.update_indexer(
        mx.ones((rows, written, IDX_DIM)),
        _positions(rows, written, positions_ndim),
    )
    return cache


# --------------------------------------------------------------------------
# the allocated tail must stay invisible
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3], ids=["text", "mrope"])
def test_indexer_exposes_only_the_written_prefix(ndim):
    cache = _cache(positions_ndim=ndim)
    # The backing allocation really is longer -- otherwise this proves nothing.
    assert cache._index_keys.shape[1] > WRITTEN
    assert cache.index_keys.shape[1] == WRITTEN
    assert cache.index_position_ids.shape[-1] == WRITTEN


@pytest.mark.parametrize("ndim", [2, 3], ids=["text", "mrope"])
def test_state_does_not_leak_the_allocated_tail(ndim):
    keys, values, index_keys, positions = _cache(positions_ndim=ndim).state
    assert keys.shape[2] == WRITTEN
    assert values.shape[2] == WRITTEN
    assert index_keys.shape[1] == WRITTEN
    assert positions.shape[-1] == WRITTEN
    # Every exposed indexer column is written data, never zero padding.
    assert mx.all(index_keys == 1).item()


def test_state_round_trips_through_the_setter():
    cache = _cache()
    restored = QSAKVCache()
    restored.state = cache.state
    assert restored.offset == WRITTEN
    assert restored.index_keys.shape[1] == WRITTEN
    assert restored.index_position_ids.shape[-1] == WRITTEN
    assert mx.all(restored.index_keys == cache.index_keys).item()


def test_state_setter_rejects_misaligned_indexer_arrays():
    """Upstream validates the pair on restore; a short index cannot sneak in."""
    cache = QSAKVCache()
    with pytest.raises(ValueError, match="misaligned"):
        cache.state = (
            mx.zeros((1, HEADS, WRITTEN, DIM)),
            mx.zeros((1, HEADS, WRITTEN, DIM)),
            mx.ones((1, WRITTEN, IDX_DIM)),
            _positions(1, WRITTEN - 1, 2),
        )


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3], ids=["text", "mrope"])
def test_extract_drops_the_allocated_tail(ndim):
    row = _cache(rows=2, positions_ndim=ndim).extract(1)
    assert row.keys.shape[2] == WRITTEN
    assert row.index_keys.shape == (1, WRITTEN, IDX_DIM)
    assert row.index_position_ids.shape[-1] == WRITTEN


def test_extract_of_mrope_row_keeps_all_three_channels():
    """The 3-D branch slices axis 1, not axis 0 -- easy to get backwards."""
    row = _cache(rows=2, positions_ndim=3).extract(1)
    assert row.index_position_ids.shape == (3, 1, WRITTEN)
    # Channels stay distinct (t/h/w were offset by 0 / 100 / 200).
    for channel, base in enumerate((0, 100, 200)):
        assert mx.all(
            row.index_position_ids[channel, 0]
            == mx.arange(WRITTEN, dtype=mx.int32) + base
        ).item()


def test_extract_without_keys_still_bounds_the_indexer():
    """A cache with indexer state but no KV must not hand back the tail."""
    cache = QSAKVCache()
    cache.update_indexer(
        mx.ones((1, WRITTEN, IDX_DIM)), _positions(1, WRITTEN, 2)
    )
    row = cache.extract(0)
    assert row.keys is None
    assert row.offset == 0
    assert row.index_keys.shape[1] == WRITTEN


# --------------------------------------------------------------------------
# trim -- what replaced the omni "clamp to self.offset" invariant
# --------------------------------------------------------------------------


def test_trim_reconciles_the_indexer_with_the_kv_offset():
    """Upstream holds the two offsets equal at trim, not continuously.

    This is the load-bearing half of the omni audit item that survived: after
    a trim the indexer must not still describe tokens the KV no longer has.
    """
    cache = _cache()
    cache.trim(2)
    assert cache.offset == WRITTEN - 2
    assert cache.index_keys.shape[1] == WRITTEN - 2
    assert cache.index_position_ids.shape[-1] == WRITTEN - 2


def test_trim_past_the_start_empties_both_sides():
    cache = _cache()
    cache.trim(WRITTEN + 10)
    assert cache.offset == 0
    assert cache.index_keys.shape[1] == 0
