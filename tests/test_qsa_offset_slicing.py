# SPDX-License-Identifier: Apache-2.0
"""QSAKVCache must cut its indexer arrays to ``offset``, like keys/values.

Audit item 7. ``state`` sliced keys/values to ``offset`` but returned
``index_keys`` / ``index_position_ids`` at full length, and ``extract``
sliced neither. The two arrays grow differently -- keys/values live in a
step-allocated buffer while the indexer arrays are grown by concatenation in
``update_indexer`` -- so a consumer could receive keys describing N tokens
beside indexer state describing more, with nothing marking the difference.

``BatchQSAKVCache.state`` and ``QSAKVCache.to_batch`` already slice, so this
makes the singleton agree with the classes it converts into.

Needs real ``mlx``; tiny tensors, no model.
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
OFFSET, ALLOCATED = 5, 8


def _cache(rows=1, positions_ndim=2, offset=OFFSET, allocated=ALLOCATED):
    """A cache whose buffers are longer than its logical offset."""
    cache = QSAKVCache()
    cache.keys = mx.zeros((rows, HEADS, allocated, DIM))
    cache.values = mx.zeros((rows, HEADS, allocated, DIM))
    cache.offset = offset
    cache.index_keys = mx.ones((rows, allocated, IDX_DIM))
    if positions_ndim == 3:
        cache.index_position_ids = mx.zeros((3, rows, allocated), dtype=mx.int32)
    else:
        cache.index_position_ids = mx.zeros((rows, allocated), dtype=mx.int32)
    return cache


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3], ids=["text", "mrope"])
def test_state_slices_indexer_arrays_to_offset(ndim):
    keys, values, index_keys, positions = _cache(positions_ndim=ndim).state
    mx.eval(index_keys, positions)
    assert keys.shape[2] == OFFSET
    assert values.shape[2] == OFFSET
    assert index_keys.shape[1] == OFFSET, "index_keys must match keys"
    assert positions.shape[-1] == OFFSET, "positions must match keys"


def test_state_handles_absent_keys_without_leaking_full_indexer():
    cache = _cache()
    cache.keys = None
    cache.values = None
    cache.offset = 0
    keys, values, index_keys, positions = cache.state
    mx.eval(index_keys, positions)
    assert keys is None and values is None
    assert index_keys.shape[1] == 0
    assert positions.shape[-1] == 0


def test_state_tolerates_absent_indexer_arrays():
    cache = _cache()
    cache.index_keys = None
    cache.index_position_ids = None
    keys, _, index_keys, positions = cache.state
    assert keys.shape[2] == OFFSET
    assert index_keys is None and positions is None


def test_state_round_trips_through_the_setter():
    """The setter derives offset from keys, so a sliced state must survive."""
    original = _cache()
    restored = QSAKVCache()
    restored.state = original.state
    mx.eval(restored.index_keys)
    assert restored.offset == OFFSET
    assert restored.keys.shape[2] == OFFSET
    assert restored.index_keys.shape[1] == OFFSET


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3], ids=["text", "mrope"])
def test_extract_drops_the_allocated_tail(ndim):
    extracted = _cache(rows=3, positions_ndim=ndim).extract(1)
    mx.eval(extracted.keys, extracted.index_keys, extracted.index_position_ids)
    assert extracted.keys.shape == (1, HEADS, OFFSET, DIM)
    assert extracted.values.shape == (1, HEADS, OFFSET, DIM)
    assert extracted.index_keys.shape == (1, OFFSET, IDX_DIM)
    assert extracted.index_position_ids.shape[-1] == OFFSET
    assert extracted.offset == OFFSET


def test_extract_selects_the_requested_row():
    cache = _cache(rows=3)
    # Make row 1 distinguishable.
    cache.index_keys = mx.concatenate(
        [
            mx.zeros((1, ALLOCATED, IDX_DIM)),
            mx.ones((1, ALLOCATED, IDX_DIM)),
            mx.zeros((1, ALLOCATED, IDX_DIM)),
        ],
        axis=0,
    )
    extracted = cache.extract(1)
    mx.eval(extracted.index_keys)
    assert mx.all(extracted.index_keys == 1).item()


def test_extract_of_mrope_row_keeps_all_three_channels():
    cache = _cache(rows=2, positions_ndim=3)
    cache.index_position_ids = mx.stack(
        [
            mx.full((2, ALLOCATED), value, dtype=mx.int32)
            for value in (10, 20, 30)
        ],
        axis=0,
    )
    extracted = cache.extract(0)
    mx.eval(extracted.index_position_ids)
    assert extracted.index_position_ids.shape == (3, 1, OFFSET)
    for channel, expected in enumerate((10, 20, 30)):
        assert mx.all(extracted.index_position_ids[channel] == expected).item()


def test_extract_without_keys_still_bounds_the_indexer():
    """keys=None leaves offset at 0; the indexer must not come back long."""
    cache = _cache()
    cache.keys = None
    cache.values = None
    cache.offset = 0
    extracted = cache.extract(0)
    mx.eval(extracted.index_keys)
    assert extracted.keys is None
    assert extracted.index_keys.shape[1] == 0
