# SPDX-License-Identifier: Apache-2.0
"""Warn when the generic batch conversion discards model-owned cache state.

Audit item 9, inverted. ``QSAQuantizedKVCache`` carries indexer keys and
MRoPE positions but defines no ``merge`` / ``to_batch`` / ``extend`` and
inherits none, so ``_patched_make_cache`` rebuilds it as a plain batch
cache -- which has nowhere to store that state and drops it. No error, no
fallback: attention simply runs without the indexer information.

Giving that class a real batch conversion is untested work on a path this
fork cannot exercise (KV quantization is off), and a wrong conversion
degrades output just as quietly. So the fix is to make the loss visible
rather than to guess at the conversion.

This module needs no ``mlx`` -- the warning is pure attribute inspection.
"""

import logging

import pytest

from omlx import scheduler


@pytest.fixture(autouse=True)
def _reset_warned_set():
    """The warning is once-per-class; isolate tests from each other."""
    saved = set(scheduler._auxiliary_drop_warned)
    scheduler._auxiliary_drop_warned.clear()
    yield
    scheduler._auxiliary_drop_warned.clear()
    scheduler._auxiliary_drop_warned.update(saved)


class _WithIndexerState:
    def __init__(self):
        self.index_keys = object()
        self.index_position_ids = object()


class _PositionsOnly:
    def __init__(self):
        self.index_keys = None
        self.index_position_ids = object()


class _PlainCache:
    def __init__(self):
        self.keys = object()
        self.values = object()


def test_warns_when_auxiliary_state_would_be_dropped(caplog):
    with caplog.at_level(logging.WARNING, logger=scheduler.logger.name):
        scheduler._warn_if_auxiliary_state_dropped(_WithIndexerState())
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "_WithIndexerState" in message
    assert "index_keys" in message and "index_position_ids" in message
    assert "to_batch" in message


def test_names_only_the_attributes_actually_present(caplog):
    with caplog.at_level(logging.WARNING, logger=scheduler.logger.name):
        scheduler._warn_if_auxiliary_state_dropped(_PositionsOnly())
    message = caplog.records[0].getMessage()
    assert "index_position_ids" in message
    assert "index_keys" not in message


def test_silent_for_caches_without_auxiliary_state(caplog):
    with caplog.at_level(logging.WARNING, logger=scheduler.logger.name):
        scheduler._warn_if_auxiliary_state_dropped(_PlainCache())
    assert caplog.records == []


def test_warns_once_per_class_not_once_per_layer(caplog):
    """A 60-layer model must not emit 60 identical warnings."""
    with caplog.at_level(logging.WARNING, logger=scheduler.logger.name):
        for _ in range(60):
            scheduler._warn_if_auxiliary_state_dropped(_WithIndexerState())
    assert len(caplog.records) == 1


def test_distinct_classes_each_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=scheduler.logger.name):
        scheduler._warn_if_auxiliary_state_dropped(_WithIndexerState())
        scheduler._warn_if_auxiliary_state_dropped(_PositionsOnly())
    assert len(caplog.records) == 2


def test_is_wired_into_the_conversion_fallback():
    """Guard against the call being dropped in a future merge."""
    import inspect

    source = inspect.getsource(scheduler._patched_make_cache)
    assert "_warn_if_auxiliary_state_dropped" in source
    # It must sit on the fallback path, not before the to_batch check --
    # otherwise every model-owned conversion would warn spuriously.
    warn_at = source.index("_warn_if_auxiliary_state_dropped")
    to_batch_at = source.index("if callable(to_batch)")
    assert to_batch_at < warn_at
