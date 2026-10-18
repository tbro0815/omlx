"""Arrays jang_tools memoizes at module level must be evaluated at load.

MLX streams are registered per thread. A lazy array created on the loader
thread stays bound to that thread's stream until it is evaluated; evaluating
it from the per-engine inference thread raises "There is no Stream(gpu, N) in
current thread". jang_tools.dsv4.mlx_model._Q_NORM_WEIGHT_CACHE is filled
with lazy mx.ones by the JANGTQ warmup on the loader thread, which broke the
first prefill of DeepSeek-V4-Flash-JANGTQ2.

Whether a given process reproduces the cross-thread error depends on MLX's
stream setup, so these tests pin the contract instead: every array held in
jang_tools globals reaches mx.eval, and nothing else does.
"""

import sys
import types

import mlx.core as mx
import pytest

from omlx.engine.jang import _materialize_jang_tools_globals


@pytest.fixture
def evaluated(monkeypatch):
    ids: set[int] = set()

    def fake_eval(*args):
        for arg in args:
            for a in arg if isinstance(arg, (list, tuple)) else [arg]:
                ids.add(id(a))

    monkeypatch.setattr(mx, "eval", fake_eval)
    return ids


def _module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_evaluates_arrays_in_jang_tools_globals(monkeypatch, evaluated):
    ones = mx.ones((8,), dtype=mx.bfloat16)
    eps = mx.ones((1,))
    nested = mx.zeros((2,))
    _module(
        monkeypatch,
        "jang_tools._omni_stream_test",
        _Q_NORM_WEIGHT_CACHE={(8, mx.bfloat16): ones},
        _hc_eps_array_cache=eps,
        _TABLES=[(nested,)],
    )

    _materialize_jang_tools_globals()

    assert {id(ones), id(eps), id(nested)} <= evaluated


def test_skips_arrays_that_cannot_be_evaluated(monkeypatch):
    # e.g. a placeholder left in a global by mx.compile tracing.
    good = mx.ones((4,))
    bad = mx.zeros((3,))
    _module(monkeypatch, "jang_tools._omni_stream_test", good=good, bad=bad)
    evaluated: set[int] = set()

    def fake_eval(*args):
        flat = [a for arg in args for a in (arg if isinstance(arg, list) else [arg])]
        if any(a is bad for a in flat):
            raise ValueError("[eval] Attempting to eval an array without a primitive.")
        evaluated.update(id(a) for a in flat)

    monkeypatch.setattr(mx, "eval", fake_eval)

    _materialize_jang_tools_globals()

    assert id(good) in evaluated


def test_ignores_modules_outside_jang_tools(monkeypatch, evaluated):
    other = mx.ones((4,))
    _module(monkeypatch, "not_jang_tools_stream_test", cache=other)
    _module(monkeypatch, "jang_toolsish_stream_test", cache=other)

    _materialize_jang_tools_globals()

    assert id(other) not in evaluated
